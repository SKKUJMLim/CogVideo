"""TITAN-style, backpropagation-free V-JEPA 2 guidance for CogVideoX.

At a selected denoising step, each perturbed current latent is rolled out over
all remaining timesteps. V-JEPA energy is evaluated on the predicted final
video, and central directional finite differences estimate a correction for
the current latent. No video-generation computation graph is retained.
"""

import logging
import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoVideoProcessor


logger = logging.getLogger(__name__)


@dataclass
class PhysicsGuidanceConfig:
    scale: float = 0.0
    model_id: str = "facebook/vjepa2-vitl-fpc64-256"
    interval: int = 10
    start_step: int = 20
    end_step: int = 45
    num_frames: int = 16
    energy_epsilon: float = 0.01
    energy_num_directions: int = 1
    latent_epsilon: float = 0.05
    latent_num_directions: int = 1
    rollout_to_final: bool = True
    maximize: bool = False
    seed: int = 0
    device: str = "auto"


class VJEPAPhysicsGuidance:
    """Correct the current CogVideoX latent using predicted final-video energy."""

    def __init__(self, pipeline, config: PhysicsGuidanceConfig):
        self.pipeline = pipeline
        self.config = config

        if config.scale <= 0:
            raise ValueError("Physics guidance scale must be greater than zero.")
        if config.interval < 1:
            raise ValueError("Physics guidance interval must be at least one.")
        if config.energy_epsilon <= 0:
            raise ValueError("Energy epsilon must be greater than zero.")
        if config.energy_num_directions < 1:
            raise ValueError("Energy directions must be at least one.")
        if config.latent_epsilon <= 0:
            raise ValueError("Latent finite-difference epsilon must be greater than zero.")
        if config.latent_num_directions < 1:
            raise ValueError("Latent directions must be at least one.")

        if config.device == "auto":
            self.device = pipeline._execution_device
        else:
            self.device = torch.device(config.device)

        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.processor = AutoVideoProcessor.from_pretrained(config.model_id)
        self.model = AutoModel.from_pretrained(
            config.model_id,
            dtype=self.dtype,
            attn_implementation="sdpa",
        ).eval().to(self.device)
        self.model.requires_grad_(False)

        self.image_mean = torch.tensor(
            getattr(self.processor, "image_mean", [0.485, 0.456, 0.406]),
            device=self.device,
            dtype=self.dtype,
        ).view(1, 1, 3, 1, 1)
        self.image_std = torch.tensor(
            getattr(self.processor, "image_std", [0.229, 0.224, 0.225]),
            device=self.device,
            dtype=self.dtype,
        ).view(1, 1, 3, 1, 1)
        crop_size = getattr(self.processor, "crop_size", {"height": 256, "width": 256})
        if hasattr(crop_size, "height") and hasattr(crop_size, "width"):
            self.crop_size = (int(crop_size.height), int(crop_size.width))
        elif isinstance(crop_size, dict):
            self.crop_size = (int(crop_size["height"]), int(crop_size["width"]))
        elif isinstance(crop_size, (tuple, list)):
            self.crop_size = (int(crop_size[0]), int(crop_size[1]))
        else:
            self.crop_size = (int(crop_size), int(crop_size))

        self.generator = torch.Generator(device=self.device).manual_seed(config.seed)
        self.latent_generator = torch.Generator(
            device=pipeline._execution_device
        ).manual_seed(config.seed + 1)
        logger.info(
            "Loaded V-JEPA physics guide: model=%s, device=%s, scale=%g",
            config.model_id,
            self.device,
            config.scale,
        )

    def _should_apply(self, step: int) -> bool:
        return (
            self.config.start_step <= step <= self.config.end_step
            and (step - self.config.start_step) % self.config.interval == 0
        )

    def _preprocess(self, decoded_video: torch.Tensor) -> torch.Tensor:
        # CogVideoX VAE output: B,C,T,H,W in [-1, 1].
        video = (decoded_video.clamp(-1, 1) + 1) / 2
        frame_count = video.shape[2]
        sample_count = min(self.config.num_frames, frame_count)
        indices = torch.linspace(
            0, frame_count - 1, sample_count, device=video.device
        ).round().long()
        video = video.index_select(2, indices).permute(0, 2, 1, 3, 4)

        batch, frames, channels, height, width = video.shape
        video = F.interpolate(
            video.reshape(batch * frames, channels, height, width),
            size=self.crop_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).reshape(batch, frames, channels, *self.crop_size)
        video = video.to(device=self.device, dtype=self.dtype)
        return (video - self.image_mean) / self.image_std

    def _features(self, video: torch.Tensor) -> torch.Tensor:
        output = self.model(pixel_values_videos=video, skip_predictor=True)
        return output.last_hidden_state.float().mean(dim=1)

    @torch.no_grad()
    def _video_energy(self, final_latents: torch.Tensor, probes=None):
        """Return per-sample JEPA energy for a predicted final video."""
        decoded = self.pipeline.decode_latents(final_latents)
        video = self._preprocess(decoded)
        del decoded

        base_features = self._features(video)
        if probes is None:
            probes = []
            for _ in range(self.config.energy_num_directions):
                probe = torch.randn(
                    video.shape,
                    generator=self.generator,
                    device=video.device,
                    dtype=video.dtype,
                )
                probe_dims = tuple(range(1, probe.ndim))
                probe_rms = (
                    probe.float()
                    .square()
                    .mean(dim=probe_dims, keepdim=True)
                    .sqrt()
                    .clamp_min(1e-6)
                )
                probe = probe / probe_rms.to(probe.dtype)
                probes.append(probe)
        elif len(probes) != self.config.energy_num_directions:
            raise ValueError(
                f"Expected {self.config.energy_num_directions} probes, "
                f"received {len(probes)}."
            )

        energies = []
        for probe in probes:
            perturbed_features = self._features(
                video + self.config.energy_epsilon * probe
            )
            feature_jvp = (
                perturbed_features - base_features
            ) / self.config.energy_epsilon
            energies.append(feature_jvp.square().mean(dim=1))
            del perturbed_features, feature_jvp

        energy = torch.stack(energies, dim=0).mean(dim=0)
        del video, base_features, energies
        return energy, probes

    @torch.no_grad()
    def _rollout_to_final(
        self,
        latents: torch.Tensor,
        step: int,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Denoise ``latents`` over the timesteps remaining after ``step``.

        A fresh scheduler is used so the temporary rollout cannot alter the
        scheduler state of the main generation path. CogVideoX's callback runs
        after the scheduler step, hence ``latents`` is the main path's current
        z_(t-1), and rollout starts at the following scheduler timestep.
        """
        del negative_prompt_embeds  # prompt_embeds is already [negative, positive] under CFG.

        original_scheduler = self.pipeline.scheduler
        total_steps = len(original_scheduler.timesteps)
        if step + 1 >= total_steps:
            return latents

        scheduler_cls = type(original_scheduler)
        rollout_scheduler = scheduler_cls.from_config(original_scheduler.config)
        rollout_scheduler.set_timesteps(
            total_steps, device=latents.device
        )
        remaining = rollout_scheduler.timesteps[step + 1 :]
        height = latents.shape[-2] * self.pipeline.vae_scale_factor_spatial
        width = latents.shape[-1] * self.pipeline.vae_scale_factor_spatial
        image_rotary_emb = (
            self.pipeline._prepare_rotary_positional_embeddings(
                height,
                width,
                latents.size(1),
                latents.device,
            )
            if self.pipeline.transformer.config.use_rotary_positional_embeddings
            else None
        )
        ofs_emb = (
            None
            if self.pipeline.transformer.config.ofs_embed_dim is None
            else latents.new_full((1,), fill_value=2.0)
        )
        attention_kwargs = getattr(self.pipeline, "_attention_kwargs", None)
        do_cfg = prompt_embeds.shape[0] == 2 * latents.shape[0]
        extra_step_kwargs = self.pipeline.prepare_extra_step_kwargs(
            generator=None, eta=0.0
        )

        # cli_demo.py enables dynamic CFG. Recover its original CLI scale from
        # the scale at the callback step, then reproduce the same future scale.
        current_scale = float(self.pipeline.guidance_scale)
        current_timestep = float(original_scheduler.timesteps[step].item())
        dynamic_factor = (
            1
            - math.cos(
                math.pi
                * ((total_steps - current_timestep) / total_steps) ** 5.0
            )
        ) / 2
        base_guidance_scale = (
            (current_scale - 1) / dynamic_factor
            if do_cfg and dynamic_factor > 1e-8
            else current_scale
        )

        rollout_latents = latents.clone()
        old_pred_original_sample = None
        is_dpm = scheduler_cls.__name__ == "CogVideoXDPMScheduler"

        for rollout_index, rollout_timestep in enumerate(remaining):
            latent_model_input = (
                torch.cat([rollout_latents] * 2)
                if do_cfg
                else rollout_latents
            )
            latent_model_input = rollout_scheduler.scale_model_input(
                latent_model_input, rollout_timestep
            )
            timestep_batch = rollout_timestep.expand(
                latent_model_input.shape[0]
            )

            cache_context = (
                self.pipeline.transformer.cache_context("cond_uncond")
                if hasattr(self.pipeline.transformer, "cache_context")
                else nullcontext()
            )
            with cache_context:
                noise_pred = self.pipeline.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep_batch,
                    ofs=ofs_emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0].float()

            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                t_value = float(rollout_timestep.item())
                guidance_scale = 1 + base_guidance_scale * (
                    1
                    - math.cos(
                        math.pi
                        * ((total_steps - t_value) / total_steps) ** 5.0
                    )
                ) / 2
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            if is_dpm:
                previous_timestep = (
                    rollout_scheduler.timesteps[step]
                    if rollout_index == 0
                    else remaining[rollout_index - 1]
                )
                rollout_latents, old_pred_original_sample = (
                    rollout_scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        rollout_timestep,
                        previous_timestep,
                        rollout_latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                )
            else:
                rollout_latents = rollout_scheduler.step(
                    noise_pred,
                    rollout_timestep,
                    rollout_latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]
            rollout_latents = rollout_latents.to(prompt_embeds.dtype)

        return rollout_latents.detach()

    @torch.no_grad()
    def _objective(
        self,
        latents: torch.Tensor,
        step: int,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        probes=None,
    ):
        if self.config.rollout_to_final:
            final_latents = self._rollout_to_final(
                latents,
                step,
                prompt_embeds,
                negative_prompt_embeds,
            )
        else:
            final_latents = latents
        energy, probes = self._video_energy(final_latents, probes=probes)
        del final_latents
        return energy, probes

    def __call__(
        self, pipeline, step: int, timestep: torch.Tensor, callback_kwargs: Dict
    ) -> Dict:
        latents = callback_kwargs["latents"]
        if not self._should_apply(step):
            return callback_kwargs
        prompt_embeds = callback_kwargs["prompt_embeds"]
        negative_prompt_embeds = callback_kwargs["negative_prompt_embeds"]

        with torch.no_grad():
            latent_dims = tuple(range(1, latents.ndim))
            # Finite-difference derivatives can be much larger than the FP16
            # range. Keep the complete guidance calculation in FP32 and cast
            # only the final, validated latent back to the pipeline dtype.
            latents_fp32 = latents.float()
            gradient_estimate = torch.zeros_like(
                latents, dtype=torch.float32
            )
            derivatives = []
            positive_energies = []
            negative_energies = []
            energy_probes = None
            guidance_valid = True
            skip_reason = ""

            for _ in range(self.config.latent_num_directions):
                direction_fp32 = torch.randn(
                    latents.shape,
                    generator=self.latent_generator,
                    device=latents.device,
                    dtype=torch.float32,
                )
                direction_rms = (
                    direction_fp32
                    .square()
                    .mean(dim=latent_dims, keepdim=True)
                    .sqrt()
                    .clamp_min(1e-6)
                )
                direction_fp32 = direction_fp32 / direction_rms

                # The CogVideoX rollout still runs in the pipeline dtype, but
                # form each perturbation in FP32 before the final cast.
                positive_latents = (
                    latents_fp32
                    + self.config.latent_epsilon * direction_fp32
                ).to(latents.dtype)
                negative_latents = (
                    latents_fp32
                    - self.config.latent_epsilon * direction_fp32
                ).to(latents.dtype)

                positive_energy, energy_probes = self._objective(
                    positive_latents,
                    step,
                    prompt_embeds,
                    negative_prompt_embeds,
                    probes=energy_probes,
                )
                negative_energy, _ = self._objective(
                    negative_latents,
                    step,
                    prompt_embeds,
                    negative_prompt_embeds,
                    probes=energy_probes,
                )
                del positive_latents, negative_latents

                if not (
                    torch.isfinite(positive_energy).all()
                    and torch.isfinite(negative_energy).all()
                ):
                    guidance_valid = False
                    skip_reason = "non_finite_energy"
                    break

                derivative = (
                    positive_energy.float() - negative_energy.float()
                ) / (2 * self.config.latent_epsilon)
                if not torch.isfinite(derivative).all():
                    guidance_valid = False
                    skip_reason = "non_finite_derivative"
                    break

                objective_derivative = (
                    -derivative if self.config.maximize else derivative
                )
                gradient_estimate.add_(
                    objective_derivative.view(
                        -1, *([1] * (latents.ndim - 1))
                    )
                    * direction_fp32
                )
                derivatives.append(derivative)
                positive_energies.append(positive_energy)
                negative_energies.append(negative_energy)

            if guidance_valid:
                gradient_estimate.div_(self.config.latent_num_directions)
                gradient_rms = (
                    gradient_estimate.square()
                    .mean(dim=latent_dims, keepdim=True)
                    .sqrt()
                )
                if not torch.isfinite(gradient_rms).all():
                    guidance_valid = False
                    skip_reason = "non_finite_gradient_rms"
                elif not (gradient_rms > 1e-12).all():
                    guidance_valid = False
                    skip_reason = "zero_gradient"

            if guidance_valid:
                normalized_gradient = (
                    gradient_estimate / gradient_rms.clamp_min(1e-8)
                )
                latent_rms_per_sample = (
                    latents_fp32.square()
                    .mean(dim=latent_dims, keepdim=True)
                    .sqrt()
                )
                update_fp32 = (
                    self.config.scale
                    * latent_rms_per_sample
                    * normalized_gradient
                )
                guided_latents_fp32 = latents_fp32 - update_fp32

                if not (
                    torch.isfinite(update_fp32).all()
                    and torch.isfinite(guided_latents_fp32).all()
                ):
                    guidance_valid = False
                    skip_reason = "non_finite_update"

            if guidance_valid:
                guided_latents = guided_latents_fp32.to(latents.dtype)
                if torch.isfinite(guided_latents).all():
                    callback_kwargs["latents"] = guided_latents.detach()
                else:
                    guidance_valid = False
                    skip_reason = "cast_overflow"

            # A failed guidance evaluation must never contaminate the main
            # denoising trajectory. Leave callback_kwargs["latents"] unchanged.
            if not guidance_valid:
                logger.warning(
                    "Skipping physics guide step=%d timestep=%s reason=%s",
                    step,
                    timestep.item(),
                    skip_reason,
                )
                return callback_kwargs

            derivative_values = torch.stack(derivatives, dim=0)
            positive_energy_mean = torch.stack(
                positive_energies, dim=0
            ).float().mean()
            negative_energy_mean = torch.stack(
                negative_energies, dim=0
            ).float().mean()
            update_rms = update_fp32.square().mean().sqrt()
            latent_rms = latents_fp32.square().mean().sqrt()
            relative_update = update_rms / latent_rms.clamp_min(1e-8)

        logger.info(
            "Physics guide step=%d timestep=%s "
            "positive_energy=%.6g negative_energy=%.6g "
            "derivative_mean=%.6g derivative_abs_mean=%.6g "
            "gradient_rms=%.6g update_rms=%.6g relative_update=%.6g "
            "latent_directions=%d objective=%s mode=%s",
            step,
            timestep.item(),
            positive_energy_mean.item(),
            negative_energy_mean.item(),
            derivative_values.mean().item(),
            derivative_values.abs().mean().item(),
            gradient_rms.mean().item(),
            update_rms.item(),
            relative_update.item(),
            self.config.latent_num_directions,
            "final_rollout" if self.config.rollout_to_final else "current_decode",
            "maximize" if self.config.maximize else "minimize",
        )
        return callback_kwargs