"""V-JEPA 2 guidance for CogVideoX inference.

The guide minimizes a finite-difference approximation of the local Jacobian
energy of V-JEPA 2 features with respect to the decoded video.  It is designed
to be used as a ``callback_on_step_end`` callback in Diffusers.
"""

import logging
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
    fd_epsilon: float = 0.01
    seed: int = 0
    device: str = "auto"


class VJEPAPhysicsGuidance:
    """Correct CogVideoX latents using V-JEPA feature sensitivity."""

    def __init__(self, pipeline, config: PhysicsGuidanceConfig):
        self.pipeline = pipeline
        self.config = config

        if config.scale <= 0:
            raise ValueError("Physics guidance scale must be greater than zero.")
        if config.interval < 1:
            raise ValueError("Physics guidance interval must be at least one.")
        if config.fd_epsilon <= 0:
            raise ValueError("Finite-difference epsilon must be greater than zero.")

        if config.device == "auto":
            self.device = pipeline._execution_device
        else:
            self.device = torch.device(config.device)

        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.processor = AutoVideoProcessor.from_pretrained(config.model_id)
        self.model = AutoModel.from_pretrained(
            config.model_id,
            torch_dtype=self.dtype,
            attn_implementation="sdpa",
        ).eval().to(self.device)
        self.model.requires_grad_(False)

        # Backpropagating through a video ViT is expensive. Checkpointing makes
        # the guide practical on a single GPU at the cost of extra compute.
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

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
        if isinstance(crop_size, dict):
            self.crop_size = (crop_size["height"], crop_size["width"])
        else:
            self.crop_size = (int(crop_size), int(crop_size))

        self.generator = torch.Generator(device=self.device).manual_seed(config.seed)
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
        return output.last_hidden_state

    def __call__(
        self, pipeline, step: int, timestep: torch.Tensor, callback_kwargs: Dict
    ) -> Dict:
        latents = callback_kwargs["latents"]
        if not self._should_apply(step):
            return callback_kwargs

        with torch.enable_grad():
            guided_latents = latents.detach().requires_grad_(True)
            decoded = pipeline.decode_latents(guided_latents)
            video = self._preprocess(decoded)

            direction = torch.randn(
                video.shape,
                generator=self.generator,
                device=video.device,
                dtype=video.dtype,
            )
            direction = direction / direction.square().mean().sqrt().clamp_min(1e-6)
            perturbed = video + self.config.fd_epsilon * direction

            features = self._features(video)
            perturbed_features = self._features(perturbed)
            energy = (
                (perturbed_features - features).float().square().mean()
                / self.config.fd_epsilon**2
            )
            gradient = torch.autograd.grad(energy, guided_latents)[0]

        # RMS normalization makes the scale interpretable across timesteps.
        dims = tuple(range(1, gradient.ndim))
        gradient_rms = gradient.float().square().mean(dim=dims, keepdim=True).sqrt()
        normalized_gradient = gradient / gradient_rms.to(gradient.dtype).clamp_min(1e-6)
        latent_scale = latents.float().std(dim=dims, keepdim=True).to(latents.dtype)
        callback_kwargs["latents"] = (
            latents - self.config.scale * latent_scale * normalized_gradient
        ).detach()

        logger.info(
            "Physics guide step=%d timestep=%s energy=%.6g grad_rms=%.6g",
            step,
            timestep.item(),
            energy.item(),
            gradient_rms.mean().item(),
        )
        return callback_kwargs
