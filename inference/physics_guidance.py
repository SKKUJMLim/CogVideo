"""Memory-efficient V-JEPA 2 guidance for CogVideoX inference.

The guide estimates V-JEPA Jacobian energy in pixel space, then estimates its
directional derivative in latent space using forward finite differences.
All VAE and V-JEPA evaluations run without autograd.
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

    # 영상 공간에서 JEPA Jacobian energy를 측정할 때 사용하는 epsilon
    energy_epsilon: float = 0.01
    energy_num_directions: int = 1

    # Latent 공간에서 energy 방향미분을 측정할 때 사용하는 epsilon
    latent_epsilon: float = 0.05

    # False: energy 최소화 / True: energy 최대화
    maximize: bool = False

    seed: int = 0
    device: str = "auto"


class VJEPAPhysicsGuidance:
    """Correct CogVideoX latents using V-JEPA feature sensitivity."""

    def __init__(
        self,
        pipeline,
        config: PhysicsGuidanceConfig,
    ):
        self.pipeline = pipeline
        self.config = config

        if config.scale <= 0:
            raise ValueError(
                "Physics guidance scale must be greater than zero."
            )

        if config.interval < 1:
            raise ValueError(
                "Physics guidance interval must be at least one."
            )

        if config.energy_epsilon <= 0:
            raise ValueError(
                "Energy epsilon must be greater than zero."
            )

        if config.energy_num_directions < 1:
            raise ValueError(
                "Energy directions must be at least one."
            )

        if config.latent_epsilon <= 0:
            raise ValueError(
                "Latent finite-difference epsilon must be greater than zero."
            )

        if config.device == "auto":
            self.device = pipeline._execution_device
        else:
            self.device = torch.device(config.device)

        if self.device.type == "cuda":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        self.processor = AutoVideoProcessor.from_pretrained(
            config.model_id
        )

        self.model = AutoModel.from_pretrained(
            config.model_id,
            torch_dtype=self.dtype,
            attn_implementation="sdpa",
        )

        self.model = self.model.eval().to(self.device)
        self.model.requires_grad_(False)

        self.image_mean = torch.tensor(
            getattr(
                self.processor,
                "image_mean",
                [0.485, 0.456, 0.406],
            ),
            device=self.device,
            dtype=self.dtype,
        ).view(1, 1, 3, 1, 1)

        self.image_std = torch.tensor(
            getattr(
                self.processor,
                "image_std",
                [0.229, 0.224, 0.225],
            ),
            device=self.device,
            dtype=self.dtype,
        ).view(1, 1, 3, 1, 1)

        # Transformers 버전에 따라 int, dict, tuple, SizeDict가 올 수 있음
        crop_size = getattr(
            self.processor,
            "crop_size",
            {"height": 256, "width": 256},
        )

        if (
            hasattr(crop_size, "height")
            and hasattr(crop_size, "width")
        ):
            # Transformers SizeDict
            self.crop_size = (
                int(crop_size.height),
                int(crop_size.width),
            )

        elif isinstance(crop_size, dict):
            self.crop_size = (
                int(crop_size["height"]),
                int(crop_size["width"]),
            )

        elif isinstance(crop_size, (tuple, list)):
            self.crop_size = (
                int(crop_size[0]),
                int(crop_size[1]),
            )

        else:
            size = int(crop_size)
            self.crop_size = (size, size)

        # JEPA pixel-space perturbation 생성기
        self.generator = torch.Generator(
            device=self.device
        ).manual_seed(config.seed)

        # CogVideo latent-space perturbation 생성기
        self.latent_generator = torch.Generator(
            device=pipeline._execution_device
        ).manual_seed(config.seed + 1)

        logger.info(
            "Loaded V-JEPA physics guide: "
            "model=%s, device=%s, scale=%g",
            config.model_id,
            self.device,
            config.scale,
        )

    def _should_apply(self, step: int) -> bool:
        """현재 denoising step에서 guidance를 적용할지 결정한다."""

        return (
            self.config.start_step
            <= step
            <= self.config.end_step
            and (
                step - self.config.start_step
            ) % self.config.interval
            == 0
        )

    def _preprocess(
        self,
        decoded_video: torch.Tensor,
    ) -> torch.Tensor:
        """CogVideoX의 decoded video를 V-JEPA 입력 형식으로 변환한다.

        CogVideoX VAE output:
            B, C, T, H, W
            value range: [-1, 1]

        V-JEPA input:
            B, T, C, H, W
        """

        video = (
            decoded_video.clamp(-1, 1) + 1
        ) / 2

        frame_count = video.shape[2]

        sample_count = min(
            self.config.num_frames,
            frame_count,
        )

        indices = torch.linspace(
            0,
            frame_count - 1,
            sample_count,
            device=video.device,
        ).round().long()

        video = video.index_select(
            2,
            indices,
        )

        # B,C,T,H,W -> B,T,C,H,W
        video = video.permute(
            0,
            2,
            1,
            3,
            4,
        )

        batch, frames, channels, height, width = video.shape

        video = F.interpolate(
            video.reshape(
                batch * frames,
                channels,
                height,
                width,
            ),
            size=self.crop_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        video = video.reshape(
            batch,
            frames,
            channels,
            *self.crop_size,
        )

        video = video.to(
            device=self.device,
            dtype=self.dtype,
        )

        video = (
            video - self.image_mean
        ) / self.image_std

        return video

    def _features(
        self,
        video: torch.Tensor,
    ) -> torch.Tensor:
        """V-JEPA video feature를 추출한다."""

        output = self.model(
            pixel_values_videos=video,
            skip_predictor=True,
        )

        # Token dimension을 평균 내어 video-level feature로 변환
        features = (
            output.last_hidden_state
            .float()
            .mean(dim=1)
        )

        return features

    @torch.no_grad()
    def _energy(
        self,
        latents: torch.Tensor,
        probes=None,
    ):
        """JEPA Jacobian energy를 측정한다.

        동일한 probes를 재사용하면 E(z)와 E(z + delta*v)를
        동일한 pixel perturbation 조건에서 비교할 수 있다.
        """

        decoded = self.pipeline.decode_latents(
            latents
        )

        video = self._preprocess(
            decoded
        )

        del decoded

        base_features = self._features(
            video
        )

        if probes is None:
            probes = []

            for _ in range(
                self.config.energy_num_directions
            ):
                probe = torch.randn(
                    video.shape,
                    generator=self.generator,
                    device=video.device,
                    dtype=video.dtype,
                )

                probe_norm = (
                    probe
                    .flatten(1)
                    .norm(dim=1)
                    .clamp_min(1e-6)
                )

                probe = probe / probe_norm.view(
                    -1,
                    *([1] * (probe.ndim - 1)),
                ).to(probe.dtype)

                probes.append(probe)

        elif (
            len(probes)
            != self.config.energy_num_directions
        ):
            raise ValueError(
                f"Expected "
                f"{self.config.energy_num_directions} probes, "
                f"received {len(probes)}."
            )

        energies = []

        for probe in probes:
            perturbed_video = (
                video
                + self.config.energy_epsilon
                * probe
            )

            perturbed_features = self._features(
                perturbed_video
            )

            feature_jvp = (
                perturbed_features
                - base_features
            ) / self.config.energy_epsilon

            energy = (
                feature_jvp
                .square()
                .mean(dim=1)
            )

            energies.append(energy)

            del perturbed_video
            del perturbed_features
            del feature_jvp

        energy = torch.stack(
            energies,
            dim=0,
        ).mean(dim=0)

        del video
        del base_features
        del energies

        return energy, probes

    def __call__(
        self,
        pipeline,
        step: int,
        timestep: torch.Tensor,
        callback_kwargs: Dict,
    ) -> Dict:
        """Diffusers callback_on_step_end entry point."""

        latents = callback_kwargs["latents"]

        if not self._should_apply(step):
            return callback_kwargs

        with torch.no_grad():
            # Latent-space random direction V
            direction = torch.randn(
                latents.shape,
                generator=self.latent_generator,
                device=latents.device,
                dtype=latents.dtype,
            )

            direction_norm = (
                direction
                .flatten(1)
                .norm(dim=1)
                .clamp_min(1e-8)
            )

            direction = direction / direction_norm.view(
                -1,
                *([1] * (direction.ndim - 1)),
            ).to(direction.dtype)

            # E(z)
            base_energy, energy_probes = self._energy(
                latents
            )

            # z + delta*V
            perturbed_latents = (
                latents
                + self.config.latent_epsilon
                * direction
            )

            # E(z + delta*V)
            # 동일한 JEPA probes를 공유한다.
            perturbed_energy, _ = self._energy(
                perturbed_latents,
                probes=energy_probes,
            )

            # D_V E(z)
            derivative = (
                perturbed_energy
                - base_energy
            ) / self.config.latent_epsilon

            if self.config.maximize:
                objective_derivative = -derivative
            else:
                objective_derivative = derivative

            # Rank-1 projected gradient:
            # grad(E) ≈ D_V E * V
            projected_gradient = (
                objective_derivative.view(
                    -1,
                    *([1] * (direction.ndim - 1)),
                )
                .to(direction.dtype)
                * direction
            )

            update = (
                self.config.scale
                * projected_gradient
            )

            callback_kwargs["latents"] = (
                latents - update
            ).detach()

        logger.info(
            "Physics guide "
            "step=%d "
            "timestep=%s "
            "energy=%.6g "
            "perturbed_energy=%.6g "
            "derivative=%.6g "
            "update_norm=%.6g "
            "mode=%s",
            step,
            timestep.item(),
            base_energy.mean().item(),
            perturbed_energy.mean().item(),
            derivative.mean().item(),
            update.flatten(1).norm(dim=1).mean().item(),
            (
                "maximize"
                if self.config.maximize
                else "minimize"
            ),
        )

        return callback_kwargs