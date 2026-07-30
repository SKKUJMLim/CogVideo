#!/usr/bin/env python3
"""Standalone V-JEPA2 spatiotemporal-token analyzer.

This script is intentionally independent of CogVideo.  It:
  1. samples frames from one video,
  2. extracts dense V-JEPA2 encoder tokens,
  3. verifies and restores the [T, H, W] token grid,
  4. computes temporal/spatial token-change maps, and
  5. saves a compact report and optional raw features.

Example:
    python analyze_vjepa2_tokens.py \
        --video outputs/current_decode.mp4 \
        --output_dir outputs/vjepa2_analysis/current_decode
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_MODEL = "facebook/vjepa2-vitl-fpc64-256"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the dense spatiotemporal tokens of a V-JEPA2 video encoder."
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video path.")
    parser.add_argument(
        "--model_path",
        default=DEFAULT_MODEL,
        help="Hugging Face model id or local model directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/vjepa2_analysis"),
        help="Directory for JSON/NPZ results.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of uniformly sampled input frames.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float16" if torch.cuda.is_available() else "float32",
    )
    parser.add_argument(
        "--save_features",
        action="store_true",
        help="Also save the full [T,H,W,D] feature tensor (can be large).",
    )
    parser.add_argument(
        "--topk_ratio",
        type=float,
        default=0.05,
        help="Fraction of highest-energy tokens summarized in the report.",
    )
    return parser.parse_args()


def sample_video(path: Path, num_frames: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Return RGB uint8 frames as [T,H,W,C], using uniform frame indices."""
    try:
        from decord import VideoReader, cpu
    except ImportError as exc:
        raise RuntimeError(
            "Video loading requires decord. Install it with: pip install decord"
        ) from exc

    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")
    if num_frames <= 0:
        raise ValueError("--num_frames must be positive.")

    reader = VideoReader(str(path), ctx=cpu(0))
    total_frames = len(reader)
    if total_frames == 0:
        raise ValueError(f"Video contains no readable frames: {path}")

    indices = np.linspace(0, total_frames - 1, num_frames).round().astype(np.int64)
    frames = reader.get_batch(indices).asnumpy()
    metadata = {
        "source_total_frames": int(total_frames),
        "source_fps": float(reader.get_avg_fps()),
        "sampled_indices": indices.tolist(),
        "decoded_shape_thwc": list(frames.shape),
    }
    return frames, metadata


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def scalar_config(config: Any, name: str) -> int:
    value = getattr(config, name, None)
    if value is None:
        raise AttributeError(f"Model config has no '{name}' field.")
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError(f"Model config field '{name}' is empty.")
        if len(set(value)) != 1:
            raise ValueError(
                f"Expected a scalar/equal-valued '{name}', but received {value}."
            )
        value = value[0]
    return int(value)


def resolve_tokens(outputs: Any) -> torch.Tensor:
    """Resolve the encoder's dense token tensor without silently pooling it."""
    tokens = getattr(outputs, "last_hidden_state", None)
    if tokens is None and isinstance(outputs, (tuple, list)) and outputs:
        tokens = outputs[0]
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
        shape = getattr(tokens, "shape", None)
        raise RuntimeError(
            "Expected dense tokens with shape [B,N,D] in last_hidden_state; "
            f"received {shape!r}."
        )
    return tokens


def restore_token_grid(
    tokens: torch.Tensor,
    pixel_values: torch.Tensor,
    patch_size: int,
    tubelet_size: int,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Restore flattened [B,N,D] tokens to [B,T,H,W,D]."""
    if pixel_values.ndim != 5:
        raise ValueError(
            f"Expected processor pixel_values [B,T,C,H,W], got {pixel_values.shape}."
        )

    _, frames, _, height, width = pixel_values.shape
    t_grid = frames // tubelet_size
    h_grid = height // patch_size
    w_grid = width // patch_size
    expected = t_grid * h_grid * w_grid
    actual = tokens.shape[1]

    # V-JEPA2 has no CLS token. Do not trim or guess if a model variant disagrees.
    if actual != expected:
        raise RuntimeError(
            "Token-grid mismatch: "
            f"N={actual}, but floor(T/{tubelet_size}) * floor(H/{patch_size}) "
            f"* floor(W/{patch_size}) = {expected} "
            f"({t_grid}*{h_grid}*{w_grid})."
        )

    batch, _, dim = tokens.shape
    grid = tokens.reshape(batch, t_grid, h_grid, w_grid, dim)
    return grid, (t_grid, h_grid, w_grid)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a.float(), dim=-1)
    b = torch.nn.functional.normalize(b.float(), dim=-1)
    return 1.0 - (a * b).sum(dim=-1)


def compute_maps(grid: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute maps while preserving every available time/space index."""
    if grid.shape[0] != 1:
        raise ValueError("This analysis script currently expects batch size 1.")
    x = grid[0]

    maps: dict[str, torch.Tensor] = {
        "token_l2_norm": x.float().norm(dim=-1),
    }
    if x.shape[0] >= 2:
        maps["temporal_cosine_distance"] = cosine_distance(x[1:], x[:-1])
        maps["temporal_l2_distance"] = (x[1:].float() - x[:-1].float()).norm(
            dim=-1
        )
    if x.shape[0] >= 3:
        maps["temporal_curvature_l2"] = (
            x[2:].float() - 2.0 * x[1:-1].float() + x[:-2].float()
        ).norm(dim=-1)
    if x.shape[2] >= 2:
        maps["spatial_x_cosine_distance"] = cosine_distance(x[:, :, 1:], x[:, :, :-1])
    if x.shape[1] >= 2:
        maps["spatial_y_cosine_distance"] = cosine_distance(x[:, 1:], x[:, :-1])
    return maps


def summarize_map(values: torch.Tensor, topk_ratio: float) -> dict[str, Any]:
    flat = values.detach().float().cpu().flatten()
    count = flat.numel()
    k = max(1, min(count, math.ceil(count * topk_ratio)))
    top_values, top_indices = torch.topk(flat, k=k)
    return {
        "shape": list(values.shape),
        "count": count,
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "topk_ratio": topk_ratio,
        "topk_count": k,
        "topk_mean": float(top_values.mean()),
        "topk_flat_indices": top_indices[: min(k, 20)].tolist(),
        "topk_values": top_values[: min(k, 20)].tolist(),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 < args.topk_ratio <= 1.0:
        raise ValueError("--topk_ratio must be in the interval (0, 1].")
    if args.device.startswith("cpu") and args.dtype != "float32":
        raise ValueError("Use --dtype float32 on CPU.")

    try:
        from transformers import AutoModel, AutoVideoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "A recent Transformers version with V-JEPA2 support is required. "
            "Install/upgrade with: pip install -U transformers"
        ) from exc

    frames, video_metadata = sample_video(args.video, args.num_frames)
    processor = AutoVideoProcessor.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype(args.dtype),
    ).to(args.device)
    model.eval()

    inputs = processor(list(frames), return_tensors="pt")
    if "pixel_values_videos" in inputs:
        pixel_key = "pixel_values_videos"
    elif "pixel_values" in inputs:
        pixel_key = "pixel_values"
    else:
        raise KeyError(
            "Video processor returned neither 'pixel_values_videos' nor 'pixel_values'."
        )

    pixel_values = inputs[pixel_key]
    # Some processor versions use [B,C,T,H,W]; the V-JEPA2 API expects [B,T,C,H,W].
    if pixel_values.ndim != 5:
        raise RuntimeError(f"Unexpected processor output shape: {pixel_values.shape}")
    if pixel_values.shape[1] == 3 and pixel_values.shape[2] != 3:
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4).contiguous()
        inputs[pixel_key] = pixel_values

    inputs = {
        key: value.to(args.device)
        if not torch.is_floating_point(value)
        else value.to(args.device, dtype=torch_dtype(args.dtype))
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)
    tokens = resolve_tokens(outputs)

    patch_size = scalar_config(model.config, "patch_size")
    tubelet_size = scalar_config(model.config, "tubelet_size")
    grid, grid_shape = restore_token_grid(
        tokens=tokens,
        pixel_values=inputs[pixel_key],
        patch_size=patch_size,
        tubelet_size=tubelet_size,
    )
    maps = compute_maps(grid)

    report = {
        "video": str(args.video.resolve()),
        "model_path": args.model_path,
        "device": args.device,
        "dtype": args.dtype,
        "video_metadata": video_metadata,
        "model_config": {
            "patch_size": patch_size,
            "tubelet_size": tubelet_size,
            "hidden_size": int(getattr(model.config, "hidden_size", tokens.shape[-1])),
        },
        "processor_pixel_shape_btchw": list(inputs[pixel_key].shape),
        "flat_token_shape_bnd": list(tokens.shape),
        "token_grid_shape_bthwd": list(grid.shape),
        "token_grid_thw": list(grid_shape),
        "has_cls_token": False,
        "maps": {
            name: summarize_map(values, args.topk_ratio)
            for name, values in maps.items()
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    maps_path = args.output_dir / "token_maps.npz"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        maps_path,
        **{name: values.detach().float().cpu().numpy() for name, values in maps.items()},
    )
    if args.save_features:
        torch.save(
            grid.detach().to(device="cpu", dtype=torch.float16),
            args.output_dir / "token_features_bthwd.pt",
        )

    print(json.dumps(report, indent=2))
    print(f"\nSaved report: {report_path}")
    print(f"Saved token maps: {maps_path}")
    if args.save_features:
        print(f"Saved token features: {args.output_dir / 'token_features_bthwd.pt'}")


if __name__ == "__main__":
    main()

    """
    python inference/analyze_vjepa2_tokens.py 
       --video outputs/current_decode.mp4 
       --model_path facebook/vjepa2-vitl-fpc64-256 
       --num_frames 16 
       --device cuda 
       --dtype float16 
       --output_dir outputs/vjepa2_analysis/current_decode 
       --save_features
    """
