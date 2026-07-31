"""Generate a resumable VideoPhy baseline/current-decode experiment.

The model is loaded once and reused for every prompt.  The shared metadata file
keeps prompt/seed pairs stable so a later current-decode run is directly paired
with the baseline run.
"""

import argparse
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline
from diffusers.utils import export_to_video

from physics_guidance import PhysicsGuidanceConfig, VJEPAPhysicsGuidance


LOGGER = logging.getLogger("videophy_batch")
MODES = ("baseline", "current_decode")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-generate paired VideoPhy videos with CogVideoX."
    )
    parser.add_argument("--prompts_path", required=True)
    parser.add_argument("--output_root", default="outputs/videophy")
    parser.add_argument("--mode", choices=MODES, default="baseline")
    parser.add_argument("--model_path", default="zai-org/CogVideoX-5b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument(
        "--overwrite", action="store_true", help="Regenerate existing non-empty MP4 files."
    )

    # Used only by --mode current_decode. Baseline always disables physics guidance.
    parser.add_argument("--physics_guidance_scale", type=float, default=0.01)
    parser.add_argument(
        "--physics_guidance_model",
        default="facebook/vjepa2-vitl-fpc64-256",
    )
    parser.add_argument("--physics_guidance_start_step", type=int, default=0)
    parser.add_argument("--physics_guidance_end_step", type=int, default=49)
    parser.add_argument("--physics_guidance_interval", type=int, default=1)
    parser.add_argument("--physics_guidance_frames", type=int, default=16)
    parser.add_argument("--physics_guidance_epsilon", type=float, default=0.01)
    parser.add_argument("--physics_guidance_directions", type=int, default=1)
    parser.add_argument(
        "--physics_guidance_token_aggregation",
        choices=("pooled", "token_mean", "token_topk", "motion_topk"),
        default="pooled",
    )
    parser.add_argument(
        "--physics_guidance_token_topk_ratio", type=float, default=0.10
    )
    parser.add_argument("--physics_guidance_motion_ratio", type=float, default=0.25)
    parser.add_argument("--physics_guidance_latent_epsilon", type=float, default=0.05)
    parser.add_argument("--physics_guidance_latent_directions", type=int, default=1)
    parser.add_argument("--physics_guidance_device", default="cuda")
    return parser.parse_args()


def read_prompts(path):
    prompts = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def prompt_digest(prompts):
    payload = "\n".join(prompts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_jsonl(path, rows):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp_path, path)


def prepare_metadata(args, prompts, output_root):
    metadata_path = output_root / "metadata.jsonl"
    expected = [
        {
            "index": index,
            "prompt": prompt,
            "seed": args.seed,
            "baseline_path": f"baseline/{index:04d}.mp4",
            "current_decode_path": f"current_decode/{index:04d}.mp4",
        }
        for index, prompt in enumerate(prompts)
    ]

    if metadata_path.exists():
        existing = [
            json.loads(line)
            for line in metadata_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        existing_pairs = [
            (row.get("index"), row.get("prompt"), row.get("seed")) for row in existing
        ]
        expected_pairs = [
            (row["index"], row["prompt"], row["seed"]) for row in expected
        ]
        if existing_pairs != expected_pairs:
            raise ValueError(
                f"{metadata_path} does not match the current prompt order/seed. "
                "Use the original prompt file and seed, or choose a new output_root."
            )
        return existing

    atomic_write_jsonl(metadata_path, expected)
    experiment = {
        "prompt_count": len(prompts),
        "prompt_sha256": prompt_digest(prompts),
        "seed_policy": "fixed_seed_for_every_prompt",
        "seed": args.seed,
        "model_path": args.model_path,
        "num_inference_steps": args.num_inference_steps,
        "num_frames": args.num_frames,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "fps": args.fps,
        "dtype": args.dtype,
        "physics_guidance_token_aggregation": (
            args.physics_guidance_token_aggregation
        ),
        "physics_guidance_token_topk_ratio": (
            args.physics_guidance_token_topk_ratio
        ),
        "physics_guidance_motion_ratio": args.physics_guidance_motion_ratio,
    }
    (output_root / "experiment.json").write_text(
        json.dumps(experiment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return expected


def append_progress(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_pipeline(args, dtype):
    pipe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe


def build_physics_guidance(args, pipe, seed):
    if args.mode == "baseline":
        return None
    if args.physics_guidance_scale <= 0:
        raise ValueError("current_decode requires --physics_guidance_scale > 0")
    return VJEPAPhysicsGuidance(
        pipe,
        PhysicsGuidanceConfig(
            scale=args.physics_guidance_scale,
            model_id=args.physics_guidance_model,
            interval=args.physics_guidance_interval,
            start_step=args.physics_guidance_start_step,
            end_step=args.physics_guidance_end_step,
            num_frames=args.physics_guidance_frames,
            energy_epsilon=args.physics_guidance_epsilon,
            energy_num_directions=args.physics_guidance_directions,
            token_aggregation=args.physics_guidance_token_aggregation,
            token_topk_ratio=args.physics_guidance_token_topk_ratio,
            motion_ratio=args.physics_guidance_motion_ratio,
            latent_epsilon=args.physics_guidance_latent_epsilon,
            latent_num_directions=args.physics_guidance_latent_directions,
            rollout_to_final=False,
            maximize=False,
            seed=seed,
            device=args.physics_guidance_device,
        ),
    )


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    prompts = read_prompts(args.prompts_path)
    if args.end_index is None:
        args.end_index = len(prompts)
    if not 0 <= args.start_index < args.end_index <= len(prompts):
        raise ValueError(
            f"Expected 0 <= start_index < end_index <= {len(prompts)}, got "
            f"{args.start_index}, {args.end_index}"
        )

    output_root = Path(args.output_root)
    mode_dir = output_root / args.mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    metadata = prepare_metadata(args, prompts, output_root)
    selected = metadata[args.start_index : args.end_index]
    pending = []
    for row in selected:
        output_path = output_root / row[f"{args.mode}_path"]
        if args.overwrite or not output_path.is_file() or output_path.stat().st_size == 0:
            pending.append(row)

    LOGGER.info(
        "Mode=%s, prompts=%d, selected=%d, pending=%d",
        args.mode,
        len(prompts),
        len(selected),
        len(pending),
    )
    if not pending:
        LOGGER.info("Nothing to generate.")
        return

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    pipe = load_pipeline(args, dtype)
    # The experiment uses one fixed seed, so the optional V-JEPA guide can also
    # be loaded once and reused instead of reloading its encoder per prompt.
    physics_guidance = build_physics_guidance(args, pipe, args.seed)
    callback_inputs = (
        ["latents", "prompt_embeds", "negative_prompt_embeds"]
        if physics_guidance is not None
        else ["latents"]
    )
    progress_path = output_root / "progress.jsonl"

    for position, row in enumerate(pending, start=1):
        index, prompt, seed = row["index"], row["prompt"], row["seed"]
        output_path = output_root / row[f"{args.mode}_path"]
        started = time.monotonic()
        LOGGER.info("[%d/%d] index=%04d seed=%d", position, len(pending), index, seed)
        try:
            frames = pipe(
                height=args.height,
                width=args.width,
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=args.num_inference_steps,
                num_frames=args.num_frames,
                use_dynamic_cfg=True,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator().manual_seed(seed),
                callback_on_step_end=physics_guidance,
                callback_on_step_end_tensor_inputs=callback_inputs,
            ).frames[0]
            export_to_video(frames, str(output_path), fps=args.fps)
            append_progress(
                progress_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "mode": args.mode,
                    "index": index,
                    "seed": seed,
                    "status": "completed",
                    "output_path": str(output_path),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                },
            )
            del frames
            torch.cuda.empty_cache()
        except Exception as error:
            LOGGER.exception("Generation failed for index=%04d; continuing.", index)
            append_progress(
                progress_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "mode": args.mode,
                    "index": index,
                    "seed": seed,
                    "status": "failed",
                    "error": repr(error),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                },
            )


if __name__ == "__main__":
    main()