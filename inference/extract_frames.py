import argparse
from pathlib import Path

import cv2


def extract_frames(
    video_path: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> None:
    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    existing_frames = sorted(
        output_dir.glob("frame_*.png")
    )

    if existing_frames and not overwrite:
        raise FileExistsError(
            f"{len(existing_frames)} frames already exist in "
            f"{output_dir}. Use --overwrite to replace them."
        )

    if overwrite:
        for frame_path in existing_frames:
            frame_path.unlink()

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Failed to open video: {video_path}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    frame_index = 0

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            frame_path = (
                output_dir
                / f"frame_{frame_index:06d}.png"
            )

            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(
                    f"Failed to save frame: {frame_path}"
                )

            frame_index += 1

    finally:
        capture.release()

    if frame_index == 0:
        raise RuntimeError(
            f"No frames were extracted from {video_path}"
        )

    duration = (
        frame_index / fps
        if fps > 0
        else float("nan")
    )

    print(f"[VIDEO] path: {video_path}")
    print(f"[VIDEO] resolution: {width}x{height}")
    print(f"[VIDEO] fps: {fps:.3f}")
    print(f"[VIDEO] duration: {duration:.3f} seconds")
    print(f"[VIDEO] reported frames: {reported_frames}")
    print(f"[VIDEO] extracted frames: {frame_index}")
    print(f"[VIDEO] output directory: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract all frames from a video."
    )

    parser.add_argument(
        "--video_path",
        type=Path,
        required=True,
        help="Path to the input video.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Frame output directory. "
            "Default: <video_stem>_frames"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing extracted frames.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir

    if output_dir is None:
        output_dir = (
            args.video_path.parent
            / f"{args.video_path.stem}_frames"
        )

    extract_frames(
        video_path=args.video_path,
        output_dir=output_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":

    # python inference/extract_frames.py --video_path outputs/honey_milk_5b_baseline.mp4 --overwrite
    # python inference/extract_frames.py --video_path outputs/titan_test.mp4 --overwrite

    # prompt/vjepa_test/light_physics_ok_frames/
    # ├── frame_000000.png
    # ├── frame_000001.png
    # ├── frame_000002.png
    # └── ...

    main()