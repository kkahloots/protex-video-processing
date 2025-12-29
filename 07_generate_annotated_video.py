#!/usr/bin/env python
"""
Generate annotated video from processed frames and cleaned COCO annotations.
Shows the complete pipeline output as a video for easy review.
"""

import argparse
import json
from pathlib import Path
import cv2
from utils.config_loader import get_config


def load_coco(path: str):
    with open(path, "r") as f:
        return json.load(f)


def draw_annotations(frame, anns, cats_by_id):
    """Draw bounding boxes on frame."""
    for ann in anns:
        bbox = ann.get("bbox", [0, 0, 0, 0])
        x, y, w, h = map(int, bbox)

        cat_id = ann.get("category_id")
        label = cats_by_id.get(cat_id, str(cat_id))
        conf = ann.get("score", ann.get("confidence", 1.0))

        # Color coding: person=red, equipment=blue, other=green, low-conf=orange
        if label == "person":
            color = (0, 0, 255)  # Red
        elif label in ["car", "truck", "bus"]:
            color = (255, 0, 0)  # Blue
        else:
            color = (0, 255, 0)  # Green

        if conf < 0.6:
            color = (0, 165, 255)  # Orange for low confidence

        # Draw box
        thickness = 3 if label == "person" else 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        # Draw label
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - th - 4), (x + tw + 4, y), color, -1)
        cv2.putText(
            frame,
            text,
            (x + 2, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame


def generate_annotated_video(
    frames_dir: str,
    coco_path: str,
    output_path: str,
    fps: float = 2.0,
    include_empty: bool = False,
    verbose: bool = True,
):
    """Generate video from annotated frames."""

    if verbose:
        print(f"[STEP] Loading COCO annotations from: {coco_path}")

    coco = load_coco(coco_path)
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])

    # Load frames metadata to get all processed frames
    frames_path = Path(frames_dir)
    metadata_path = frames_path / "frames_metadata.json"

    if include_empty and metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        images = metadata.get("frames", [])
        if verbose:
            print(
                f"[INFO] Using all {len(images)} processed frames (include_empty=True)"
            )
    else:
        images = coco.get("images", [])
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            valid_filenames = {
                frame["file_name"] for frame in metadata.get("frames", [])
            }
            images = [img for img in images if img["file_name"] in valid_filenames]
            if verbose:
                print(f"[INFO] Filtered to {len(images)} frames with detections")
        else:
            if verbose:
                print(f"[WARN] frames_metadata.json not found, using all COCO images")

    # Build indices
    cats_by_id = {c["id"]: c.get("name", f"id_{c['id']}") for c in cats}
    anns_by_img = {}
    for ann in anns:
        img_id = ann["image_id"]
        anns_by_img.setdefault(img_id, []).append(ann)

    if verbose:
        print(f"[INFO] Found {len(images)} images, {len(anns)} annotations")

    # Get first frame to determine video size
    frames_path = Path(frames_dir)
    first_img = images[0] if images else None
    if not first_img:
        raise RuntimeError("No images in COCO file")

    first_frame_path = frames_path / first_img["file_name"]
    sample_frame = cv2.imread(str(first_frame_path))
    if sample_frame is None:
        raise RuntimeError(f"Cannot read frame: {first_frame_path}")

    height, width = sample_frame.shape[:2]

    # Initialize video writer with H.264 codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if verbose:
        print(f"[STEP] Generating video: {width}x{height} @ {fps} FPS")

    processed = 0
    for img_info in images:
        frame_path = frames_path / img_info["file_name"]
        if not frame_path.exists():
            if verbose:
                print(f"[WARN] Frame not found: {frame_path}")
            continue

        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        # Draw annotations
        img_id = img_info["id"]
        img_anns = anns_by_img.get(img_id, [])
        if img_anns:
            frame = draw_annotations(frame, img_anns, cats_by_id)

        out.write(frame)
        processed += 1

        if verbose and processed % 10 == 0:
            print(f"[INFO] Processed {processed}/{len(images)} frames")

    out.release()

    # Convert to proper MP4 using ffmpeg if available
    temp_path = output_path.replace(".mp4", "_temp.mp4")
    Path(output_path).rename(temp_path)

    import subprocess

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                temp_path,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-y",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
        Path(temp_path).unlink()
    except (subprocess.CalledProcessError, FileNotFoundError):
        Path(temp_path).rename(output_path)

    if verbose:
        duration = processed / fps
        print(f"\n[SUMMARY]")
        print(f"  Frames processed: {processed}")
        print(f"  Video duration:   {duration:.1f} seconds")
        print(f"  Output:           {output_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Generate annotated video")
    parser.add_argument(
        "--frames_dir",
        type=str,
        default=get_config("samples.images_dir", "traceables/frames"),
        help="Directory with frames",
    )
    parser.add_argument(
        "--coco_path",
        type=str,
        default=get_config(
            "cleanup.output_json", "traceables/pre_tags/pre_tags_cleaned.json"
        ),
        help="Cleaned COCO JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="traceables/sample_annotated_video.mp4",
        help="Output video path",
    )
    parser.add_argument("--fps", type=float, default=2.0, help="Output video FPS")
    parser.add_argument(
        "--include_empty", action="store_true", help="Include frames without detections"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args, _ = parser.parse_known_args()

    verbose = get_config("defaults.verbose", True)
    if args.verbose:
        verbose = True
    if args.quiet:
        verbose = False

    if verbose:
        print("=" * 60)
        print(" Protex AI – Stage 7: Annotated Video Generation")
        print("=" * 60)
        print(f"[CONFIG]")
        print(f"  Frames dir:  {args.frames_dir}")
        print(f"  COCO path:   {args.coco_path}")
        print(f"  Output:      {args.output}")
        print(f"  FPS:         {args.fps}")
        print("=" * 60)
        print()

    success = generate_annotated_video(
        args.frames_dir,
        args.coco_path,
        args.output,
        args.fps,
        args.include_empty,
        verbose,
    )

    if verbose:
        print("\n✅ Annotated video generation completed")
        print("=" * 60)


if __name__ == "__main__":
    main()
