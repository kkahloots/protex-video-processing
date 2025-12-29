#!/usr/bin/env python
"""
===============================================================================
    Protex AI – Computer Vision Ops Pipeline
    -----------------------------------------------------------
    Module: data_preprocessing.py
    Stage: 1 / 4 (Data Preprocessing – Video → Frames)

    Purpose:
        This module performs the first step of the Computer Vision pipeline:
        transforming raw video footage from client cameras into a clean,
        structured set of image frames suitable for downstream processing
        (object detection, pre-tagging, human annotation, and model training).

        It supports multiple operational modes ("fast", "balanced", "accurate")
        so that we can trade off between speed, annotation cost, and fidelity.

    What This Script Does:
        ✓ Loads an input video (e.g., CCTV / timelapse / site camera)
        ✓ Extracts frames at a MODE-driven sampling rate
        ✓ Removes:
            • Black frames (camera glitches, night mode, transitions)
            • Duplicate frames (static scenes with no movement)
        ✓ Resizes frames to a standard resolution: 960×544 (configurable)
        ✓ Writes frames as JPEG images with stable naming
        ✓ Saves metadata (frames_metadata.json) with:
            – Original FPS
            – Sampled FPS
            – Total frames in the video
            – Number of frames kept
            – Per-frame info (id, filename, width, height)

    Why This Matters for Protex AI:
        Industrial safety footage often contains:
            • Long static intervals (no activity)
            • Dark/black segments (off-hours, camera glitches)
            • Highly repetitive timelapse content

        Removing non-informative frames reduces:
            → GPU inference load
            → Annotation effort & cost
            → Storage & bandwidth
            → Time-to-signal for safety teams

        This preprocessing is foundational for scaling Protex AI's deployments
        across many cameras and sites in 20+ countries.

    Typical Usage:
        Local (after running setup_env.py):
            $ python data_preprocessing.py --video_path client_video.mp4

        In Google Colab:
            !python data_preprocessing.py --video_path timelapse_test.mp4

    Operational Modes:
        MODE = "fast"
            - Very low sampling
            - Optimized for speed / low cost

        MODE = "balanced"   (default)
            - Good trade-off for annotation pipelines

        MODE = "accurate"
            - Higher sampling & stricter duplicate detection
            - Better temporal resolution, more frames per clip

    Dependencies:
        - Python 3.8+
        - opencv-python
        - numpy
        - (Optional, for GPU info logging) torch

        Installed via:
            $ python setup_env.py

    Outputs:
        /frames/
            frame_00000.jpg
            frame_00001.jpg
            ...
            frames_metadata.json

===============================================================================
"""


import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from utils.config_loader import get_config, get_mode_config
from utils.pipeline_tracker import save_stage_config

# Optional GPU
try:
    import torch

    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_DEVICE = "cuda" if GPU_AVAILABLE else "cpu"
except ImportError:
    GPU_AVAILABLE = False
    GPU_DEVICE = "cpu"


# =============================================================================
# MODE DEFAULTS
# =============================================================================


def get_mode_defaults(mode: str):
    mode = mode.lower()
    desired_fps = get_mode_config("preprocessing", mode, "desired_fps", 1.0)
    duplicate_threshold = get_mode_config(
        "preprocessing", mode, "duplicate_threshold", 1.0
    )
    min_brightness = get_mode_config("preprocessing", mode, "min_brightness", 30.0)
    min_laplacian_var = get_mode_config(
        "preprocessing", mode, "min_laplacian_var", 50.0
    )
    return desired_fps, duplicate_threshold, min_brightness, min_laplacian_var


def is_quality_acceptable(frame, min_brightness, min_laplacian_var):
    """
    Check if frame meets quality standards for detection.

    Filtering hierarchy:
    - Black frames (mean < 5): Camera glitches, off-hours, complete darkness
    - Dark frames (mean < 20-35): Too low SNR for reliable object detection
    - Blurry frames (Laplacian var < 40-60): Motion blur, out-of-focus
    - Noise frames: TV static with edges everywhere
    """
    small = cv2.resize(frame, (320, 180))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    if brightness < min_brightness:
        return False, "dark"

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < min_laplacian_var:
        return False, "blurry"

    # Noise detection: high edges + high entropy + uniform histogram
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float((edges > 0).mean())
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    hist = hist / (hist.sum() + 1e-8)
    entropy = float(-(hist * np.log2(hist + 1e-8)).sum())

    is_noise = lap_var > min_laplacian_var * 3.0 and edge_density > 0.20 and entropy > 4.0

    if is_noise:
        return False, "noise"

    return True, None


# =============================================================================
# FRAME EXTRACTION
# =============================================================================


def create_static_mask(width: int, height: int, mask_path: str = None):
    """
    Create or load a static mask to remove black polygons/regions.

    Args:
        width: Target frame width
        height: Target frame height
        mask_path: Optional path to pre-computed mask image (white=keep, black=remove)

    Returns:
        Binary mask (255=keep, 0=remove) or None if no masking needed
    """
    if mask_path and Path(mask_path).exists():
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height))
        return mask

    # Default: no masking (return None to skip mask operations)
    return None


def apply_mask(frame, mask):
    """
    Apply static mask to frame using bitwise AND.
    Masked regions become black (0, 0, 0).

    Args:
        frame: BGR image
        mask: Binary mask (255=keep, 0=remove)

    Returns:
        Masked frame
    """
    if mask is None:
        return frame
    return cv2.bitwise_and(frame, frame, mask=mask)


def extract_and_preprocess_frames(
    video_path: str,
    output_dir: str,
    desired_fps: float,
    target_width: int,
    target_height: int,
    duplicate_threshold: float,
    min_brightness: float,
    min_laplacian_var: float,
    mask_path: str = None,
    verbose: bool = True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if verbose:
        print("\n[STEP] Opening video stream...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        raise RuntimeError("Video FPS invalid.")

    frame_step = max(1, int(round(original_fps / desired_fps)))

    # Create static mask once (reused for all frames)
    static_mask = create_static_mask(target_width, target_height, mask_path)
    if verbose and static_mask is not None:
        print(f"[INFO] Static mask loaded: {target_width}x{target_height}")

    if verbose:
        print(f"[INFO] Original video FPS: {original_fps:.2f}")
        print(f"[INFO] Sampling every {frame_step} frame(s)")

    frames_metadata = []
    saved_count = 0
    frame_idx = 0
    prev_saved_frame = None

    num_black = 0  # Dark/black frames
    num_dup = 0
    num_blurry = 0
    num_noise = 0

    if verbose:
        print("\n[STEP] Beginning frame iteration...")

    while True:
        ret, frame = cap.read()
        if not ret:
            if verbose:
                print("[INFO] End of video reached.")
            break

        frame_idx += 1

        # --- Sampling check ---
        if (frame_idx - 1) % frame_step != 0:
            if verbose and frame_idx <= 5:
                print(f"[DEBUG] Skipping unsampled frame idx={frame_idx}")
            continue

        # --- Quality filter (brightness + sharpness) ---
        # Filters: black/dark frames (camera glitches, low SNR) + blurry frames (motion blur)
        quality_ok, reason = is_quality_acceptable(
            frame, min_brightness, min_laplacian_var
        )
        if not quality_ok:
            if reason == "dark":
                num_black += 1
                if verbose and num_black <= 3:
                    print(f"[DEBUG] Dark/black frame #{num_black} @ idx={frame_idx}")
            elif reason == "blurry":
                num_blurry += 1
                if verbose and num_blurry <= 3:
                    print(f"[DEBUG] Blurry frame #{num_blurry} @ idx={frame_idx}")
            elif reason == "noise":
                num_noise += 1
                if verbose and num_noise <= 3:
                    print(f"[DEBUG] Noise frame #{num_noise} @ idx={frame_idx}")
            continue

        # --- Duplicate frame filter ---
        if prev_saved_frame is not None:
            mad = float(np.mean(cv2.absdiff(frame, prev_saved_frame)))
            if mad < duplicate_threshold:
                num_dup += 1
                if verbose and num_dup <= 5:
                    print(
                        f"[DEBUG] Duplicate #{num_dup} @ idx={frame_idx}, MAD={mad:.4f}"
                    )
                continue

        # --- Resize and apply mask ---
        resized = cv2.resize(frame, (target_width, target_height))
        masked = apply_mask(resized, static_mask)

        filename = f"frame_{saved_count:05d}.jpg"
        cv2.imwrite(str(output_dir / filename), masked)

        if verbose and saved_count < 5:
            print(f"[WRITE] {filename} saved.")

        prev_saved_frame = frame.copy()

        frames_metadata.append(
            {
                "id": saved_count,
                "file_name": filename,
                "width": target_width,
                "height": target_height,
            }
        )
        saved_count += 1

    cap.release()

    if verbose:
        print("\n[SUMMARY]")
        print(f"  Total frames read:         {frame_idx}")
        print(f"  Saved frames:              {saved_count}")
        print(f"  Dark/black frames skipped: {num_black}")
        print(f"  Blurry frames skipped:     {num_blurry}")
        print(f"  Noise frames skipped:      {num_noise}")
        print(f"  Duplicate frames skipped:  {num_dup}")
        print("------------------------------------------------------------")

    return frames_metadata, original_fps, frame_idx


# =============================================================================
# METADATA
# =============================================================================


def save_metadata(
    metadata,
    original_fps,
    total_frames,
    desired_fps,
    output_dir,
    mode,
    w,
    h,
    dup_thr,
    min_bright,
    min_lap,
    verbose=True,
):

    payload = {
        "mode": mode,
        "original_fps": original_fps,
        "desired_fps": desired_fps,
        "target_width": w,
        "target_height": h,
        "duplicate_threshold": dup_thr,
        "min_brightness": min_bright,
        "min_laplacian_var": min_lap,
        "total_video_frames": total_frames,
        "num_saved_frames": len(metadata),
        "frames": metadata,
    }

    output_dir = Path(output_dir)
    with open(output_dir / "frames_metadata.json", "w") as f:
        json.dump(payload, f, indent=2)

    if verbose:
        print("[WRITE] frames_metadata.json written.")


# =============================================================================
# CLEANUP
# =============================================================================


def cleanup_frames(output_dir: str, metadata: list, cleanup_mode: str, verbose: bool):
    if cleanup_mode == "off":
        return

    output_dir = Path(output_dir)
    keep = {m["file_name"] for m in metadata}

    removed = 0
    for f in output_dir.glob("frame_*.jpg"):
        if f.name not in keep:
            f.unlink()
            removed += 1

    if verbose:
        print(f"[CLEANUP] Removed {removed} frames ({cleanup_mode}).")


# =============================================================================
# ARG PARSER
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Protex AI – Data Preprocessing", allow_abbrev=False
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default=get_config("preprocessing.video_path", "data/timelapse_test.mp4"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=get_config("preprocessing.output_dir", "frames"),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=get_config("defaults.mode", "balanced"),
        choices=["fast", "balanced", "accurate"],
    )

    parser.add_argument(
        "--target_width",
        type=int,
        default=get_config("preprocessing.target_width", 960),
    )
    parser.add_argument(
        "--target_height",
        type=int,
        default=get_config("preprocessing.target_height", 544),
    )

    parser.add_argument(
        "--mask_path",
        type=str,
        default=get_config("preprocessing.mask_path", None),
        help="Path to static mask image (white=keep, black=remove)",
    )

    parser.add_argument("--desired_fps", type=float, default=None)
    parser.add_argument("--duplicate_threshold", type=float, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (future use)",
    )

    parser.add_argument(
        "--cleanup",
        type=str,
        default=get_config("defaults.cleanup_mode", "off"),
        choices=["off", "unused", "all"],
    )

    parser.add_argument(
        "--no-clean_traceables",
        action="store_false",
        dest="clean_traceables",
        default=get_config("defaults.clean_traceables", True),
        help="Skip cleaning traceables folder before processing",
    )

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[INFO] Ignoring Jupyter args: {unknown}")

    return args


# =============================================================================
# MAIN
# =============================================================================


def main():
    args = parse_args()

    verbose = get_config("defaults.verbose", True)
    if args.verbose:
        verbose = True
    if args.quiet:
        verbose = False

    # Clean traceables folder if requested
    if args.clean_traceables:
        import shutil
        runs_dir = Path("traceables")
        if runs_dir.exists():
            if verbose:
                print("[CLEANUP] Removing traceables folder...")
            shutil.rmtree(runs_dir)
            if verbose:
                print("[CLEANUP] Runs folder cleaned.")

    mode = args.mode.lower()
    mode_fps, mode_dup, mode_brightness, mode_lap = get_mode_defaults(mode)

    # Customer overrides
    desired_fps = args.desired_fps or mode_fps
    dup_thr = args.duplicate_threshold or mode_dup
    min_brightness = mode_brightness
    min_laplacian = mode_lap

    if verbose:
        print("\n============================================================")
        print(" Protex AI – Stage 1: Data Preprocessing")
        print("============================================================")
        print(f"[ENV] GPU available: {GPU_AVAILABLE} | Device: {GPU_DEVICE}")
        print("------------------------------------------------------------")
        print("[CONFIGURATION]")
        print(f"  Video Path:           {args.video_path}")
        print(f"  Output Directory:     {args.output_dir}")
        print(f"  Mode:                 {mode}")
        print(f"  Cleanup Mode:         {args.cleanup}")
        print("------------------------------------------------------------")
        print("[FRAME PARAMETERS]")
        print(f"  Target Resolution:    {args.target_width} x {args.target_height}")
        print(f"  Desired FPS:          {desired_fps}")
        print(f"  Min Brightness:       {min_brightness}")
        print(f"  Min Laplacian Var:    {min_laplacian}")
        print(f"  Duplicate Threshold:  {dup_thr}")
        print(f"  Random Seed:          {args.seed} (deterministic sampling)")
        print("============================================================\n")

    frames_metadata, original_fps, total_frames = extract_and_preprocess_frames(
        args.video_path,
        args.output_dir,
        desired_fps,
        args.target_width,
        args.target_height,
        dup_thr,
        min_brightness,
        min_laplacian,
        args.mask_path,
        verbose,
    )

    save_metadata(
        frames_metadata,
        original_fps,
        total_frames,
        desired_fps,
        args.output_dir,
        mode,
        args.target_width,
        args.target_height,
        dup_thr,
        min_brightness,
        min_laplacian,
        verbose,
    )

    # Save pipeline configuration for report traceability
    save_stage_config(
        "preprocessing",
        {
            "mode": mode,
            "desired_fps": desired_fps,
            "min_brightness": min_brightness,
            "min_laplacian_var": min_laplacian,
            "duplicate_threshold": dup_thr,
            "target_resolution": f"{args.target_width}x{args.target_height}",
        },
    )

    if verbose:
        print("[STEP] Cleanup...")
    cleanup_frames(args.output_dir, frames_metadata, args.cleanup, verbose)

    if verbose:
        print("\n✅ Preprocessing Complete.")
        print("============================================================\n")


if __name__ == "__main__":
    main()
