#!/usr/bin/env python
"""
===============================================================================
    Protex AI – Computer Vision Ops Pipeline
    -----------------------------------------------------------
    Module: pretag_cleanup.py
    Stage: 3 / 4 (Pre-tag Cleanup – COCO → cleaned COCO)

    Purpose:
        This module cleans up the raw pre-tag COCO annotations produced in
        Stage 2 (data_pretagging.py). It applies configurable filtering rules
        such as removing:
            • Very small bounding boxes (area < min_area)
            • Low-confidence detections (score < min_score)
            • Invalid boxes (non-positive width/height)

        The result is a cleaner COCO annotation file that is easier and safer
        for annotators to work with.

    What This Script Does:
        ✓ Loads a COCO JSON file (default: pre_tags/pre_tags_raw.json)
        ✓ Filters annotations based on:
            – min_area (bounding box area threshold)
            – min_score (model confidence threshold, if present)
        ✓ Ensures all surviving boxes are valid (w > 0, h > 0)
        ✓ Optionally removes images that have zero remaining annotations from
            the image directory (cleanup mode)
        ✓ Writes a cleaned COCO JSON (default: pre_tags/pre_tags_cleaned.json)

    Inputs:
        • COCO JSON from Stage 2 (with "score" per annotation)
        • Optional image directory for file-level cleanup (default: ./frames/)

    Outputs:
        • Cleaned COCO file: pre_tags/pre_tags_cleaned.json
        • Optionally fewer image files in ./frames/ if cleanup is enabled.

    Operational Modes (same idea as previous stages):
        fast       – more aggressive cleanup (higher thresholds)
        balanced   – reasonable default for annotation
        accurate   – more conservative (keep more boxes)

    ----------------------------------------------------------------------------
    HOW TO RUN (LOCALLY)
    ----------------------------------------------------------------------------

        $ python pretag_cleanup.py \
            --input_json pre_tags/pre_tags_raw.json \
            --output_json pre_tags/pre_tags_cleaned.json \
            --images_dir frames \
            --mode balanced \
            --cleanup off

    ----------------------------------------------------------------------------
    HOW TO RUN (IN GOOGLE COLAB)
    ----------------------------------------------------------------------------

        !python pretag_cleanup.py \
            --input_json pre_tags/pre_tags_raw.json \
            --output_json pre_tags/pre_tags_cleaned.json \
            --images_dir frames \
            --mode balanced \
            --cleanup empty_images

    Notes for Colab:
        • Unknown Jupyter/Colab runtime args (-f kernel.json) are ignored.
        • Image cleanup is optional. By default, all images are kept and only
            the COCO file is cleaned.

===============================================================================
"""

import argparse
import json
from pathlib import Path
from utils.config_loader import get_config, get_mode_config
from utils.pipeline_tracker import save_stage_config

# Optional GPU info (for consistency with other stages)
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
    """Return (min_area, min_score, min_area_person) defaults for a given mode."""
    m = mode.lower()
    min_area = get_mode_config("cleanup", m, "min_area", 1000.0)
    min_score = get_mode_config("cleanup", m, "min_score", 0.5)
    min_area_person = get_mode_config("cleanup", m, "min_area_person", 300.0)
    return min_area, min_score, min_area_person


# =============================================================================
# COCO LOADING & CLEANUP
# =============================================================================


def load_coco(path: str, verbose: bool = True):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input COCO JSON not found: {p}")
    if verbose:
        print(f"[STEP] Loading COCO annotations from: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_mask(mask_path: str, target_width: int, target_height: int):
    """Load mask for filtering boxes in masked regions."""
    if not mask_path or not Path(mask_path).exists():
        return None
    
    import cv2
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask.shape != (target_height, target_width):
        mask = cv2.resize(mask, (target_width, target_height))
    return mask


def is_box_in_masked_region(bbox, mask, threshold=0.3):
    """Check if box overlaps significantly with masked (black) region."""
    if mask is None:
        return False
    
    x, y, w, h = bbox
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    
    # Clip to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(mask.shape[1], x2), min(mask.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        return False
    
    # Check overlap with masked region (black pixels = 0)
    box_region = mask[y1:y2, x1:x2]
    masked_pixels = (box_region == 0).sum()
    total_pixels = box_region.size
    
    if total_pixels == 0:
        return False
    
    overlap_ratio = masked_pixels / total_pixels
    return overlap_ratio > threshold


def clean_annotations(
    coco: dict,
    min_area: float,
    min_score: float,
    min_area_person: float,
    person_class_id: int,
    mask_path: str = None,
    verbose: bool = True,
):
    """
    Filter annotations by min_area and min_score (if 'score' exists).
    Applies class-aware filtering: people get lower area threshold (safety-critical).
    Optionally filters boxes in masked regions (likely false positives).

    Returns:
        cleaned_coco, per_image_counts, filtering_stats
    """
    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    categories = coco.get("categories", [])
    
    # Load mask if provided
    mask = None
    if mask_path and images:
        target_width = images[0].get("width", 960)
        target_height = images[0].get("height", 544)
        mask = load_mask(mask_path, target_width, target_height)
        if verbose and mask is not None:
            print(f"[INFO] Loaded mask for filtering: {mask_path}")

    if verbose:
        print("\n[STEP] Cleaning annotations...")
        print(f"[INFO] Starting with {len(anns)} annotations.")

    cleaned_annotations = []
    per_image_count = {}

    removed_small = 0
    removed_score = 0
    removed_invalid = 0
    removed_masked = 0
    processed_count = 0

    for ann in anns:
        processed_count += 1
        image_id = ann["image_id"]
        bbox = ann.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4:
            removed_invalid += 1
            continue

        x, y, w, h = bbox
        # Compute area (ignore stored 'area' in case it's stale)
        area = float(w) * float(h)

        # Class-aware area filtering (people are safety-critical)
        category_id = ann.get("category_id")
        area_threshold = min_area_person if category_id == person_class_id else min_area

        if area < area_threshold:
            removed_small += 1
            continue

        # Filter by score if present
        score = ann.get("score", None)
        if score is not None and score < min_score:
            removed_score += 1
            continue

        # Validate bbox
        if w <= 0 or h <= 0:
            removed_invalid += 1
            continue
        
        # Filter boxes in masked regions (likely false positives)
        if is_box_in_masked_region(bbox, mask, threshold=0.3):
            removed_masked += 1
            continue

        # Keep annotation – preserve score as 'confidence' for sample generation
        cleaned_ann = dict(ann)
        cleaned_ann["area"] = area
        if "score" in cleaned_ann:
            cleaned_ann["confidence"] = cleaned_ann["score"]
            del cleaned_ann["score"]

        cleaned_annotations.append(cleaned_ann)
        per_image_count[image_id] = per_image_count.get(image_id, 0) + 1

        if verbose and processed_count % max(1, len(anns) // 10) == 0:
            print(f"[INFO] Processed {processed_count}/{len(anns)} annotations...")

    cleaned_coco = {
        "images": images,
        "annotations": cleaned_annotations,
        "categories": categories,
    }

    filtering_stats = {
        "original": len(anns),
        "removed_small": removed_small,
        "removed_score": removed_score,
        "removed_invalid": removed_invalid,
        "removed_masked": removed_masked,
        "kept": len(cleaned_annotations),
    }

    if verbose:
        print("\n[SUMMARY – Cleanup]")
        print(f"  Original annotations:      {len(anns)}")
        print(f"  Removed (small area):      {removed_small}")
        print(f"  Removed (low score):       {removed_score}")
        print(f"  Removed (invalid bbox):    {removed_invalid}")
        if removed_masked > 0:
            print(f"  Removed (in masked region): {removed_masked}")
        print(f"  Kept annotations:          {len(cleaned_annotations)}")
        print("------------------------------------------------------------")

    return cleaned_coco, per_image_count, filtering_stats


def save_coco(coco: dict, path: str, verbose: bool = True):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)
    if verbose:
        print(f"[WRITE] Cleaned COCO written to: {out_path.resolve()}")


# =============================================================================
# IMAGE CLEANUP (OPTIONAL)
# =============================================================================


def cleanup_images(
    images_dir: str, per_image_counts: dict, cleanup_mode: str, verbose: bool
):
    """
    cleanup_mode:
        off            – do nothing
        empty_images   – delete images that have 0 annotations after cleanup
    """
    if cleanup_mode == "off":
        return

    images_dir = Path(images_dir)
    if not images_dir.exists():
        if verbose:
            print(f"[WARN] Images directory not found: {images_dir} (cleanup skipped)")
        return

    removed = 0
    # ASSUMPTION: Image IDs in COCO match sorted file order (1..N)
    # This is consistent with Stage 2's sequential ID assignment
    # PRODUCTION IMPROVEMENT: Load image_id_map.json from Stage 2 instead
    paths = sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )

    if cleanup_mode == "empty_images":
        for idx, p in enumerate(paths, start=1):
            count = per_image_counts.get(idx, 0)
            if count == 0:
                p.unlink()
                removed += 1

    if verbose:
        print(f"[CLEANUP] Removed {removed} image files ({cleanup_mode}).")


# =============================================================================
# ARGUMENT PARSING
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Protex AI – Pre-tag Cleanup (COCO → cleaned COCO)",
        allow_abbrev=False,
    )

    parser.add_argument(
        "--input_json",
        type=str,
        default=get_config("cleanup.input_json", "pre_tags/pre_tags_raw.json"),
        help="Path to input COCO JSON",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=get_config("cleanup.output_json", "pre_tags/pre_tags_cleaned.json"),
        help="Path for cleaned COCO JSON",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=get_config("cleanup.images_dir", "frames"),
        help="Directory with images for optional cleanup",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=get_config("defaults.mode", "balanced"),
        choices=["fast", "balanced", "accurate"],
        help="Cleanup aggressiveness mode",
    )

    parser.add_argument(
        "--min_area",
        type=float,
        default=None,
        help="Override minimum bbox area (pixels^2). If not set, derived from --mode.",
    )
    parser.add_argument(
        "--min_score",
        type=float,
        default=None,
        help="Override minimum detection score. If not set, derived from --mode.",
    )

    parser.add_argument(
        "--mask_path",
        type=str,
        default=get_config("preprocessing.mask_path", None),
        help="Path to mask for filtering boxes in masked regions",
    )

    parser.add_argument(
        "--cleanup",
        type=str,
        default=get_config("defaults.cleanup_mode", "off"),
        choices=["off", "empty_images"],
        help="Optional image cleanup mode",
    )

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[INFO] Ignoring unknown args from Jupyter/Colab: {unknown}")
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

    mode = args.mode.lower()
    mode_min_area, mode_min_score, mode_min_area_person = get_mode_defaults(mode)

    min_area = args.min_area if args.min_area is not None else mode_min_area
    min_score = args.min_score if args.min_score is not None else mode_min_score
    min_area_person = mode_min_area_person
    person_class_id = get_config("cleanup.person_class_id", 1)

    if verbose:
        print("============================================================")
        print(" Protex AI – Stage 3: Pre-tag Cleanup")
        print("============================================================")
        print(f"[ENV] GPU available: {GPU_AVAILABLE} | Device: {GPU_DEVICE}")
        print("------------------------------------------------------------")
        print("[CONFIGURATION]")
        print(f"  Input JSON:          {args.input_json}")
        print(f"  Output JSON:         {args.output_json}")
        print(f"  Images dir:          {args.images_dir}")
        print(f"  Mode:                {mode}")
        print(f"  Min area (px^2):     {min_area}")
        print(f"  Min score:           {min_score}")
        print(f"  Cleanup mode:        {args.cleanup}")
        print("============================================================\n")

    coco = load_coco(args.input_json, verbose=verbose)

    cleaned_coco, per_image_counts, filtering_stats = clean_annotations(
        coco,
        min_area=min_area,
        min_score=min_score,
        min_area_person=min_area_person,
        person_class_id=person_class_id,
        mask_path=args.mask_path,
        verbose=verbose,
    )

    save_coco(cleaned_coco, args.output_json, verbose=verbose)

    # Save pipeline configuration with filtering stats
    save_stage_config(
        "cleanup",
        {
            "mode": mode,
            "min_area": min_area,
            "min_score": min_score,
            "min_area_person": min_area_person,
            "person_class_id": person_class_id,
            "mask_path": args.mask_path if args.mask_path else "none",
            "filtering_stats": filtering_stats,
        },
    )

    if verbose:
        print("[STEP] Optional image cleanup...")
    cleanup_images(args.images_dir, per_image_counts, args.cleanup, verbose)

    if verbose:
        print("\n✅ Pre-tag cleanup completed.")
        print("============================================================\n")


if __name__ == "__main__":
    main()
