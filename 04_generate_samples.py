#!/usr/bin/env python
"""
===============================================================================
    Protex AI – Computer Vision Ops Pipeline
    -----------------------------------------------------------
    Module: generate_samples.py
    Stage: 4 / 4 (Sample Generation – COCO → annotated images)

    Purpose:
        This module generates human-friendly visual samples from the cleaned
        pre-tagged dataset. It overlays bounding boxes and labels on a subset
        of images and exports them to a dedicated samples directory. These
        samples can be used in:
            • The presentation demo
            • Quick qualitative QA of pre-tags
            • Sharing with stakeholders / annotators

    What This Script Does:
        ✓ Loads cleaned COCO annotations (Stage 3 output)
        ✓ Selects up to N random images (default: 20) that have annotations
        ✓ Draws bounding boxes and category labels onto the images
        ✓ Saves the annotated images into a samples directory

        Optional:
        ✓ If cleanup is enabled, clears the samples directory before writing.

    Inputs:
        • Cleaned COCO JSON (default: pre_tags/pre_tags_cleaned.json)
        • Original image directory (default: ./frames/)

    Outputs:
        • Annotated image files in: ./samples/
                (e.g., sample_00001.jpg, sample_00002.jpg, ...)

    ----------------------------------------------------------------------------
    HOW TO RUN (LOCALLY)
    ----------------------------------------------------------------------------

        $ python generate_samples.py \
            --input_json pre_tags/pre_tags_cleaned.json \
            --images_dir frames \
            --samples_dir traceables/samples \
            --num_samples 20 \
            --cleanup clear_dir

    ----------------------------------------------------------------------------
    HOW TO RUN (IN GOOGLE COLAB)
    ----------------------------------------------------------------------------

        !python generate_samples.py \
            --input_json pre_tags/pre_tags_cleaned.json \
            --images_dir frames \
            --samples_dir traceables/samples \
            --num_samples 20 \
            --cleanup clear_dir

    Notes:
        • Unknown Jupyter/Colab runtime args are ignored.
        • You can adjust --num_samples and --seed to control sample diversity.
        • This stage does not require the GPU, but logs its availability for
            consistency with earlier stages.

===============================================================================
"""

import argparse
import json
import random
from pathlib import Path

import cv2
from utils.config_loader import get_config

# Optional GPU info (for consistent logging)
try:
    import torch

    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_DEVICE = "cuda" if GPU_AVAILABLE else "cpu"
except ImportError:
    GPU_AVAILABLE = False
    GPU_DEVICE = "cpu"


# =============================================================================
# COCO HELPERS
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


def build_image_index(coco: dict):
    """Return (image_by_id, annotations_by_image_id, categories_by_id)."""
    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])

    image_by_id = {img["id"]: img for img in images}

    anns_by_img = {}
    for ann in anns:
        img_id = ann["image_id"]
        anns_by_img.setdefault(img_id, []).append(ann)

    cats_by_id = {c["id"]: c.get("name", f"id_{c['id']}") for c in cats}

    return image_by_id, anns_by_img, cats_by_id


# =============================================================================
# DRAWING
# =============================================================================


def draw_annotations_on_image(
    img_path: Path,
    anns: list,
    cats_by_id: dict,
    output_path: Path,
    verbose: bool = False,
):
    img = cv2.imread(str(img_path))
    if img is None:
        if verbose:
            print(f"[WARN] Failed to read image: {img_path}")
        return False, None

    # Color coding: person=red (safety-critical), equipment=blue, other=green
    person_classes = ["person"]
    equipment_classes = ["forklift", "truck", "car"]

    detection_count = len(anns)
    low_conf_count = 0
    class_counts = {}
    issues = []

    for ann in anns:
        bbox = ann.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4:
            continue
        x, y, w, h = bbox
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        cat_id = ann.get("category_id")
        label = cats_by_id.get(cat_id, str(cat_id))
        conf = ann.get("score", ann.get("confidence", 1.0))

        class_counts[label] = class_counts.get(label, 0) + 1

        # Color coding
        if label in person_classes:
            color = (0, 0, 255)  # Red for people (safety-critical)
        elif label in equipment_classes:
            color = (255, 0, 0)  # Blue for equipment
        else:
            color = (0, 255, 0)  # Green for other

        # Track low confidence
        if conf < 0.6:
            low_conf_count += 1
            color = (0, 165, 255)  # Orange for low confidence
            issues.append(f"Low confidence {label} ({conf:.2f})")

        # Draw rectangle
        thickness = 3 if label in person_classes else 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Draw label with confidence
        text = f"{label} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty1 = max(y1 - th - 4, 0)
        ty2 = ty1 + th + 4
        tx1 = x1
        tx2 = x1 + tw + 4

        cv2.rectangle(img, (tx1, ty1), (tx2, ty2), color, -1)
        cv2.putText(
            img,
            text,
            (tx1 + 2, ty2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)

    metadata = {
        "detection_count": detection_count,
        "low_confidence_count": low_conf_count,
        "class_counts": class_counts,
        "issues": issues,
    }

    return True, metadata


# =============================================================================
# CLEANUP (SAMPLES DIRECTORY)
# =============================================================================


def cleanup_samples_dir(samples_dir: str, cleanup_mode: str, verbose: bool):
    """
    cleanup_mode:
        off        – do nothing
        clear_dir  – delete all existing files in samples_dir before writing
    """
    if cleanup_mode == "off":
        return

    path = Path(samples_dir)
    if not path.exists():
        return

    removed = 0
    if cleanup_mode == "clear_dir":
        for p in path.iterdir():
            if p.is_file():
                p.unlink()
                removed += 1

    if verbose:
        print(f"[CLEANUP] Removed {removed} existing sample files ({cleanup_mode}).")


# =============================================================================
# ARGUMENT PARSING
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Protex AI – Sample Generation (annotated images)",
        allow_abbrev=False,
    )

    parser.add_argument(
        "--input_json",
        type=str,
        default=get_config("samples.input_json", "traceables/pre_tags/pre_tags_cleaned.json"),
        help="Cleaned COCO JSON",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=get_config("samples.images_dir", "traceables/frames"),
        help="Directory with source images",
    )
    parser.add_argument(
        "--samples_dir",
        type=str,
        default=get_config("samples.samples_dir", "traceables/samples"),
        help="Directory to store annotated samples",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=get_config("defaults.mode", "balanced"),
        choices=["fast", "balanced", "accurate"],
        help="Mode (kept for consistency)",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=get_config("samples.num_samples", 20),
        help="Number of sample images to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=get_config("samples.seed", 42),
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--cleanup",
        type=str,
        default=get_config("defaults.cleanup_mode", "off"),
        choices=["off", "clear_dir"],
        help="Cleanup behavior for samples_dir",
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

    if verbose:
        print("============================================================")
        print(" Protex AI – Stage 4: Sample Generation")
        print("============================================================")
        print(f"[ENV] GPU available: {GPU_AVAILABLE} | Device: {GPU_DEVICE}")
        print("------------------------------------------------------------")
        print("[CONFIGURATION]")
        print(f"  Input JSON:          {args.input_json}")
        print(f"  Images dir:          {args.images_dir}")
        print(f"  Samples dir:         {args.samples_dir}")
        print(f"  Mode:                {args.mode}")
        print(f"  Num samples:         {args.num_samples}")
        print(f"  Seed:                {args.seed}")
        print(f"  Cleanup mode:        {args.cleanup}")
        print(
            f"  Color coding:        Always enabled (Red=people, Blue=equipment, Orange=low-conf)"
        )
        print("============================================================\n")

    # Load COCO and build indices
    coco = load_coco(args.input_json, verbose=verbose)
    image_by_id, anns_by_img, cats_by_id = build_image_index(coco)

    # Select candidate images (those with at least one annotation)
    candidate_ids = [img_id for img_id, a in anns_by_img.items() if len(a) > 0]
    if not candidate_ids:
        raise RuntimeError("No images with annotations found in COCO file.")

    if verbose:
        print(f"[INFO] Found {len(candidate_ids)} images with annotations.")

    # Stratified sampling for diverse coverage
    # 1. Density stratification (empty/low/medium/high)
    # 2. Low-confidence bias (annotator-centric)
    # 3. Time-of-day coverage (early/mid/late)
    
    rng = random.Random(args.seed)
    
    # Bucket by detection density
    low_density = [img_id for img_id in candidate_ids if len(anns_by_img[img_id]) <= 2]
    med_density = [img_id for img_id in candidate_ids if 3 <= len(anns_by_img[img_id]) <= 5]
    high_density = [img_id for img_id in candidate_ids if len(anns_by_img[img_id]) > 5]
    
    # Find images with low-confidence detections (annotator focus)
    low_conf_imgs = set()
    for img_id, anns in anns_by_img.items():
        if any(ann.get('score', ann.get('confidence', 1.0)) < 0.6 for ann in anns):
            low_conf_imgs.add(img_id)
    
    # Time-of-day stratification (early/mid/late thirds)
    sorted_ids = sorted(candidate_ids)
    third = len(sorted_ids) // 3
    early = set(sorted_ids[:third])
    mid = set(sorted_ids[third:2*third])
    late = set(sorted_ids[2*third:])
    
    # Smart sampling: ensure diversity
    selected_ids = []
    target = args.num_samples
    
    # 40% from low-confidence (annotator-centric)
    low_conf_target = min(int(target * 0.4), len(low_conf_imgs))
    selected_ids.extend(rng.sample(sorted(low_conf_imgs), low_conf_target))
    
    # Remaining: stratify by density and time
    remaining = target - len(selected_ids)
    buckets = [low_density, med_density, high_density]
    per_bucket = remaining // 3
    
    for bucket in buckets:
        available = [x for x in bucket if x not in selected_ids]
        if available:
            # Within bucket, ensure time diversity
            early_b = [x for x in available if x in early]
            mid_b = [x for x in available if x in mid]
            late_b = [x for x in available if x in late]
            
            per_time = per_bucket // 3
            for time_bucket in [early_b, mid_b, late_b]:
                if time_bucket:
                    n = min(per_time, len(time_bucket))
                    selected_ids.extend(rng.sample(time_bucket, n))
    
    # Fill remaining with random
    if len(selected_ids) < target:
        remaining_ids = [x for x in candidate_ids if x not in selected_ids]
        if remaining_ids:
            n = min(target - len(selected_ids), len(remaining_ids))
            selected_ids.extend(rng.sample(remaining_ids, n))
    
    selected_ids = selected_ids[:target]

    if verbose:
        print(f"[STEP] Selected {len(selected_ids)} images for sample generation.")
        print(f"  - Low-confidence images: {len([x for x in selected_ids if x in low_conf_imgs])}")
        print(f"  - High-density scenes: {len([x for x in selected_ids if x in high_density])}")
        print(f"  - Time coverage: early={len([x for x in selected_ids if x in early])}, "
              f"mid={len([x for x in selected_ids if x in mid])}, late={len([x for x in selected_ids if x in late])}\n")

    # Cleanup samples directory if requested
    cleanup_samples_dir(args.samples_dir, args.cleanup, verbose)

    images_dir_path = Path(args.images_dir)
    samples_dir_path = Path(args.samples_dir)
    samples_dir_path.mkdir(parents=True, exist_ok=True)

    generated = 0
    for idx, img_id in enumerate(selected_ids, start=1):
        img_info = image_by_id.get(img_id)
        if img_info is None:
            if verbose:
                print(f"[WARN] No image info for image_id={img_id}, skipping.")
            continue

        file_name = img_info["file_name"]
        src_path = images_dir_path / file_name
        if not src_path.exists():
            if verbose:
                print(f"[WARN] Image file not found: {src_path}, skipping.")
            continue

        anns = anns_by_img.get(img_id, [])
        if not anns:
            if verbose:
                print(f"[WARN] No annotations found for image_id={img_id}, skipping.")
            continue

        out_name = f"sample_{idx:05d}.jpg"
        out_path = samples_dir_path / out_name

        ok, metadata = draw_annotations_on_image(
            src_path,
            anns,
            cats_by_id,
            out_path,
            verbose=verbose,
        )
        if ok:
            generated += 1
            if verbose:
                det_count = metadata["detection_count"]
                low_conf = metadata["low_confidence_count"]
                class_counts = metadata["class_counts"]
                issues = metadata["issues"]
                class_str = ", ".join(
                    [f"{k}:{v}" for k, v in sorted(class_counts.items())]
                )
                status = "⚠️ Low conf" if low_conf > 0 else "✓ Good"
                print(
                    f"[WRITE] Sample {generated}/{len(selected_ids)}: {out_name} - {det_count} detections ({class_str}) {status}"
                )
                if issues:
                    for issue in issues:
                        print(f"        ⚠️ {issue}")

    # Generate sample catalog
    catalog_path = samples_dir_path / "SAMPLE_CATALOG.md"
    with open(catalog_path, "w") as f:
        f.write("# Sample Image Catalog\n\n")
        f.write(
            "**Purpose**: Representative samples covering diverse scenarios for QA review.\n\n"
        )
        f.write("**Color Coding**:\n")
        f.write("- 🔴 Red boxes: People (safety-critical)\n")
        f.write("- 🔵 Blue boxes: Equipment (forklift, truck)\n")
        f.write("- 🟢 Green boxes: Other objects\n")
        f.write("- 🟠 Orange boxes: Low confidence (<0.6) - requires review\n\n")
        f.write("---\n\n")
        f.write("## Sample Overview\n\n")
        f.write(f"Total samples: {generated}\n\n")
        f.write("### Annotation Priority Guide\n\n")
        f.write("1. **Review orange boxes first** - Low confidence detections\n")
        f.write("2. **Verify red boxes** - People are safety-critical\n")
        f.write(
            "3. **Check for missing objects** - Especially PPE and safety equipment\n"
        )
        f.write(
            "4. **Validate crowded scenes** - Look for merged or missed detections\n\n"
        )
        f.write("---\n\n")
        f.write("## Interesting Failures & Edge Cases\n\n")
        f.write("This dataset includes samples specifically chosen to highlight:\n\n")
        f.write(f"- **Low-confidence detections** ({len([x for x in selected_ids if x in low_conf_imgs])} samples): "
                "Require manual review and correction\n")
        f.write("- **Occlusion cases**: Workers behind equipment, partial visibility\n")
        f.write("- **Lighting challenges**: Dawn/dusk transitions, shadows, glare\n")
        f.write("- **Crowded scenes**: Multiple overlapping objects, merged boxes\n")
        f.write("- **Small/distant objects**: May be missed or misclassified\n")
        f.write("- **Edge of frame**: Partial objects at image boundaries\n\n")
        f.write("**Why these matter**: These edge cases help calibrate annotation standards "
                "and identify systematic model weaknesses before full dataset processing.\n\n")
        f.write("---\n\n")
        f.write("## Common Issues to Watch For\n\n")
        f.write("- **Occlusion**: Workers behind equipment (low confidence)\n")
        f.write("- **Lighting**: False positives at dawn/dusk, shadows\n")
        f.write("- **Crowding**: Merged boxes in busy scenes\n")
        f.write("- **Small objects**: Missed fire extinguishers, tools\n")
        f.write("- **PPE**: Misclassified hard hats, missed safety vests\n\n")
        f.write("---\n\n")
        f.write(
            "**Note**: These samples represent the diversity of the full dataset, "
            "including intentionally selected edge cases and failure modes. "
            "Use them to calibrate annotation standards before processing the complete set.\n"
        )

    if verbose:
        print("\n[SUMMARY – Samples]")
        print(f"  Requested samples:   {args.num_samples}")
        print(f"  Generated samples:   {generated}")
        print(f"  Samples directory:   {samples_dir_path.resolve()}")
        print(f"  Sample catalog:      {catalog_path.resolve()}")
        print("============================================================")
        print("✅ Sample generation completed.\n")


if __name__ == "__main__":
    main()
