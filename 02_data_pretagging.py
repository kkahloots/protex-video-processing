#!/usr/bin/env python
"""
===============================================================================
    Protex AI – Computer Vision Ops Pipeline
    -----------------------------------------------------------
    Module: data_pretagging.py
    Stage: 2 / 4 (Data Pre-tagging – Frames → COCO detections)

    Purpose:
        This module performs the second stage in the Protex AI CV Ops pipeline.
        After video frames have been extracted (stage 1), this script loads each
        image, traceables an object detector (PyTorch / Faster R-CNN), and generates
        COCO-format annotations.

        These “pre-tags” are used by annotators to speed up labeling and reduce
        manual effort — a critical step when scaling to thousands of frames
        across multiple industrial client sites.

    What This Script Does:
        ✓ Loads frames from a directory (e.g., /frames/)
        ✓ Runs a pretrained object detector (default: Faster R-CNN ResNet50 FPN)
        ✓ Produces COCO-format annotations containing:
            • Bounding boxes
            • Class IDs
            • Confidence scores (kept for cleanup stage)
        ✓ Saves results into: pre_tags/pre_tags_raw.json
        ✓ Optional cleanup:
            --cleanup empty_images  → deletes frames with zero detections

    Why This Matters for Protex AI:
        Pre-tagging accelerates the annotation workflow by surfacing candidate
        detections upfront. Annotators only adjust or correct bounding boxes,
        drastically reducing labeling latency across multiple time zones and
        deployment regions (20+ countries).

        High-quality pre-tags reduce:
            → Manual labeling cost
            → Turnaround time for model fine-tuning
            → Human error in industrial safety datasets

    Inputs:
        A directory of preprocessed frames (default: ./frames/)

    Outputs:
        pre_tags/pre_tags_raw.json (COCO-format detection file)
        Optionally cleaned frame directory (if cleanup enabled)

    Operational Modes (matching Stage 1 style):
        fast       – Higher confidence threshold, fewer detections
        balanced   – Good default for annotation pipelines
        accurate   – Lower threshold, more sensitive detection

    ----------------------------------------------------------------------------
    HOW TO RUN (LOCALLY)
    ----------------------------------------------------------------------------

        $ python data_pretagging.py \
            --images_dir frames \
            --output_dir pre_tags \
            --mode balanced \
            --batch_size 64 \
            --cleanup off

    ----------------------------------------------------------------------------
    HOW TO RUN (IN GOOGLE COLAB)
    ----------------------------------------------------------------------------

        !python data_pretagging.py \
            --images_dir frames \
            --output_dir pre_tags \
            --mode balanced \
            --batch_size 64 \
            --cleanup empty_images

    Notes for Colab:
        • The script automatically detects and uses GPU if available.
        • Unknown Jupyter/Colab runtime args are ignored safely.
        • Ensure Stage 1 (data_preprocessing.py) has been run so that /frames/
            contains resized and filtered images.

===============================================================================
"""

import argparse
import json
from pathlib import Path

import cv2
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from utils.config_loader import get_config, get_mode_config
from utils.pipeline_tracker import save_stage_config

# =============================================================================
# GPU / DEVICE
# =============================================================================

GPU_AVAILABLE = torch.cuda.is_available()
GPU_DEVICE = torch.device("cuda" if GPU_AVAILABLE else "cpu")


# =============================================================================
# GLOBAL DEFAULTS
# =============================================================================

DEFAULT_CLEANUP_MODE = "off"

# COCO 91-class labels (indices 1..90)
COCO_CLASSES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# Safety-relevant class whitelist for street monitoring
# Reduces annotation noise by filtering out irrelevant objects (bananas, wine glasses, etc.)
SAFETY_RELEVANT_CLASSES = {
    1: "person",  # Pedestrians (safety-critical)
    2: "bicycle",  # Street vehicles
    3: "car",  # Vehicles
    4: "motorcycle",  # Vehicles
    6: "bus",  # Large vehicles
    8: "truck",  # Heavy vehicles
    10: "traffic light",  # Street infrastructure
    11: "fire hydrant",  # Street equipment
    13: "stop sign",  # Traffic signage
}

# Street-common objects get lower confidence threshold (0.5)
# Other objects get higher threshold (0.7) to reduce false positives
STREET_COMMON_CLASSES = {
    1,  # person
    2,  # bicycle
    3,  # car
    4,  # motorcycle
    6,  # bus
    8,  # truck
    10, # traffic light
    13, # stop sign}  
}
STREET_RARE_CLASSES = {11}  # fire hydrant


# =============================================================================
# MODE-DEPENDENT DEFAULTS
# =============================================================================


def get_mode_defaults(mode: str):
    """Return min_confidence for a given mode."""
    return get_mode_config("pretagging", mode.lower(), "min_confidence", 0.5)


# =============================================================================
# MODEL LOADING
# =============================================================================


def load_detector(model_name: str, nms_iou_threshold: float = 0.5, verbose: bool = True):
    if verbose:
        print("\n[STEP] Loading detection model:", model_name)
    if model_name != "fasterrcnn_resnet50_fpn":
        raise ValueError(f"Unsupported model_name: {model_name}")

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Configure NMS IoU threshold
    # Lower IoU = more aggressive NMS (fewer overlapping boxes)
    # Higher IoU = keep more overlapping boxes
    model.roi_heads.nms_thresh = nms_iou_threshold
    
    model.to(GPU_DEVICE)
    model.eval()

    if verbose:
        print(f"[INFO] Model loaded on device: {GPU_DEVICE}")
        print(f"[INFO] NMS IoU threshold: {nms_iou_threshold}")
    return model


# =============================================================================
# IMAGE LOADING & BATCHING
# =============================================================================


def list_images(images_dir: str, max_images: int = 0):
    img_dir = Path(images_dir)
    paths = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if max_images > 0:
        paths = paths[:max_images]
    return paths


def prepare_image(path: Path):
    """Load a single image and convert to tensor for detection."""
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = transforms.ToTensor()(img_rgb)  # [0,1], CxHxW
    return img_tensor, img_rgb.shape[1], img_rgb.shape[0]  # width, height


# =============================================================================
# DETECTION + COCO GENERATION
# =============================================================================


def run_detection(
    model,
    image_paths: list,
    output_path: str,
    min_confidence: float,
    batch_size: int,
    verbose: bool = True,
):
    if verbose:
        print(f"[INFO] Processing {len(image_paths)} images.")

    if not image_paths:
        raise RuntimeError("No images found for pre-tagging.")

    # Build categories - only safety-relevant classes
    categories = [
        {"id": cid, "name": name, "supercategory": "safety"}
        for cid, name in SAFETY_RELEVANT_CLASSES.items()
    ]

    coco = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    ann_id = 1
    image_id_map = {}  # path -> id
    per_image_det_count = {}

    # batching over images
    total_detections = 0
    total_filtered = 0
    processed_count = 0

    if verbose:
        print("\n[STEP] Running detection over batches...")

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch_tensors = []
        batch_sizes = []

        if verbose:
            print(
                f"[BATCH] Processing images {start}..{start + len(batch_paths) - 1} / {len(image_paths)}"
            )

        # prepare batch
        for p in batch_paths:
            img_tensor, w, h = prepare_image(p)
            batch_tensors.append(img_tensor.to(GPU_DEVICE))
            batch_sizes.append((w, h))

        with torch.no_grad():
            outputs = model(batch_tensors)

        # per-image post-processing
        for p, (w, h), out in zip(batch_paths, batch_sizes, outputs):
            processed_count += 1
            fname = p.name
            if p not in image_id_map:
                img_id = len(image_id_map) + 1
                image_id_map[p] = img_id
                coco["images"].append(
                    {
                        "id": img_id,
                        "file_name": fname,
                        "width": w,
                        "height": h,
                    }
                )
            else:
                img_id = image_id_map[p]

            boxes = out["boxes"].cpu().numpy()
            labels = out["labels"].cpu().numpy().astype(int)
            scores = out["scores"].cpu().numpy()

            per_image_det = 0

            for box, label, score in zip(boxes, labels, scores):
                total_detections += 1

                # Filter: safety-relevant classes only (ignore bananas, wine glasses, etc.)
                if int(label) not in SAFETY_RELEVANT_CLASSES:
                    total_filtered += 1
                    continue

                # Class-specific confidence thresholds
                # Street-common objects: lower threshold (0.5) for better recall
                # Street-rare objects: higher threshold (0.7) to reduce false positives
                class_threshold = (
                    min_confidence
                    if int(label) in STREET_COMMON_CLASSES
                    else min_confidence * 1.4
                )

                if score < class_threshold:
                    total_filtered += 1
                    continue

                x1, y1, x2, y2 = box.astype(float)
                bw = x2 - x1
                bh = y2 - y1

                if bw <= 0 or bh <= 0:
                    continue

                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(label),
                        "bbox": [float(x1), float(y1), float(bw), float(bh)],
                        "area": float(bw * bh),
                        "iscrowd": 0,
                        "segmentation": [[]],
                        "score": float(score),
                    }
                )
                ann_id += 1
                per_image_det += 1

            per_image_det_count[img_id] = per_image_det

            if verbose and per_image_det == 0:
                print(
                    f"[INFO] {fname} ({processed_count}/{len(image_paths)}): No detections (>= {min_confidence})"
                )
            elif verbose and per_image_det > 0:
                print(
                    f"[INFO] {fname} ({processed_count}/{len(image_paths)}): {per_image_det} detections kept."
                )

    # write COCO file
    out_path = Path(output_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    if verbose:
        print("\n[SUMMARY]")
        print(f"  Total images processed:     {len(image_paths)}")
        print(f"  Raw detections:            {total_detections}")
        print(f"  Filtered (< conf):         {total_filtered}")
        print(f"  Kept detections:           {total_detections - total_filtered}")
        print(f"  COCO JSON written to:      {out_path.resolve()}")
        print("------------------------------------------------------------")

    return coco, per_image_det_count, total_detections, total_filtered


# =============================================================================
# CLEANUP (OPTIONAL)
# =============================================================================


def cleanup_images(
    images_dir: str, per_image_det_count: dict, cleanup_mode: str, verbose: bool
):
    """
    cleanup_mode:
        off           -> keep everything
        empty_images  -> delete frames with 0 detections

    ASSUMPTION: Image IDs were assigned sequentially (1..N) to sorted image paths.
    This matches the assignment logic in run_detection where:
        image_paths = sorted(list_images(images_dir))
        image_id = len(image_id_map) + 1  # sequential assignment

    PRODUCTION IMPROVEMENT: Persist image_id_map.json from Stage 2:
        {"frame_00001.jpg": 1, "frame_00002.jpg": 2, ...}
    Then Stage 3 loads it instead of assuming sorted order.
    """
    if cleanup_mode == "off":
        return

    images_dir = Path(images_dir)
    removed = 0

    if cleanup_mode == "empty_images":
        # Reconstruct image_id mapping using same sorted order as Stage 2
        paths = list_images(str(images_dir))
        # Images were assigned ids 1..N in sorted order
        for idx, p in enumerate(paths, start=1):
            if per_image_det_count.get(idx, 0) == 0:
                p.unlink()
                removed += 1

    if verbose:
        print(f"[CLEANUP] Removed {removed} images ({cleanup_mode}).")


# =============================================================================
# ARGUMENT PARSING
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Protex AI – Data Pre-tagging (object detection)",
        allow_abbrev=False,
    )

    parser.add_argument(
        "--images_dir",
        type=str,
        default=get_config("pretagging.images_dir", "frames"),
        help="Directory with input frames",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=get_config("pretagging.output_dir", "pre_tags"),
        help="Directory to store outputs",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=get_config("defaults.mode", "balanced"),
        choices=["fast", "balanced", "accurate"],
        help="Speed/accuracy mode",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=get_config("pretagging.model_name", "fasterrcnn_resnet50_fpn"),
        help="Detector model to use",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=get_config("pretagging.batch_size", 64),
        help="Batch size for inference",
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=get_config("pretagging.max_images", 0),
        help="Limit number of images (0 = no limit).",
    )

    parser.add_argument(
        "--min_confidence",
        type=float,
        default=None,
        help="Override per-detection confidence threshold.",
    )
    
    parser.add_argument(
        "--nms_iou_threshold",
        type=float,
        default=None,
        help="NMS IoU threshold (0-1). Lower = fewer overlapping boxes.",
    )

    parser.add_argument(
        "--cleanup",
        type=str,
        default=get_config("defaults.cleanup_mode", "off"),
        choices=["off", "empty_images"],
        help="Optional image cleanup after detection.",
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
    mode_min_conf = get_mode_defaults(mode)
    min_conf = args.min_confidence if args.min_confidence is not None else mode_min_conf
    
    # Mode-specific NMS IoU threshold from config
    mode_nms_iou = get_mode_config("pretagging", mode, "nms_iou_threshold", 0.5)
    nms_iou = args.nms_iou_threshold if args.nms_iou_threshold is not None else mode_nms_iou

    images_dir = args.images_dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coco_out_path = output_dir / "pre_tags_raw.json"

    if verbose:
        print("============================================================")
        print(" Protex AI – Stage 2: Data Pre-tagging")
        print("============================================================")
        print(f"[ENV] GPU available: {GPU_AVAILABLE} | Device: {GPU_DEVICE}")
        print("------------------------------------------------------------")
        print("[CONFIGURATION]")
        print(f"  Images dir:          {images_dir}")
        print(f"  Output dir:          {output_dir}")
        print(f"  Mode:                {mode}")
        print(f"  Model:               {args.model_name}")
        print(f"  Batch size:          {args.batch_size}")
        print(f"  Min confidence:      {min_conf}")
        print(f"  NMS IoU threshold:   {nms_iou}")
        print(f"  Max images:          {args.max_images}")
        print(f"  Cleanup mode:        {args.cleanup}")
        print("============================================================\n")

    model = load_detector(args.model_name, nms_iou_threshold=nms_iou, verbose=verbose)

    # List and limit images (max_images for quick debugging/dry traceables)
    image_paths = list_images(images_dir, max_images=args.max_images)

    if verbose and args.max_images > 0:
        print(
            f"[INFO] Limited to {len(image_paths)} images (max_images={args.max_images})"
        )

    # run detection
    coco, per_image_det_count, total_detections, total_filtered = run_detection(
        model=model,
        image_paths=image_paths,
        output_path=str(coco_out_path),
        min_confidence=min_conf,
        batch_size=args.batch_size,
        verbose=verbose,
    )

    # Save pipeline configuration
    save_stage_config(
        "pretagging",
        {
            "mode": mode,
            "model_name": args.model_name,
            "min_confidence": min_conf,
            "nms_iou_threshold": nms_iou,
            "batch_size": args.batch_size,
            "max_images": args.max_images if args.max_images > 0 else "all",
            "raw_detections": total_detections,
            "filtered_low_conf": total_filtered,
        },
    )

    # cleanup (optional)
    if verbose:
        print("[STEP] Optional image cleanup...")
    cleanup_images(images_dir, per_image_det_count, args.cleanup, verbose)

    if verbose:
        print("\n✅ Pre-tagging completed.")
        print("============================================================\n")


if __name__ == "__main__":
    main()
