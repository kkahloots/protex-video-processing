#!/usr/bin/env python
"""
Generate a polished MP4 presentation video for Protex AI pipeline.
Requires: pip install moviepy pillow pyttsx3

Presentation Guide:
- Duration: ~6.5 minutes (35 slides)
- Key Metrics: 64% time reduction, $58M/year savings at scale
- Focus: EM thinking, production readiness, business value
- Sources: docs/FEATURES.md, traceables/report.md, ANNOTATOR_GUIDE.md
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

try:
    from moviepy.editor import ImageClip, concatenate_videoclips

    MOVIEPY_AVAILABLE = True
except ImportError as e:
    MOVIEPY_AVAILABLE = False
    print(f"[WARN] moviepy not available: {e}")

try:
    import pyttsx3

    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("[WARN] pyttsx3 not available. Install with: pip install pyttsx3")


def create_slide(
    text, title, slide_num, width=1920, height=1080, bg_color=(20, 20, 40)
):
    """Create a slide image with title and text."""
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 70
        )
        text_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40
        )
    except:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()

    # Title
    draw.text((100, 100), title, fill=(0, 200, 255), font=title_font)

    # Content
    y_offset = 300
    for line in text.split("\n"):
        draw.text((100, y_offset), line, fill=(255, 255, 255), font=text_font)
        y_offset += 80

    # Slide number
    draw.text(
        (width - 200, height - 100),
        f"Slide {slide_num}",
        fill=(100, 100, 100),
        font=text_font,
    )

    return img


def generate_narration(text, output_path):
    """Generate narration audio using TTS."""
    if not TTS_AVAILABLE:
        print(f"[SKIP] TTS not available, skipping narration for: {output_path}")
        return False

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        return True
    except Exception as e:
        print(f"[ERROR] TTS failed: {e}")
        return False


def create_image_slide(
    image_path, title, slide_num, comment="", width=1920, height=1080
):
    """Create a slide with an image."""
    img = Image.open(image_path)
    img = img.resize((width, height - 200), Image.LANCZOS)

    slide = Image.new("RGB", (width, height), (20, 20, 40))
    slide.paste(img, (0, 100))

    draw = ImageDraw.Draw(slide)
    try:
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60
        )
        comment_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 35
        )
    except:
        title_font = ImageFont.load_default()
        comment_font = ImageFont.load_default()

    draw.text((50, 20), title, fill=(0, 200, 255), font=title_font)
    if comment:
        draw.text((50, 900), comment, fill=(200, 200, 200), font=comment_font)
    draw.text(
        (width - 200, height - 80),
        f"Slide {slide_num}",
        fill=(100, 100, 100),
        font=title_font,
    )

    return slide


def find_matching_frame(sample_path, frames_dir):
    """Find the original frame file that corresponds to a sample."""
    import json

    # Try to load COCO to find the mapping
    coco_path = Path("traceables/pre_tags/pre_tags_cleaned.json")
    if coco_path.exists():
        try:
            with open(coco_path) as f:
                coco = json.load(f)

            # Get sample number from filename (e.g., sample_00001.jpg -> 1)
            sample_name = sample_path.name
            if sample_name.startswith("sample_") and sample_name.endswith(".jpg"):
                sample_num = int(sample_name[7:12])  # Extract 5-digit number

                # Get the image info for this sample (samples are 1-indexed)
                images = coco.get("images", [])
                if sample_num <= len(images):
                    # Find images with annotations
                    anns_by_img = {}
                    for ann in coco.get("annotations", []):
                        img_id = ann["image_id"]
                        anns_by_img.setdefault(img_id, []).append(ann)

                    # Get images with annotations (same logic as Stage 4)
                    candidate_ids = [
                        img["id"] for img in images if img["id"] in anns_by_img
                    ]

                    if sample_num <= len(candidate_ids):
                        # Get the image_id for this sample
                        img_id = candidate_ids[sample_num - 1]

                        # Find the image info
                        for img in images:
                            if img["id"] == img_id:
                                frame_file = frames_dir / img["file_name"]
                                if frame_file.exists():
                                    return frame_file
        except:
            pass

    # Fallback: return the sample itself (will show annotated version on both sides)
    return sample_path


def create_before_after_slide(
    frame_path, sample_path, title, comment, slide_num, width=1920, height=1080
):
    """Create side-by-side anno/not slide."""
    slide = Image.new("RGB", (width, height), (20, 20, 40))

    # Find the correct original frame for this sample
    frames_dir = Path("traceables/frames")
    actual_frame_path = find_matching_frame(Path(sample_path), frames_dir)

    # Load the same base frame for both sides
    # BEFORE: raw frame without annotations
    # AFTER: annotated sample (which should be the same frame with bounding boxes)
    frame = Image.open(actual_frame_path).resize(
        (width // 2 - 20, height - 300), Image.LANCZOS
    )
    sample = Image.open(sample_path).resize(
        (width // 2 - 20, height - 300), Image.LANCZOS
    )

    # Paste side by side (lowered 40px) - SWAPPED
    # LEFT: annotated sample
    slide.paste(sample, (10, 190))
    # RIGHT: raw frame
    slide.paste(frame, (width // 2 + 10, 190))

    draw = ImageDraw.Draw(slide)
    try:
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60
        )
        label_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40
        )
        comment_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 35
        )
    except:
        title_font = label_font = comment_font = ImageFont.load_default()

    # Title
    draw.text((50, 20), title, fill=(0, 200, 255), font=title_font)

    # Comment on top of images (lowered 30px)
    if comment:
        draw.text((50, 150), comment, fill=(200, 200, 200), font=comment_font)

    # Labels above images - SWAPPED
    draw.text(
        (width // 4 - 100, 100), "ANNOTATED", fill=(100, 255, 100), font=label_font
    )
    draw.text(
        (3 * width // 4 - 100, 100),
        "NOT ANNOTATED",
        fill=(255, 100, 100),
        font=label_font,
    )

    draw.text(
        (width - 200, height - 80),
        f"Slide {slide_num}",
        fill=(100, 100, 100),
        font=comment_font,
    )

    return slide


def load_report_stats():
    """Load stats from report.json and pipeline_summary.json."""
    import json

    stats = {
        "num_images": 0,
        "num_annotations": 0,
        "avg_per_image": 0.0,
        "class_dist": {},
        "total_classes": 0,
        "pipeline": {},
        "quality": {},
    }

    report_path = Path("traceables/report/stats/report.json")
    if report_path.exists():
        try:
            with open(report_path) as f:
                report = json.load(f)

            ds = report.get("dataset_summary", {})
            stats["num_images"] = ds.get("total_images", 0)
            stats["num_annotations"] = ds.get("total_annotations", 0)
            stats["avg_per_image"] = ds.get("annotations_per_image", 0.0)
            stats["images_with_anns"] = ds.get("images_with_annotations", 0)
            stats["high_activity"] = ds.get("images_high_activity", 0)

            # Load class distribution
            class_dist = report.get("class_distribution", [])
            stats["class_dist"] = {
                c["class"]: (c["count"], c["percentage"]) for c in class_dist
            }
            stats["total_classes"] = len(class_dist)

            # Quality metrics
            qm = report.get("quality_metrics", {})
            stats["quality"] = {
                "avg_bbox_area": qm.get("avg_bbox_area_px2", 0),
                "avg_confidence": qm.get("avg_confidence", 0),
                "low_conf_count": qm.get("low_confidence_count", 0),
            }

            # Pipeline summary
            ps = report.get("pipeline_summary", {}).get("steps", [])
            if len(ps) >= 3:
                stats["pipeline"] = {
                    "stage1_saved": ps[0].get("statistics", {}).get("saved_frames", 0),
                    "stage1_total": ps[0]
                    .get("statistics", {})
                    .get("total_video_frames", 0),
                    "stage1_reduction": ps[0]
                    .get("statistics", {})
                    .get("reduction_pct", 0),
                    "stage2_raw": ps[1].get("statistics", {}).get("raw_detections", 0),
                    "stage2_kept": ps[1]
                    .get("statistics", {})
                    .get("kept_detections", 0),
                    "stage3_kept": ps[2].get("statistics", {}).get("kept", 0),
                    "stage3_removed": ps[2]
                    .get("statistics", {})
                    .get("removed_small", 0),
                }
        except:
            pass

    return stats


def create_presentation_video(output_path="protex_presentation.mp4"):
    """Generate complete presentation video."""

    if not MOVIEPY_AVAILABLE:
        print("[ERROR] moviepy required. Install with: pip install moviepy")
        return False

    slides_dir = Path("traceables/presentation_slides")
    slides_dir.mkdir(parents=True, exist_ok=True)

    samples_dir = Path("traceables/samples")
    sample_images = (
        sorted(samples_dir.glob("sample_*.jpg")) if samples_dir.exists() else []
    )

    frames_dir = Path("traceables/frames")
    frame_images = sorted(frames_dir.glob("frame_*.jpg")) if frames_dir.exists() else []

    # Load dynamic stats
    stats = load_report_stats()

    # Define slides based on documentation
    slides = [
        {
            "title": "Protex AI: \nIndustrial Safety Computer Vision Pipeline",
            "text": "Kal Kahloot – Engineering Manager Assignment\n\n\nStory: From 24h of raw timelapse factory footage to a curated,\nannotator-ready dataset using a 4-stage, production-oriented\nCV pipeline for industrial safety monitoring.",
            "duration": 10,
        },
        {
            "title": "Business & Safety Context",
            "text": "Street monitoring systems capture traffic, pedestrians, and vehicles.\n\nCCTV cameras are already there:\n        • 24h feeds generate huge volumes of video\n        • Most frames are routine, a few contain risky behaviours\n        • We want traffic violations and unsafe patterns surfaced quickly",
            "duration": 9,
        },
        {
            "title": "The Scale Problem: 24h Video",
            "text": "A single 24-hour camera feed at 0.1 FPS yields thousands of frames.\n\nNaive manual workflow:\n        • Export all frames\n        • Manually draw boxes on every candidate object\n        • Repeat for every camera, every day\n\nResult: annotation spirals out of control as we add more cameras.",
            "duration": 10,
        },
        {
            "title": "What We Actually Want",
            "text": "Safety teams do not want raw frames – they want structured signals:\n        • Clips where people and vehicles interact in risky ways\n        • Evidence for near-misses and policy violations\n        • Clear, auditable datasets to fine-tune detection models\n\nOur job: turn always-on video into a targeted, annotator-friendly dataset.",
            "duration": 9,
        },
        {
            "title": "Pipeline Overview – 4 Stages",
            "text": "Stage 1 – Preprocessing: video → cleaned frames\nStage 2 – Pre-tagging: frames → COCO detections\nStage 3 – Cleanup: raw pre-tags → filtered COCO\nStage 4 – Sampling & Report: curated clips + MD report\n\nEach stage is config-driven, logged, and can be tuned per client site.",
            "duration": 9,
        },
        {
            "type": "image",
            "path": "docs/imgs/stage1.png",
            "title": "Stage 1: Preprocessing",
            "comment": "60-80% frame reduction via FPS sampling, brightness, blur, duplicates",
            "duration": 8,
        },
        {
            "type": "image",
            "path": "docs/imgs/stage2.png",
            "title": "Stage 2: Pre-tagging",
            "comment": "Faster R-CNN inference with batch processing and GPU acceleration",
            "duration": 8,
        },
        {
            "type": "image",
            "path": "docs/imgs/stage3.png",
            "title": "Stage 3: Cleanup",
            "comment": "Class-aware filtering: People 300px², Equipment 1000px²",
            "duration": 8,
        },
        {
            "title": "Stage 1 – Preprocessing (Video → Frames)",
            "text": "Ref: data_preprocessing.py\nResponsibilities:\n        • Sample frames at mode-dependent FPS (fast / balanced / accurate)\n        • Drop dark, blurry, noisey, and duplicated frames\n        • Apply static masks to ignore irrelevant regions\n        • Standardize resolution and generate frames_metadata.json",
            "duration": 9,
        },
        {
            "title": f"Stage 1 – Results: {stats['pipeline'].get('stage1_reduction', 0):.1f}% Reduction",
            "text": f"This dataset:\n        • Input: {stats['pipeline'].get('stage1_total', 0):,} video frames (32 FPS)\n        • Sampled: Every 40 frames (0.8 FPS)\n        • Output: {stats['pipeline'].get('stage1_saved', 0)} filtered frames\n        • Reduction: {stats['pipeline'].get('stage1_reduction', 0):.1f}%\n\nFilters removed:\n        • 71 duplicates, 4 noise frames\n\nResult: Only quality frames sent to GPU",
            "duration": 9,
        },
        {
            "title": "Stage 1 – Why It Matters",
            "text": "Street CCTV video is messy:\n        • Long stretches of no activity\n        • Camera glitches, night mode, lens flare\n        • Static backgrounds with tiny changes\n\nCleaning early saves:\n        • GPU time downstream\n        • Annotation cost (no one labels dark or duplicate frames)\n        • Storage and bandwidth",
            "duration": 9,
        },
        {
            "title": "Stage 2 – Pre-tagging \n(Frames → COCO Detections)",
            "text": "data_pretagging.py\n\nResponsibilities:\n        • Load sampled frames in batches\n        • Run Faster R-CNN ResNet-50 FPN on GPU if available\n        • Keep only safety-relevant COCO classes:\n  people, bikes, cars, motorbikes, buses, trucks, traffic lights, stop signs\n        • Write pre_tags_raw.json in COCO format",
            "duration": 9,
        },
        {
            "title": f"Stage 2 – Results: {stats['pipeline'].get('stage2_kept', 0)} Detections",
            "text": f"This dataset:\n        • Processed: {stats['num_images']} images (2 batches)\n        • Raw detections: {stats['pipeline'].get('stage2_raw', 0):,}\n        • Filtered (low confidence <0.6): {stats['pipeline'].get('stage2_raw', 0) - stats['pipeline'].get('stage2_kept', 0):,}\n        • Kept detections: {stats['pipeline'].get('stage2_kept', 0)}\n\nModel: Faster R-CNN ResNet50 FPN (CPU)\nBatch size: 16\n\nResult: High-quality pre-tags for cleanup",
            "duration": 9,
        },
        {
            "title": "Stage 2 – Class & Threshold Strategy",
            "text": "Key idea: filter at the right place.\n\n        • Whitelist only safety-relevant categories\n        • Lower thresholds for common street objects (person, car, truck)\n        • Slightly higher thresholds for rare ones\n\nOutcome: fewer irrelevant boxes, better starting point for annotators.",
            "duration": 9,
        },
        {
            "title": "Stage 3 – Pre-tag Cleanup \n(COCO → Cleaned COCO)",
            "text": "pretag_cleanup.py\n\nResponsibilities:\n        • Drop tiny boxes and low-confidence detections\n        • Apply class-aware area thresholds (people vs vehicles)\n        • Remove boxes that sit mostly in masked regions\n        • Optionally delete images with zero remaining annotations",
            "duration": 9,
        },
        {
            "title": f"Stage 3 – Results: {stats['pipeline'].get('stage3_kept', 0)} Final Annotations",
            "text": f"This dataset:\n        • Input: {stats['pipeline'].get('stage2_kept', 0)} detections\n        • Removed (small area <1000px²): {stats['pipeline'].get('stage3_removed', 0)}\n        • Removed (low score): 0\n        • Final annotations: {stats['pipeline'].get('stage3_kept', 0)}\n\nClass-aware filtering:\n        • Vehicles: 1000px² min\n        • People: 300px² min (safety-critical)\n\nResult: {stats.get('images_with_anns', 0)}/{stats['num_images']} images with annotations",
            "duration": 9,
        },
        {
            "title": "Stage 3 – Filtering Logic",
            "text": "Filtering rules:\n        • Compute area from bbox; ignore stale area fields\n        • Enforce min_area_person and min_area for other classes\n        • Respect min_score when present\n        • Require w > 0 and h > 0\n        • Drop boxes mostly overlapping masked regions\n\nResult: cleaner COCO file that is safe to hand to annotators.",
            "duration": 10,
        },
        {
            "type": "before_after",
            "frame_idx": 5,
            "title": "Example 1: Detection Quality",
            "comment": "Person detection in varying lighting conditions",
            "duration": 6,
        },
        {
            "type": "before_after",
            "frame_idx": 7,
            "title": "Example 2: Multiple Objects",
            "comment": "Multiple persons detected in complex scene",
            "duration": 6,
        },
        {
            "type": "before_after",
            "frame_idx": 10,
            "title": "Example 3: Occlusion Challenge",
            "comment": "Partial occlusion - some objects behind others not detected",
            "duration": 6,
        },
        {
            "type": "before_after",
            "frame_idx": 2,
            "title": "Example 4: Distance & Scale",
            "comment": "Car and bus at different distances from camera",
            "duration": 6,
        },
        {
            "title": "Design Rationale: Config-Driven (1/2)",
            "text": "Ref: docs/PARAMETERS.md\n\nWhy YAML configuration?\n    - fast/balanced/accurate modes\n    - Per-client tuning (no code changes)\n    - Environment-specific (dev/staging/prod)\n    - Compare parameter sets (e.g., FPS 0.5 vs 1.0)",
            "duration": 8,
        },
        {
            "title": "Design Rationale: Config-Driven (2/2)",
            "text": "Example scenarios:\n    - Pilot/Quick iteration → fast mode\n    - Production annotation → balanced mode\n    - Critical area/Incident review → accurate mode\n\nSee PARAMETERS.md for full trade-offs guide",
            "duration": 8,
        },
        {
            "title": "Design Rationale: Model Choice",
            "text": "Why Faster R-CNN ResNet50 FPN?\n      ✅ Battle-tested on COCO (80 classes)\n      ✅ Good accuracy/speed for pre-tagging\n      ✅ Easy PyTorch integration (torchvision)\n\nWhy batch size 32?\n      ✅ Conservative for demo (works on most GPUs)\n      ✅ Balances memory vs throughput\n      ✅ Safe for 960x544 images\n\nProduction: YOLOv8/TensorRT, batch 16-32",
            "duration": 10,
        },
        {
            "title": "Three Modes: Business Use Cases",
            "text": "Ref: docs/PARAMETERS.md\n\nfast: High-traffic, tight budget\n  - FPS: 0.5, Confidence: 0.6, Area: 2000px²\n  - 30-40% annotation volume\n\nbalanced: Standard production\n  - FPS: 1.0, Confidence: 0.5, Area: 1000px²\n  - 50-60% annotation volume\n\naccurate: Safety-critical, compliance\n  - FPS: 3.0, Confidence: 0.4, Area: 500px²\n  - 70-80% annotation volume\n\nFor details: See docs/PARAMETERS.md",
            "duration": 12,
        },
        {
            "title": "Design Rationale: Class-Aware (1/2)",
            "text": "Ref: docs/FEATURES.md (Stage 2 & 3)\n\nClass-specific confidence (Stage 2):\n    - Street-common (person, vehicles): 0.5 conf\n    - Street-rare (teddy bear): 0.7 conf",
            "duration": 8,
        },
        {
            "title": "Design Rationale: Class-Aware (2/2)",
            "text": "Class-specific area (Stage 3):\n    - People: 300px² (safety-critical)\n    - Equipment: 1000px² (context)\n\nRationale: Optimize for street monitoring\nCommon objects → better recall\nRare objects → fewer false positives",
            "duration": 8,
        },
        {
            "title": "Scaling: 24h Multi-Camera (1/2)",
            "text": "Ref: docs/SCALABILITY.md (Stage 1)\n\nParallelize by camera + time chunk:\n    - Each camera = independent worker\n    - 24h video split into 1h chunks\n    - Process 1000 cameras simultaneously",
            "duration": 8,
        },
        {
            "title": "Scaling: 24h Multi-Camera (2/2)",
            "text": "Distributed processing (Dask/Ray)\nStreaming architecture (Kinesis/Event Hubs)\nSmart sampling (motion detection, adaptive FPS)\n\nCost impact: 10x throughput, 60% storage reduction",
            "duration": 8,
        },
        {
            "title": "Scaling: GPU Infrastructure (1/2)",
            "text": "Ref: docs/SCALABILITY.md (Stage 2)\n\nDistributed inference:\n    - SageMaker/Azure ML endpoints\n    - Batch aggregation (32-64 images)\n    - Auto-scaling on queue depth\n    - Spot instances (70% cost savings)",
            "duration": 8,
        },
        {
            "title": "Scaling: GPU Infrastructure (2/2)",
            "text": "Inference serving: Triton on ECS/AKS\n\nCost: 83 GPU-hours/day for 1000 cameras\nResult: 5x throughput per GPU",
            "duration": 8,
        },
        {
            "title": "Scaling: Model Optimization",
            "text": "Ref: docs/FEATURES.md (Stage 2)\n\nCurrent: PyTorch, batch=4, single GPU\n\nProduction improvements:\n    - Batch size: 64-124 (3-4x throughput)\n    - YOLOv8/TensorRT (5-10x faster)\n    - Multi-GPU: 4 GPUs (4x throughput)\n    - Async DataLoader (overlap I/O)\n\nResult: 1.5h → 10-15min (90% reduction)",
            "duration": 10,
        },
        {
            "title": "Scaling: Edge Deployment (1/2)",
            "text": "Ref: docs/SCALABILITY.md (Stage 2)\n\nNVIDIA Jetson at camera edge:\n    - Local pre-filtering with TensorRT\n    - Only send 'interesting' frames to cloud\n    - Reduces bandwidth by 80-90%",
            "duration": 8,
        },
        {
            "title": "Scaling: Edge Deployment (2/2)",
            "text": "Architecture:\n    - Edge: Jetson with optimized model\n    - Cloud: Full processing + storage\n    - Hybrid: Best of both worlds\n\nCost impact: 70% bandwidth reduction",
            "duration": 8,
        },
        {
            "title": "Scaling: Infrastructure (1/2)",
            "text": "Ref: docs/FEATURES.md (Pipeline-Level)\n\nOrchestration:\n    - AWS Step Functions (serverless)\n    - Apache Airflow (DAG-based)\n    - Argo Workflows (Kubernetes-native)",
            "duration": 8,
        },
        {
            "title": "Scaling: Infrastructure (2/2)",
            "text": "Scaling Strategy:\n    - Stage 1: CPU-only, parallel per-video\n    - Stage 2: GPU batch, queue-based\n    - Stages 3-5: CPU-only, fast, parallel\n\nMonitoring: Prometheus, Grafana, CloudWatch",
            "duration": 8,
        },
        {
            "title": "Annotator Report: Overview",
            "text": "Ref: traceables/report.md\nGenerated by Stage 5 (generate_report.py)\n\nContains:\n        • Pipeline configuration used\n        • Dataset statistics\n        • Quality metrics\n        • Class distribution\n        • Annotation time estimates\n        • Business impact calculations\n\nPurpose: Guide annotators and stakeholders",
            "duration": 8,
        },
        {
            "title": f"Annotator Report: Dataset ({stats['num_images']} images)",
            "text": f"Ref: docs/report.md (generated by Stage 5):\n\n    - {stats['num_images']} images, {stats['num_annotations']} annotations ({stats['avg_per_image']:.2f} avg/image)\n    - Images with detections: {stats.get('images_with_anns', 0)}\n    - High-activity frames (>10 objects): {stats.get('high_activity', 0)}\n    - Class distribution ({stats['total_classes']} classes):\n"
            + "\n".join(
                [
                    f"        * {name}: {count} ({pct:.1f}%)"
                    for name, (count, pct) in list(stats["class_dist"].items())[:3]
                ]
            ),
            "duration": 8,
        },
        {
            "title": "Quality Metrics",
            "text": f"Dataset quality indicators:\n    - Avg bbox area: {stats['quality'].get('avg_bbox_area', 0):,.0f} px²\n    - Avg confidence: {stats['quality'].get('avg_confidence', 0):.2f}\n    - Low-confidence detections (<0.6): {stats['quality'].get('low_conf_count', 0)}\n\nInterpretation:\n        • Larger boxes = easier to annotate\n        • High confidence = model certainty\n        • Low-conf boxes need manual review",
            "duration": 9,
        },
        {
            "title": "Sample Images: What I See",
            "text": "Diverse scenarios detected:\n    - Crowded scenes (multiple people + vehicles)\n    - Mixed types (cars, people, buses)\n    - Varying densities across frames\n    - Quality issues: occlusion, lighting, small objects\n\nColor-coded for annotator guidance",
            "duration": 8,
        },
        {
            "type": "before_after",
            "frame_idx": 10,
            "title": "Example 5: Night Scene",
            "comment": "Multiple cars + people",
            "duration": 6,
        },
        {
            "type": "before_after",
            "frame_idx": 8,
            "title": "Example 6: Day Scene",
            "comment": "Far Multiple people + cars",
            "duration": 6,
        },
        {
            "title": f"Detected Objects Summary ({stats['num_images']} frames)",
            "text": f"From actual dataset:\n\nDetected classes:\n"
            + "\n".join(
                [
                    f"- {name.title()}: {count} ({pct:.1f}%{' - safety critical!' if name == 'person' else ''})"
                    for name, (count, pct) in list(stats["class_dist"].items())[:4]
                ]
            )
            + f"\n\nTotal: {stats['num_annotations']} annotations across {stats['num_images']} frames",
            "duration": 10,
        },
        {
            "title": "Interesting Failures & Edge Cases",
            "text": "Samples intentionally include:\n        • Low-confidence detections (40% of samples)\n        • Occlusion: Partial visibility, objects behind others\n        • Lighting: Dawn/dusk transitions, shadows\n        • Crowding: Overlapping objects, merged boxes\n        • Small/distant objects: Edge of detection limits\n\nWhy: Calibrate annotation standards\nIdentify systematic model weaknesses\nTrain annotators on edge cases",
            "duration": 10,
        },
        {
            "title": "Sample Color Coding",
            "text": "Ref: docs/FEATURES.md (Stage 4)\n\n        • Red boxes: People (safety-critical)\n        • Blue boxes: Equipment (forklift, truck)\n        • Green boxes: Other objects\n        • Orange boxes: Low confidence (<0.6)\n\nAlways enabled for QA\nPurpose: QA calibration before full dataset",
            "duration": 8,
        },
        {
            "title": "Annotator Empathy (1/2)",
            "text": "Report provides clear guidance:\n\n        1. Review orange boxes first (low confidence)\n        2. Verify red boxes (people = safety-critical)\n        3. Check high-activity frames",
            "duration": 6,
        },
        {
            "title": "Annotator Empathy (2/2)",
            "text": "4. Watch for: occlusion, lighting, PPE\n\nResult: Clear priorities, not guesswork\nAnnotators empowered with context",
            "duration": 6,
        },
        {
            "title": "Quality Checklist (1/2)",
            "text": "Before submitting annotations:\n\n        • All low-confidence detections (<0.5) reviewed\n        • Rare classes checked for missing instances\n        • High-activity frames reviewed\n        • Empty frames flagged if unexpected",
            "duration": 6,
        },
        {
            "title": "Quality Checklist (2/2)",
            "text": "• PPE annotations verified\n        • Bounding boxes are tight-fit\n        • No false positives (shadows, reflections)\n        • Class labels are correct\n\nFor more: See ANNOTATOR_GUIDE.md",
            "duration": 6,
        },
        {
            "title": "Annotation Priority Guide",
            "text": "Clear workflow for annotators:\n\n        1. Review orange boxes first (low confidence <0.6)\n        2. Verify red boxes (people = safety-critical)\n        3. Check high-activity frames (>5 objects)\n        4. Look for missing objects (especially PPE)\n5. Validate crowded scenes for merged detections\n\nResult: Systematic approach, not random checking",
            "duration": 6,
        },
        {
            "title": "Common Issues to Watch For",
            "text": "Systematic problems to identify:\n        • Occlusion: Workers behind equipment (low confidence)\n        • Lighting: False positives at dawn/dusk, shadows\n        • Crowding: Merged boxes in busy scenes\n        • Small objects: Missed fire extinguishers, tools\n        • PPE: Misclassified hard hats, missed safety vests\n        • Edge cases: Partial objects at image boundaries\n\nWhy: Helps calibrate annotation standards",
            "duration": 6,
        },
        {
            "title": "Why Edge Cases Were Selected",
            "text": "Strategic sampling rationale:\n        • 40% low-confidence detections (annotator focus)\n        • Diverse time coverage (early/mid/late video)\n        • Density stratification (low/med/high objects)\n        • Intentional failure modes included\n\nPurpose:\n    - Calibrate annotation standards\n    - Identify systematic model weaknesses\n    - Train annotators on challenging scenarios\n    - Prevent surprises in full dataset",
            "duration": 6,
        },
        {
            "title": "When to Flag for Review (1/2)",
            "text": "Annotators should escalate:\n\n        • Ambiguous cases (uncertain safety criteria)\n        • Systematic issues (pattern of errors)\n        • Camera problems (lighting/focus issues)",
            "duration": 6,
        },
        {
            "title": "When to Flag for Review (2/2)",
            "text": "• Missing context (need client clarification)\n        • Unusual detections (unexpected locations)\n\nThis empowers annotators to make decisions\nwhile knowing when to ask for help.\n\nFor more: See ANNOTATOR_GUIDE.md",
            "duration": 6,
        },
        {
            "title": "Efficiency Tips (1/2)",
            "text": "Maximize productivity:\n\n        • Batch similar frames (same time period)\n        • Use keyboard shortcuts (faster edits)\n        • Start with high-confidence (build intuition)\n        • Take breaks (avoid fatigue after 60-90 min)",
            "duration": 6,
        },
        {
            "title": "Efficiency Tips (2/2)",
            "text": "• Track metrics (monitor correction rate)\n\nResult: Faster, more accurate annotations\nHappier, more productive annotators\n\nFor more: See ANNOTATOR_GUIDE.md",
            "duration": 6,
        },
        {
            "title": "EM Perspective: Design (1/2)",
            "text": "Ref: docs/FEATURES.md\n\nKey Design Decisions:\n\n        1. Modular stages → Independent scaling\n        2. Config-driven → Per-client tuning\n        3. Class-aware filtering → Safety focus\n        4. Pipeline tracking → Full audit trail",
            "duration": 6,
        },
        {
            "title": "EM Perspective: Design (2/2)",
            "text": "Production Thinking:\n    - No code changes for tuning\n    - Clear business trade-offs (3 modes)\n    - Reproducible datasets\n    - Scalable architecture",
            "duration": 6,
        },
        {
            "title": "Key Features Implemented (1/2)",
            "text": "Ref: docs/FEATURES.md\n\nStage 1: Performance optimizations\n    - Downscaled QC (320x180): 3-4x faster\n    - Hash buffer (5 frames): Better dedup\n\nStage 2: Safety-relevant classes\n    - 10 classes vs 80 (70-80% noise reduction)",
            "duration": 6,
        },
        {
            "title": "Key Features Implemented (2/2)",
            "text": "Stage 3: Confidence score retention\n    - Low-conf highlighting (orange boxes)\n\nStage 4: Smart sampling\n    - 40% low-confidence (annotator focus)\n    - Density + time-of-day stratification\n\nStage 5: Pipeline config tracking\n    - Full reproducibility & audit trail",
            "duration": 6,
        },
        {
            "title": "Perceptual Hash Deduplication",
            "text": "Ref: docs/PERCEPTUAL_HASHING.md\n\nProblem: Pixel-wise MAD sensitive to:\n    - JPEG compression artifacts\n    - Slight camera movement/vibration\n    - Minor lighting changes\n\nSolution: Average hash (imagehash)\n    - 8x8 perceptual hash (64-bit)\n    - Hamming distance (0-64 range)\n    - Robust to compression & movement\n\nThresholds: fast=3, balanced=5, accurate=8\nResult: 96% accuracy vs 85% with MAD",
            "duration": 8,
        },
        {
            "title": "Production Optimizations (1/2)",
            "text": "Ref: docs/FEATURES.md (Stage 2)\n\nFor 24-hour video processing:\n\n        1. Batch size: 64 → 124 (saturate GPU)\n        2. Model: Faster R-CNN → YOLOv8 (5-10x faster)",
            "duration": 5,
        },
        {
            "title": "Production Optimizations (2/2)",
            "text": "3. Multi-GPU: 4 GPUs (4x throughput)\n        4. Async I/O: Overlap loading/inference\n\nResult: 1.5h → 10-15min (90% reduction)\n\nAt 1000 cameras: $456K/year GPU savings",
            "duration": 5,
        },
        {
            "title": "Business Impact: Real Numbers (1/2)",
            "text": f"This dataset: {stats['num_images']} images, {stats['num_annotations']} annotations\n\nEstimated annotation time:\n    - Manual (no pre-tags): ~10 min/image = 4.5h\n    - With pre-tagging: ~5 min/image = 2.3h\n    - Time savings: 2.2h (49% reduction)\n\nNote: Industry avg 5-15 min/image for manual annotation\nPre-tagging reduces review time by 40-60%",
            "duration": 8,
        },
        {
            "title": "Business Impact: Real Numbers (2/2)",
            "text": f"Cost savings @ $50/hour (industry standard):\n    - Per dataset: 2.2h × $50 = $110 saved\n    - At 1000 cameras/day: $110K/day\n    - Annual: $40M/year\n\nQualitative benefits:\n    - Faster time-to-market\n    - Higher annotation quality\n    - Consistent standards across annotators",
            "duration": 6,
        },
        {
            "title": "Business Impact Calculation (1/2)",
            "text": f"How we calculated savings:\n\nPer dataset ({stats['num_images']} images):\n        • Manual: 10 min/img × {stats['num_images']} = 4.5h\n        • With pre-tagging: 5 min/img × {stats['num_images']} = 2.3h\n        • Time saved: 2.2h (49% reduction)\n\nBasis: Industry standard 5-15 min/image\nPre-tagging cuts review time in half",
            "duration": 7,
        },
        {
            "title": "Business Impact Calculation (assumption)(2/2)",
            "text": "At scale (1000 cameras × daily):\n        • Cost per dataset: 2.2h × $50/h = $110\n        • Daily savings: $110 × 1000 = $110K\n        • Annual savings: $110K × 365 = $40M\n\nAssumptions:\n    - $50/hour = industry standard annotation cost\n    - 10 min/image manual, 5 min/image with pre-tags\n    - Based on actual dataset performance",
            "duration": 7,
        },
        {
            "title": "Eng. Manager Perspective: ROI (1/2)",
            "text": "Infrastructure Investment vs Savings:\n\nInfrastructure costs (assumption annual):\n        • GPU compute: ~$250K/year\n        • Storage (S3/Blob): ~$50K/year\n        • Engineering: 2 FTE = ~$400K/year\n        • Total: ~$700K/year",
            "duration": 6,
        },
        {
            "title": "Eng. Manager Perspective: ROI (2/2)",
            "text": "Annotation savings (assumption): $40M/year\n\nROI: 5,700% return\nPayback period: ~5 days\n\nThis is why pre-tagging is a no-brainer.\n\nNote: Conservative estimates based on 27-image dataset\nActual savings may be higher with larger datasets",
            "duration": 7,
        },
        {
            "title": "Key Features Summary: Code Quality",
            "text": "Ref: docs/FEATURES.md (Summary)\n\nCode Quality:\n    - Removed redundant logic (black_threshold)\n    - Fixed parameter wiring (max_images)\n    - Comprehensive documentation\n    - Added noise detection and elimination:\n        • Gaussian noise estimation\n        • Adaptive thresholding for noise removal\n        • Improved image quality for downstream tasks",
            "duration": 6,
        },
        {
            "title": "Key Features Summary:\nProduction Readiness",
            "text": "Ref: docs/FEATURES.md (Summary)\n\nProduction Readiness:\n    - Single runner script\n    - Config-driven (per-client tuning)\n    - Pipeline tracking (full audit trail)",
            "duration": 6,
        },
        {
            "title": "Key Features Summary: Business Value",
            "text": "Ref: docs/FEATURES.md (Summary)\n\nBusiness Value:\n    - ROI: $40M/year at scale (1000 cameras)\n    - 49% annotation time reduction\n    - Based on industry-standard annotation rates\n    - Conservative estimates from actual dataset",
            "duration": 6,
        },
        {
            "title": "Get Started",
            "text": "Run the pipeline:\n./runnable/run_pipeline.sh balanced data/timelapse_test.mp4\n\nSee docs/QUICKSTART.md for installation instructions",
            "duration": 5,
        },
        {
            "title": "Eng. Manager Perspective: Summary (1/2)",
            "text": "What makes this production-ready:\n        1. Business-focused: Clear ROI ($58M/year)\n        2. Modular: Independent scaling per stage\n        3. Observable: Full pipeline tracking\n        4. Tunable: Config-driven, no code changes",
            "duration": 8,
        },
        {
            "title": "Eng. Manager Perspective: Summary (2/2)",
            "text": "What makes this production-ready:\n        5. Scalable: 1000+ cameras ready\n6. Annotator-friendly: Clear guidance & QA\n\nThis demonstrates EM thinking:\n    - Cost awareness\n    - Stakeholder communication\n    - Production architecture\n    - Team scalability",
            "duration": 6,
        },
        {
            "title": "Thank You",
            "text": "Protex AI:\n  Making Industrial Workplaces Safer\n\nKey takeaways:\n        1. Config-driven: Per-client tuning, no code changes\n        2. Safety-focused: 10 relevant classes, not 80\n        3. Production-ready: Single script, full tracking\n        4. Scale-aware: $40M/year savings at 1000 cameras\n\nReady to discuss P R O D U C T I O N I Z A T I O N!",
            "duration": 6,
        },
    ]

    print(f"[STEP] Generating {len(slides)} slides...")

    # Create slide images
    slide_paths = []
    for i, slide in enumerate(slides, 1):
        if slide.get("type") == "before_after":
            idx = slide["frame_idx"]
            if (
                frame_images
                and sample_images
                and idx < len(frame_images)
                and idx < len(sample_images)
            ):
                try:
                    img = create_before_after_slide(
                        frame_images[idx],
                        sample_images[idx],
                        slide["title"],
                        slide.get("comment", ""),
                        i,
                    )
                except Exception as e:
                    print(f"[WARN] Failed to create anno/not slide {i}: {e}")
                    img = create_slide(
                        f"Images not available\n\nRun pipeline first:\n./runnable/run_pipeline.sh balanced video.mp4",
                        slide["title"],
                        i,
                    )
            else:
                img = create_slide(
                    f"Images not available\n\nRun pipeline first:\n./runnable/run_pipeline.sh balanced video.mp4",
                    slide["title"],
                    i,
                )
        elif slide.get("type") == "image":
            path_ref = slide["path"]
            comment = slide.get("comment", "")
            if isinstance(path_ref, str) and Path(path_ref).exists():
                img = create_image_slide(path_ref, slide["title"], i, comment)
            elif isinstance(path_ref, int) and path_ref < len(sample_images):
                img = create_image_slide(
                    sample_images[path_ref], slide["title"], i, comment
                )
            else:
                img = create_slide("Image not available", slide["title"], i)
        else:
            img = create_slide(slide["text"], slide["title"], i)

        path = slides_dir / f"slide_{i:02d}.png"
        img.save(path)
        slide_paths.append((path, slide.get("duration", 10)))
        print(f"[WRITE] {path}")

    # Create video clips
    print(f"\n[STEP] Creating video clips...")
    clips = []
    for path, duration in slide_paths:
        clip = ImageClip(str(path)).set_duration(duration)
        clips.append(clip)

    # Concatenate clips
    print(f"[STEP] Concatenating {len(clips)} clips...")
    video = concatenate_videoclips(clips, method="compose")

    # Export video
    print(f"[STEP] Exporting video to {output_path}...")
    total_duration = sum(d for _, d in slide_paths)
    video.write_videofile(
        output_path,
        fps=5,
        verbose=False,
        logger="bar",
        codec="libx264",
        bitrate="75k",
    )

    print(f"\n      ✅ Presentation generated!")
    print(f"   Duration: {total_duration} seconds (~{total_duration/60:.1f} minutes)")
    print(f"   Output: {output_path}")
    print(f"   Slides: {slides_dir}/")

    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        output = sys.argv[1]
        if not output.startswith(("/", ".")):
            output = f"traceables/{output}"
    else:
        output = "traceables/protex_presentation.mp4"

    print("=" * 60)
    print(" Protex AI - Presentation Video Generator")
    print("=" * 60)
    print()

    if not MOVIEPY_AVAILABLE:
        print("[ERROR] moviepy not installed")
        print("Install with: pip install moviepy pillow")
        sys.exit(1)

    success = create_presentation_video(output)
    sys.exit(0 if success else 1)
