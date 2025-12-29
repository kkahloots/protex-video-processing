#!/usr/bin/env python
"""
Extract presentation text and generate needed images from 06_generate_presentation.py
Creates a folder with all presentation content for easy review.
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def load_report_stats():
    """Load stats from report.json."""
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
        with open(report_path) as f:
            report = json.load(f)

        ds = report.get("dataset_summary", {})
        stats["num_images"] = ds.get("total_images", 0)
        stats["num_annotations"] = ds.get("total_annotations", 0)
        stats["avg_per_image"] = ds.get("annotations_per_image", 0.0)
        stats["images_with_anns"] = ds.get("images_with_annotations", 0)
        stats["high_activity"] = ds.get("images_high_activity", 0)

        class_dist = report.get("class_distribution", [])
        stats["class_dist"] = {
            c["class"]: (c["count"], c["percentage"]) for c in class_dist
        }
        stats["total_classes"] = len(class_dist)

        qm = report.get("quality_metrics", {})
        stats["quality"] = {
            "avg_bbox_area": qm.get("avg_bbox_area_px2", 0),
            "avg_confidence": qm.get("avg_confidence", 0),
            "low_conf_count": qm.get("low_confidence_count", 0),
        }

        ps = report.get("pipeline_summary", {}).get("steps", [])
        if len(ps) >= 3:
            stats["pipeline"] = {
                "stage1_saved": ps[0].get("statistics", {}).get("saved_frames", 0),
                "stage1_total": ps[0].get("statistics", {}).get("total_video_frames", 0),
                "stage1_reduction": ps[0].get("statistics", {}).get("reduction_pct", 0),
                "stage2_raw": ps[1].get("statistics", {}).get("raw_detections", 0),
                "stage2_kept": ps[1].get("statistics", {}).get("kept_detections", 0),
                "stage3_kept": ps[2].get("statistics", {}).get("kept", 0),
                "stage3_removed": ps[2].get("statistics", {}).get("removed_small", 0),
            }

    return stats


def create_simple_slide(text, title, slide_num, width=1920, height=1080):
    """Create a simple text slide."""
    img = Image.new("RGB", (width, height), (20, 20, 40))
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 70)
        text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
    except:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()

    draw.text((100, 100), title, fill=(0, 200, 255), font=title_font)
    y_offset = 300
    for line in text.split("\n"):
        draw.text((100, y_offset), line, fill=(255, 255, 255), font=text_font)
        y_offset += 80

    draw.text((width - 200, height - 100), f"Slide {slide_num}", fill=(100, 100, 100), font=text_font)
    return img


def generate_presentation_content():
    """Generate presentation text and images."""
    
    output_dir = Path("presentation_content")
    output_dir.mkdir(exist_ok=True)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    stats = load_report_stats()
    
    # Define all slides with their content
    slides = [
        {
            "title": "Protex AI: Industrial Safety Computer Vision Pipeline",
            "text": "Kal Kahloot – Engineering Manager Assignment\n\nStory: From 24h of raw timelapse factory footage to a curated,\nannotator-ready dataset using a 4-stage, production-oriented\nCV pipeline for industrial safety monitoring.",
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
            "title": "Stage 1: Preprocessing",
            "text": "60-80% frame reduction via FPS sampling, brightness, blur, duplicates",
            "image_ref": "docs/imgs/stage1.png",
            "duration": 8,
        },
        {
            "title": "Stage 2: Pre-tagging",
            "text": "Faster R-CNN inference with batch processing and GPU acceleration",
            "image_ref": "docs/imgs/stage2.png",
            "duration": 8,
        },
        {
            "title": "Stage 3: Cleanup",
            "text": "Class-aware filtering: People 300px², Equipment 1000px²",
            "image_ref": "docs/imgs/stage3.png",
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
            "title": "Business Impact: Real Numbers",
            "text": f"This dataset: {stats['num_images']} images, {stats['num_annotations']} annotations\n\nEstimated annotation time:\n    - Manual (no pre-tags): ~10 min/image = 4.5h\n    - With pre-tagging: ~5 min/image = 2.3h\n    - Time savings: 2.2h (49% reduction)\n\nCost savings @ $50/hour:\n    - Per dataset: $110 saved\n    - At 1000 cameras/day: $110K/day\n    - Annual: $40M/year",
            "duration": 8,
        },
        {
            "title": "Thank You",
            "text": "Protex AI: Making Industrial Workplaces Safer\n\nKey takeaways:\n        1. Config-driven: Per-client tuning, no code changes\n        2. Safety-focused: 10 relevant classes, not 80\n        3. Production-ready: Single script, full tracking\n        4. Scale-aware: $40M/year savings at 1000 cameras\n\nReady to discuss PRODUCTIONIZATION!",
            "duration": 6,
        },
    ]
    
    # Generate text file with all slide content
    text_output = []
    text_output.append("=" * 80)
    text_output.append("PROTEX AI - PRESENTATION CONTENT")
    text_output.append("=" * 80)
    text_output.append("")
    
    for i, slide in enumerate(slides, 1):
        text_output.append(f"\n{'='*80}")
        text_output.append(f"SLIDE {i}: {slide['title']}")
        text_output.append(f"Duration: {slide['duration']} seconds")
        text_output.append(f"{'='*80}")
        text_output.append("")
        text_output.append(slide['text'])
        
        if 'image_ref' in slide:
            text_output.append(f"\n[IMAGE: {slide['image_ref']}]")
        
        text_output.append("")
    
    # Write text file
    text_file = output_dir / "presentation_text.txt"
    with open(text_file, 'w') as f:
        f.write('\n'.join(text_output))
    
    print(f"✅ Generated: {text_file}")
    
    # Generate JSON with structured data
    json_output = {
        "presentation_title": "Protex AI - Industrial Safety Computer Vision Pipeline",
        "total_slides": len(slides),
        "total_duration_seconds": sum(s['duration'] for s in slides),
        "slides": [
            {
                "slide_number": i,
                "title": s['title'],
                "content": s['text'],
                "duration": s['duration'],
                "image_reference": s.get('image_ref', None)
            }
            for i, s in enumerate(slides, 1)
        ],
        "statistics": stats
    }
    
    json_file = output_dir / "presentation_data.json"
    with open(json_file, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print(f"✅ Generated: {json_file}")
    
    # Generate sample slide images
    print("\nGenerating sample slide images...")
    for i, slide in enumerate(slides[:5], 1):  # First 5 slides as samples
        img = create_simple_slide(slide['text'], slide['title'], i)
        img_path = images_dir / f"slide_{i:02d}_sample.png"
        img.save(img_path)
        print(f"  ✅ {img_path}")
    
    # Copy referenced images if they exist
    print("\nCopying referenced images...")
    for slide in slides:
        if 'image_ref' in slide:
            src = Path(slide['image_ref'])
            if src.exists():
                import shutil
                dst = images_dir / src.name
                shutil.copy(src, dst)
                print(f"  ✅ {dst}")
    
    # Generate summary
    summary = f"""
{'='*80}
PRESENTATION CONTENT GENERATED
{'='*80}

Output Directory: {output_dir.absolute()}

Files Generated:
  • presentation_text.txt - Full text of all slides
  • presentation_data.json - Structured JSON data
  • images/ - Sample slide images and referenced images

Presentation Stats:
  • Total Slides: {len(slides)}
  • Total Duration: {sum(s['duration'] for s in slides)} seconds (~{sum(s['duration'] for s in slides)/60:.1f} minutes)
  • Sample Images: {len(list(images_dir.glob('*.png')))}

Dataset Stats (from report.json):
  • Images: {stats['num_images']}
  • Annotations: {stats['num_annotations']}
  • Classes: {stats['total_classes']}

Next Steps:
  1. Review presentation_text.txt for all slide content
  2. Check presentation_data.json for structured data
  3. View sample images in images/ folder
  4. Run 06_generate_presentation.py to create full video

{'='*80}
"""
    
    print(summary)
    
    summary_file = output_dir / "README.txt"
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    return True


if __name__ == "__main__":
    print("=" * 80)
    print(" Protex AI - Presentation Content Generator")
    print("=" * 80)
    print()
    
    success = generate_presentation_content()
    
    if success:
        print("\n✅ SUCCESS: Presentation content generated!")
        print("\nView the content in: presentation_content/")
    else:
        print("\n❌ ERROR: Failed to generate presentation content")
