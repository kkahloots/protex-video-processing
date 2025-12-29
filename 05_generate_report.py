#!/usr/bin/env python
"""
Generate structured dataset report with organized outputs.
Creates folders: text/, images/, stats/ under traceables/report/
"""

import argparse
import json
from pathlib import Path
from collections import Counter
from utils.config_loader import get_config

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def create_report_structure():
    """Create organized folder structure for report outputs."""
    base = Path("traceables/report")
    folders = {
        "text": base / "text",
        "images": base / "images", 
        "stats": base / "stats",
        "base": base
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def load_coco(path: str):
    with open(path, "r") as f:
        return json.load(f)


def generate_pipeline_summary(pipeline_config):
    """Generate pipeline steps summary with statistics."""
    summary = {
        "steps": [],
        "total_reduction": {}
    }
    
    # Load frames metadata for Stage 1 stats
    frames_meta = {}
    frames_meta_path = Path("traceables/frames/frames_metadata.json")
    if frames_meta_path.exists():
        with open(frames_meta_path, "r") as f:
            frames_meta = json.load(f)
    
    if "preprocessing" in pipeline_config:
        p = pipeline_config["preprocessing"]
        step = {
            "stage": "1. Preprocessing",
            "input": "Raw video",
            "output": "Filtered frames",
            "filters": [
                f"FPS sampling: {p.get('desired_fps', 'N/A')}",
                f"Min brightness: {p.get('min_brightness', 'N/A')}",
                f"Min sharpness: {p.get('min_laplacian_var', 'N/A')}",
                "Perceptual hash deduplication"
            ]
        }
        if frames_meta:
            total = frames_meta.get('total_video_frames', 0)
            saved = frames_meta.get('num_saved_frames', 0)
            step["statistics"] = {
                "total_video_frames": total,
                "saved_frames": saved,
                "reduction_pct": round(100 * (1 - saved/max(1,total)), 1)
            }
        summary["steps"].append(step)
    
    if "pretagging" in pipeline_config:
        p = pipeline_config["pretagging"]
        step = {
            "stage": "2. Pre-tagging",
            "input": "Filtered frames",
            "output": "COCO detections",
            "model": p.get('model_name', 'N/A'),
            "batch_size": p.get('batch_size', 'N/A'),
            "confidence": p.get('min_confidence', 'N/A'),
            "nms_iou": p.get('nms_iou_threshold', 'N/A')
        }
        if 'raw_detections' in p:
            step["statistics"] = {
                "raw_detections": p['raw_detections'],
                "filtered_low_conf": p.get('filtered_low_conf', 0),
                "kept_detections": p['raw_detections'] - p.get('filtered_low_conf', 0)
            }
        summary["steps"].append(step)
    
    if "cleanup" in pipeline_config:
        p = pipeline_config["cleanup"]
        stats = p.get('filtering_stats', {})
        summary["steps"].append({
            "stage": "3. Cleanup",
            "input": "Raw detections",
            "output": "Cleaned detections",
            "filters": [
                f"Min area: {p.get('min_area', 'N/A')} px²",
                f"Min score: {p.get('min_score', 'N/A')}",
                f"Person area: {p.get('min_area_person', 'N/A')} px²"
            ],
            "statistics": {
                "original": stats.get('original', 0),
                "removed_small": stats.get('removed_small', 0),
                "removed_score": stats.get('removed_score', 0),
                "removed_masked": stats.get('removed_masked', 0),
                "kept": stats.get('kept', 0)
            }
        })
    
    return summary


def generate_report(coco_path: str, folders: dict, verbose: bool = True):
    coco = load_coco(coco_path)
    
    # Load pipeline config
    pipeline_config = {}
    config_path = Path("traceables/pipeline_config.json")
    if config_path.exists():
        with open(config_path, "r") as f:
            pipeline_config = json.load(f)
    
    # Generate pipeline summary
    pipeline_summary = generate_pipeline_summary(pipeline_config)
    
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    
    # Statistics
    cat_counts = Counter(ann["category_id"] for ann in annotations)
    cat_by_id = {c["id"]: c["name"] for c in categories}
    img_ann_counts = Counter(ann["image_id"] for ann in annotations)
    anns_per_img = [img_ann_counts.get(img["id"], 0) for img in images]
    
    # Quality metrics
    areas = [ann.get("area", 0) for ann in annotations]
    confidences = [ann.get("score", ann.get("confidence", 0)) 
                   for ann in annotations if "score" in ann or "confidence" in ann]
    
    # Build report
    report = {
        "pipeline_summary": pipeline_summary,
        "dataset_summary": {
            "total_images": len(images),
            "total_annotations": len(annotations),
            "annotations_per_image": round(sum(anns_per_img) / len(anns_per_img), 2) if anns_per_img else 0,
            "images_with_annotations": sum(1 for c in anns_per_img if c > 0),
            "images_no_detections": sum(1 for c in anns_per_img if c == 0),
            "images_high_activity": sum(1 for c in anns_per_img if c > 10),
        },
        "quality_metrics": {
            "avg_bbox_area_px2": round(sum(areas) / len(areas), 1) if areas else 0,
            "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else "N/A",
            "low_confidence_count": sum(1 for c in confidences if c < 0.5) if confidences else "N/A",
            "small_boxes_remaining": sum(1 for a in areas if a < 1000),
        },
        "class_distribution": [
            {"class": cat_by_id.get(cid, f"id_{cid}"), "count": count,
             "percentage": round(count / len(annotations) * 100, 1)}
            for cid, count in cat_counts.most_common()
        ],
    }
    
    # Save structured outputs
    with open(folders["stats"] / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    with open(folders["stats"] / "pipeline_summary.json", "w") as f:
        json.dump(pipeline_summary, f, indent=2)
    
    # Generate markdown
    generate_markdown(report, folders["text"])
    
    # Generate visualizations
    if MATPLOTLIB_AVAILABLE:
        generate_visualizations(cat_counts, cat_by_id, anns_per_img, folders["images"])
    
    if verbose:
        print(f"[WRITE] Report structure created in: {folders['base']}")
        print(f"  - Text reports: {folders['text']}")
        print(f"  - Visualizations: {folders['images']}")
        print(f"  - Statistics: {folders['stats']}")
    
    return report


def generate_markdown(report, text_folder):
    """Generate markdown reports."""
    
    # Main report
    with open(text_folder / "report.md", "w") as f:
        f.write("# Protex AI - Dataset Report\n\n")
        
        # Pipeline Summary
        f.write("## Pipeline Summary\n\n")
        for step in report["pipeline_summary"]["steps"]:
            f.write(f"### {step['stage']}\n")
            f.write(f"- **Input**: {step['input']}\n")
            f.write(f"- **Output**: {step['output']}\n")
            if "filters" in step:
                f.write("- **Filters**:\n")
                for filt in step["filters"]:
                    f.write(f"  - {filt}\n")
            if "statistics" in step:
                stats = step["statistics"]
                if 'kept' in stats:  # Stage 3
                    f.write(f"- **Statistics**: {stats['kept']}/{stats['original']} kept ")
                    f.write(f"({100*stats['kept']/max(1,stats['original']):.0f}%)\n")
                elif 'saved_frames' in stats:  # Stage 1
                    f.write(f"- **Statistics**: {stats['saved_frames']}/{stats['total_video_frames']} frames ")
                    f.write(f"({stats['reduction_pct']}% reduction)\n")
                elif 'raw_detections' in stats:  # Stage 2
                    f.write(f"- **Statistics**: {stats['kept_detections']}/{stats['raw_detections']} detections kept\n")
            f.write("\n")
        
        # Dataset Overview
        f.write("## Dataset Overview\n\n")
        ds = report["dataset_summary"]
        f.write(f"- Total Images: {ds['total_images']}\n")
        f.write(f"- Total Annotations: {ds['total_annotations']}\n")
        f.write(f"- Avg Annotations/Image: {ds['annotations_per_image']}\n")
        f.write(f"- High Activity Frames: {ds['images_high_activity']}\n\n")
        
        # Class Distribution
        f.write("## Class Distribution\n\n")
        f.write("| Class | Count | % |\n|-------|-------|---|\n")
        for item in report["class_distribution"][:10]:
            f.write(f"| {item['class']} | {item['count']} | {item['percentage']}% |\n")
        
        # Known Issues
        f.write("\n## Known Issues & Edge Cases\n\n")
        f.write("The following challenges are present in this dataset:\n\n")
        f.write("- **Low-confidence detections**: Some boxes have confidence < 0.6 (require review)\n")
        f.write("- **Occlusion**: Partial visibility of objects behind others\n")
        f.write("- **Lighting variations**: Dawn/dusk transitions may affect detection quality\n")
        f.write("- **Small objects**: Objects < 1000px² may be at detection limits\n")
        f.write("- **Crowded scenes**: Overlapping objects may have merged bounding boxes\n\n")
        f.write("**Recommendation**: Review orange-boxed (low-confidence) detections first. "
                "These represent the model's uncertainty and require careful annotation.\n")
    
    # Pipeline steps detail
    with open(text_folder / "pipeline_steps.md", "w") as f:
        f.write("# Pipeline Steps Detail\n\n")
        for step in report["pipeline_summary"]["steps"]:
            f.write(f"## {step['stage']}\n\n")
            f.write(f"**Transformation**: {step['input']} → {step['output']}\n\n")
            if "filters" in step:
                f.write("**Applied Filters**:\n")
                for filt in step["filters"]:
                    f.write(f"- {filt}\n")
                f.write("\n")
            if "statistics" in step:
                f.write("**Statistics**:\n")
                for key, val in step["statistics"].items():
                    f.write(f"- {key}: {val}\n")
                f.write("\n")


def generate_visualizations(cat_counts, cat_by_id, anns_per_img, images_folder):
    """Generate visualization charts."""
    
    # Class distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    top_cats = cat_counts.most_common(10)
    names = [cat_by_id.get(cid, f"id_{cid}") for cid, _ in top_cats]
    counts = [count for _, count in top_cats]
    ax.barh(names, counts, color="steelblue")
    ax.set_xlabel("Count")
    ax.set_title("Top 10 Classes")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(images_folder / "class_distribution.png", dpi=150)
    plt.close()
    
    # Annotation density
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(anns_per_img, bins=20, edgecolor="black", color="coral")
    ax.set_xlabel("Annotations per Image")
    ax.set_ylabel("Frequency")
    ax.set_title("Annotation Density")
    plt.tight_layout()
    plt.savefig(images_folder / "annotation_density.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate structured report")
    parser.add_argument("--input_json", type=str, 
                       default=get_config("cleanup.output_json", "traceables/pre_tags/pre_tags_cleaned.json"))
    parser.add_argument("--verbose", action="store_true")
    args, _ = parser.parse_known_args()
    
    verbose = get_config("defaults.verbose", True) or args.verbose
    
    if verbose:
        print("=" * 60)
        print(" Protex AI – Stage 5: Report Generation")
        print("=" * 60)
    
    folders = create_report_structure()
    report = generate_report(args.input_json, folders, verbose)
    
    # Copy reports to docs/ for documentation
    import shutil
    docs_report = Path("docs/report.md")
    docs_pipeline = Path("docs/pipeline_steps.md")
    runs_report = folders["text"] / "report.md"
    runs_pipeline = folders["text"] / "pipeline_steps.md"
    
    if runs_report.exists():
        shutil.copy(runs_report, docs_report)
        if verbose:
            print(f"[COPY] Report copied to {docs_report}")
    
    if runs_pipeline.exists():
        shutil.copy(runs_pipeline, docs_pipeline)
        if verbose:
            print(f"[COPY] Pipeline steps copied to {docs_pipeline}")
    
    if verbose:
        print("\n✅ Report generation completed")
        print("=" * 60)


if __name__ == "__main__":
    main()
