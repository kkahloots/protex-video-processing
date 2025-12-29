#!/usr/bin/env python
"""
Stage 8: Prepare submission.zip file for Protex AI assignment.

Creates a zip file containing:
1. Report (markdown + JSON)
2. Scripts (all Python files)
3. 20 random annotated samples
4. Documentation

Usage:
    python 08_prepare_submission.py
"""

import zipfile
from pathlib import Path
import random
import json


def prepare_submission():
    """Create submission.zip with all required deliverables."""

    submission_dir = Path("traceables/submission")
    submission_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous submission
    if (submission_dir / "submission.zip").exists():
        (submission_dir / "submission.zip").unlink()

    print("=" * 60)
    print(" Protex AI - Submission Preparation")
    print("=" * 60)
    print()

    # Check required files exist
    required_files = [
        "traceables/report/stats/report.json",
        "traceables/report/text/report.md",
        "traceables/samples",
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print("[ERROR] Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nRun the full pipeline first:")
        print("./runnable/run_pipeline.sh balanced data/timelapse_test.mp4")
        return False

    # Create zip file
    zip_path = submission_dir / "submission.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:

        # 1. Add report files
        print("[STEP] Adding reports...")
        zipf.write("traceables/report/text/report.md", "report.md")
        zipf.write("traceables/report/stats/report.json", "report.json")

        # 2. Add all Python scripts
        print("[STEP] Adding scripts...")
        python_files = [
            "00_setup_env.py",
            "01_data_preprocessing.py",
            "02_data_pretagging.py",
            "03_pretag_cleanup.py",
            "04_generate_samples.py",
            "05_generate_report.py",
            "06_generate_presentation.py",
            "07_generate_annotated_video.py",
            "08_prepare_submission.py",
        ]

        for script in python_files:
            if Path(script).exists():
                zipf.write(script, f"scripts/{script}")

        # Add utils
        utils_dir = Path("utils")
        if utils_dir.exists():
            for util_file in utils_dir.glob("*.py"):
                zipf.write(util_file, f"scripts/utils/{util_file.name}")

        # Add config
        if Path("config.yaml").exists():
            zipf.write("config.yaml", "scripts/config.yaml")

        # 3. Add 20 random annotated samples with visible annotations
        print("[STEP] Adding 20 random annotated samples...")
        samples_dir = Path("traceables/samples")
        sample_files = list(samples_dir.glob("sample_*.jpg"))

        if not sample_files:
            print("[ERROR] No annotated samples found! Run Stage 4 first:")
            print("  python 04_generate_samples.py --num_samples 20")
            return False

        if len(sample_files) < 20:
            print(f"[WARN] Only {len(sample_files)} samples available, adding all")
            selected_samples = sample_files
        else:
            selected_samples = random.sample(sample_files, 20)

        # Add sample catalog for context
        catalog_path = samples_dir / "SAMPLE_CATALOG.md"
        if catalog_path.exists():
            zipf.write(catalog_path, "samples/SAMPLE_CATALOG.md")

        for i, sample_file in enumerate(selected_samples, 1):
            zipf.write(sample_file, f"samples/sample_{i:02d}.jpg")

        print(
            f"[INFO] Added {len(selected_samples)} annotated samples with visible bounding boxes"
        )
        print(
            "[INFO] Samples include diverse scenarios: low-confidence, crowded scenes, edge cases"
        )

        # 4. Add key documentation
        print("[STEP] Adding documentation...")
        doc_files = [
            "README.md",
            "docs/QUICKSTART.md",
            "docs/FEATURES.md",
            "docs/PARAMETERS.md",
            "docs/PRESENTATION_GUIDE.md",
        ]

        for doc_file in doc_files:
            if Path(doc_file).exists():
                zipf.write(doc_file, f"docs/{Path(doc_file).name}")

        # 5. Add presentation video if exists
        presentation_files = list(Path("traceables").glob("*presentation*.mp4"))
        if presentation_files:
            print("[STEP] Adding presentation video...")
            zipf.write(presentation_files[0], "presentation.mp4")

        # 6. Add pipeline config for traceability
        if Path("traceables/pipeline_config.json").exists():
            zipf.write("traceables/pipeline_config.json", "pipeline_config.json")

    # Generate submission summary
    summary = {
        "submission_info": {
            "assignment": "Protex AI - Computer Vision Ops Pipeline",
            "candidate": "Kal Kahloot",
            "submission_date": "2024",
            "total_files": len(zipf.namelist()) if "zipf" in locals() else 0,
        },
        "deliverables": {
            "1_report": "report.md + report.json",
            "2_presentation_video": (
                "presentation.mp4" if presentation_files else "Generated separately"
            ),
            "3_scripts": f"{len(python_files)} Python scripts + utils + config",
            "4_samples": f"{len(selected_samples)} annotated samples with visible bounding boxes",
        },
        "pipeline_stages": {
            "Stage 1": "Data Preprocessing (video → frames)",
            "Stage 2": "Data Pre-tagging (frames → COCO detections)",
            "Stage 3": "Pre-tag Cleanup (filtered COCO)",
            "Stage 4": "Generate Samples (annotated images)",
            "Stage 5": "Generate Report (dataset summary)",
            "Stage 6": "Generate Presentation (MP4 video)",
            "Stage 7": "Generate Annotated Video (optional)",
            "Stage 8": "Prepare Submission (this script)",
        },
    }

    # Save summary
    summary_path = submission_dir / "submission_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Submission prepared!")
    print(f"   File: {zip_path}")
    print(f"   Size: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"   Files: {summary['submission_info']['total_files']}")
    print(f"   Summary: {summary_path}")

    print(f"\n📋 Deliverables checklist:")
    print(f"   ✅ Report (markdown + JSON)")
    print(f"   ✅ Scripts (8 stages + utils + config)")
    print(f"   ✅ Samples ({len(selected_samples)} annotated images)")
    print(f"   ✅ Documentation (README + guides)")
    print(f"   {'✅' if presentation_files else '⚠️ '} Presentation video")

    if not presentation_files:
        print(f"\n💡 Generate presentation video with:")
        print(f"   python 06_generate_presentation.py")

    return True


if __name__ == "__main__":
    success = prepare_submission()
    exit(0 if success else 1)
