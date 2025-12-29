#!/bin/bash
# Protex AI - Pipeline Runner (Linux/macOS)
# Usage: ./run_pipeline.sh [mode] [video_path]

set -e  # Exit on error

# Change to parent directory
cd "$(dirname "$0")/.."

# Configuration
MODE="${1:-balanced}"
VIDEO_PATH="${2:-data/timelapse_test.mp4}"
BATCH_SIZE=16
NUM_SAMPLES=20

echo "============================================================"
echo " Protex AI - Computer Vision Pipeline"
echo "============================================================"
echo "Mode: $MODE"
echo "Video: $VIDEO_PATH"
echo "============================================================"
echo ""

# Check if video exists
if [ ! -f "$VIDEO_PATH" ]; then
    echo "[ERROR] Video file not found: $VIDEO_PATH"
    exit 1
fi

# Start timer
START_TIME=$(date +%s)

# Stage 1: Preprocessing
echo "============================================================"
echo "[Stage 1] Preprocessing (Video -> Frames)"
echo "============================================================"
python 01_data_preprocessing.py \
    --video_path "$VIDEO_PATH" \
    --mode "$MODE" \
    --verbose

# Stage 2: Pre-tagging
echo ""
echo "============================================================"
echo "[Stage 2] Pre-tagging (Frames -> COCO Detections)"
echo "============================================================"
python 02_data_pretagging.py \
    --mode "$MODE" \
    --batch_size "$BATCH_SIZE" \
    --verbose

# Stage 3: Cleanup
echo ""
echo "============================================================"
echo "[Stage 3] Cleanup (COCO -> Cleaned COCO)"
echo "============================================================"
python 03_pretag_cleanup.py \
    --mode "$MODE" \
    --verbose

# Stage 4: Sample Generation
echo ""
echo "============================================================"
echo "[Stage 4] Sample Generation"
echo "============================================================"
python 04_generate_samples.py \
    --num_samples "$NUM_SAMPLES" \
    --verbose

# Stage 5: Report Generation
echo ""
echo "============================================================"
echo "[Stage 5] Report Generation"
echo "============================================================"
python 05_generate_report.py \
    --verbose

# # Stage 6: Presentation Generation
echo ""
echo "============================================================"
echo "[Stage 6] Presentation Generation"
echo "============================================================"
python 06_generate_presentation.py protex_presentation.mp4 \
    --verbose

echo ""
echo "============================================================"
echo "[Stage 7] Annotated Video Generation"
echo "============================================================"
python 07_generate_annotated_video.py \
    --output traceables/sample_annotated_video.mp4 \
    --fps 2.0 \
    --verbose

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "[SUCCESS] Pipeline completed successfully!"
echo "============================================================"
echo "Total time: ${TOTAL_TIME}s"
echo ""
echo "Outputs:"
echo "  - Frames: traceables/frames/"
echo "  - COCO: traceables/pre_tags/pre_tags_cleaned.json"
echo "  - Samples: traceables/samples/"
echo "  - Report: traceables/report/"
echo "  - Presentation: traceables/protex_presentation.mp4"
echo "  - Annotated Video: traceables/sample_annotated_video.mp4"
echo "============================================================"
