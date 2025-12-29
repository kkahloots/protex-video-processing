# Protex AI - Quick Start Guide

**Get the pipeline running in 5 minutes.**

## Quick Summary

1. **Setup**: `python 00_setup_env.py && source .venv/bin/activate`
2. **Run**: `./runnable/run_pipeline.sh balanced data/timelapse_test.mp4`
3. **Review**: Check `traceables/report.md` and `traceables/samples/`

**Processing time**: ~5 minutes (GPU) or ~20 minutes (CPU)

---

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- (Optional) GPU with CUDA for faster inference

---

## Installation

### 1. Clone or Download Repository

```bash
cd protex-video-processing
```

### 2. Set Up Environment

**Linux/macOS:**
```bash
python 00_setup_env.py
source .venv/bin/activate
```

**Windows:**
```cmd
python 00_setup_env.py
.venv\Scripts\activate
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

---

## Running the Pipeline

### Option 1: Run All Stages (Recommended)

**Linux/macOS:**
```bash
./runnable/run_pipeline.sh balanced data/timelapse_test.mp4
```

**Background processing (Linux/macOS):**
```bash
nohup runnable/run_pipeline.sh balanced data/timelapse_test.mp4 > runnable/run_pipeline.log 2>&1 &

# Check progress
tail -f runnable/run_pipeline.log
```

**Windows:**
```cmd
runnable\run_pipeline.bat balanced input\timelapse_test.mp4
```

**Parameters:**
- First argument: Mode (`fast`, `balanced`, or `accurate`)
- Second argument: Path to video file

### Option 2: Run Individual Stages

```bash
# Stage 1: Preprocessing
python 01_data_preprocessing.py --video_path data/timelapse_test.mp4 --mode balanced --verbose

# Stage 2: Pre-tagging
python 02_data_pretagging.py --images_dir frames --mode balanced --verbose

# Stage 3: Cleanup
python 03_pretag_cleanup.py --mode balanced --verbose

# Stage 4: Sample Generation
python 04_generate_samples.py --num_samples 20 --verbose

# Stage 5: Report Generation
python 05_generate_report.py --verbose

# Stage 6: Presentation Video
python 06_generate_presentation.py protex_presentation.mp4
```

### Option 3: Google Colab

1. Upload `runnable/run_pipeline.ipynb` to Google Colab
2. Upload your video to Colab or mount Google Drive
3. Run all cells sequentially

---

## Expected Outputs

After running the pipeline, you should see:

```
protex-video-processing/
└── traceables/                        # All generated outputs
    ├── frames/                      # Extracted and filtered frames
    │   ├── frame_00000.jpg
    │   ├── frame_00001.jpg
    │   └── frames_metadata.json
    ├── pre_tags/                    # COCO annotations
    │   ├── pre_tags_raw.json       # Raw detections
    │   └── pre_tags_cleaned.json   # Cleaned detections
    ├── samples/                     # Annotated sample images
    │   ├── sample_00001.jpg
    │   ├── sample_00002.jpg
    │   └── SAMPLE_CATALOG.md
    ├── presentation_slides/         # Generated presentation slides
    ├── report.json                  # Machine-readable report
    ├── report.md                    # Human-readable report
    ├── report.png                   # Visualization charts
    └── protex_presentation.mp4      # Presentation video
    └── annotated_video.mp4          # Annotated video output
```
---

## Typical Processing Times

**Single video (24 hours timelapse, ~1GB):**

| Stage | Time (CPU) | Time (GPU) |
|-------|-----------|-----------|
| Stage 1: Preprocessing | 2-3 min | 2-3 min |
| Stage 2: Pre-tagging | 15-20 min | 3-5 min |
| Stage 3: Cleanup | 5-10 sec | 5-10 sec |
| Stage 4: Samples | 10-15 sec | 10-15 sec |
| Stage 5: Report | 5-10 sec | 5-10 sec |
| **Total** | **~20 min** | **~5 min** |

*Times vary based on hardware and video characteristics*

---

## Configuration

Edit `config.yaml` to customize pipeline behavior:

```yaml
defaults:
  mode: "balanced"  # Change to fast or accurate

preprocessing:
  target_width: 960
  target_height: 544
  modes:
    balanced:
      desired_fps: 1.0  # Adjust frame sampling rate
```

### Auto-Calibrate Thresholds (Optional)

**Option 1: Enable in config (automatic)**
```yaml
adaptive_thresholds:
  enabled: true
  sample_size: 100
  brightness_percentile: 10
  blur_percentile: 15
```

**Option 2: Manual calibration**
```bash
python utils/calibrate_thresholds.py --video data/video.mp4 --sample_size 100
```

See [PARAMETERS.md](PARAMETERS.md) and [CALIBRATION_GUIDE.md](CALIBRATION_GUIDE.md) for details.

---

## Operational Modes

| Mode | Use Case | FPS | Confidence | Area Threshold |
|------|----------|-----|------------|----------------|
| **fast** | Rapid iteration, high-traffic sites | 0.5 | 0.6 | 2000 |
| **balanced** | Production annotation pipelines | 1.0 | 0.5 | 1000 |
| **accurate** | Critical areas, rare events | 3.0 | 0.4 | 500 |

---

## Video Masking (Optional)

### Quick Reference

**Create mask (auto-detect black regions)**:
```bash
python utils/create_mask.py --video data/video.mp4 --output masks/mask.png --auto
```

**Create mask (interactive)**:
```bash
python utils/create_mask.py --video data/video.mp4 --output masks/mask.png
```

**Run with mask**:
```bash
python 01_data_preprocessing.py \
  --video_path data/video.mp4 \
  --mask_path masks/mask.png \
  --mode balanced
```

**Or configure in config.yaml**:
```yaml
preprocessing:
  mask_path: "masks/polygon_mask.png"
```

**Key parameters**:
- `--auto`: Auto-detect black regions
- `--threshold`: Brightness threshold (0-255, default: 10)
- `--frame_idx`: Sample frame index (default: 100)

**Mask format**: Grayscale PNG/JPG (white=keep, black=remove)

See [MASKING_GUIDE.md](MASKING_GUIDE.md) for detailed guide.

---

## Troubleshooting

### "Video file not found"
- Check video path is correct
- Use absolute path if relative path fails
- Ensure video format is supported (mp4, avi, mov)

### "CUDA out of memory"
- Reduce `batch_size` in config.yaml (try 2 or 1)
- Use CPU mode (slower but works without GPU)
- Process fewer frames with `--max_images` flag

### "No module named 'torch'"
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`
- Check Python version is 3.8+

### "Permission denied" (Linux/macOS)
- Make script executable: `chmod +x runnable/run_pipeline.sh`
- Or run with: `bash runnable/run_pipeline.sh`

### Slow processing
- Use GPU if available (10x faster for Stage 2)
- Reduce FPS in config (fewer frames to process)
- Use `fast` mode for quick iteration

### Check pipeline logs
- View last run logs: `cat runnable/run_pipeline.log`
- Monitor running pipeline: `tail -f runnable/run_pipeline.log`

---

## Next Steps

1. **Review outputs**: Check `report.md` for dataset summary
2. **Inspect samples**: Look at `traceables/samples/` for quality assessment
3. **Adjust parameters**: Edit `config.yaml` based on results
4. **Scale up**: See [SCALABILITY.md](SCALABILITY.md) for production deployment

---

## Support

- **Documentation**: See README.md, PARAMETERS.md, SCALABILITY.md
- **Annotator guidance**: See ANNOTATOR_GUIDE.md
- **Calibration guide**: See CALIBRATION_GUIDE.md
- **Masking guide**: See MASKING_GUIDE.md

All documentation files can be found in the `docs/` directory.

---

**Ready to process industrial safety footage at scale!** 🚀
