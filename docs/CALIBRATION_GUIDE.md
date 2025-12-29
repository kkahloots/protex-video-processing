# Threshold Calibration Guide

## Overview

Blur and brightness thresholds are **domain-dependent** and vary based on:
- Camera quality (HD vs SD)
- Lighting conditions (indoor vs outdoor, day vs night)
- Video compression artifacts
- Environmental factors (weather, seasons)

Instead of using fixed thresholds, **calibrate them automatically** by analyzing your video's quality distribution.

---

## Quick Start

### Basic Usage

```bash
python utils/calibrate_thresholds.py --video data/timelapse_test.mp4
```

**Output**:
```
Sampling 100 frames from 86400 total frames...

============================================================
CALIBRATION RESULTS
============================================================

Recommended Thresholds:
  min_brightness: 28.3
  min_laplacian_var: 45.7

Brightness Statistics:
  mean: 85.2
  std: 32.1
  min: 5.2
  max: 245.8
  p10: 28.3
  p25: 62.1
  p50: 82.5

Laplacian Variance Statistics:
  mean: 125.4
  std: 78.3
  min: 8.2
  max: 456.7
  p10: 45.7
  p25: 68.9
  p50: 112.3

============================================================
USAGE:
============================================================

Add to config.yaml:

preprocessing:
  modes:
    balanced:
      min_brightness: 28.3
      min_laplacian_var: 45.7
```

---

## How It Works

### 1. Frame Sampling

Samples frames uniformly across the video:
- Default: 100 frames
- Strategy: Uniform (evenly spaced) or random
- Covers entire video duration (day/night cycles)

### 2. Quality Metrics

For each sampled frame:
- **Brightness**: Mean pixel intensity (0-255)
  ```python
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  brightness = np.mean(gray)
  ```
- **Laplacian Variance**: Blur detection
  ```python
  laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
  ```

### 3. Percentile-Based Thresholds

- **10th percentile brightness**: Filters darkest 10% of frames
- **15th percentile Laplacian**: Filters blurriest 15% of frames
- Robust to outliers (black frames, extreme blur)

---

## Advanced Options

### Custom Percentiles

```bash
# Stricter filtering (remove more frames)
python utils/calibrate_thresholds.py \
  --video data/video.mp4 \
  --brightness_percentile 20 \
  --blur_percentile 25

# Looser filtering (keep more frames)
python utils/calibrate_thresholds.py \
  --video data/video.mp4 \
  --brightness_percentile 5 \
  --blur_percentile 10
```

### Larger Sample Size

```bash
# More accurate calibration (slower)
python utils/calibrate_thresholds.py \
  --video data/video.mp4 \
  --sample_size 500
```

### Random Sampling

```bash
# Random instead of uniform sampling
python utils/calibrate_thresholds.py \
  --video data/video.mp4 \
  --strategy random
```

### Save Results

```bash
# Save to JSON for later use
python utils/calibrate_thresholds.py \
  --video data/video.mp4 \
  --output calibration/camera_001.json
```

---

## Use Cases

### 1. Per-Camera Calibration

Different cameras have different quality characteristics:

```bash
# Calibrate each camera
python utils/calibrate_thresholds.py --video data/camera_001.mp4 --output calibration/camera_001.json
python utils/calibrate_thresholds.py --video data/camera_002.mp4 --output calibration/camera_002.json
python utils/calibrate_thresholds.py --video data/camera_003.mp4 --output calibration/camera_003.json
```

Update `config.yaml` with computed thresholds:
```yaml
preprocessing:
  modes:
    balanced:
      min_brightness: 28.3
      min_laplacian_var: 45.7
```

### 2. Seasonal Adjustments

Lighting changes with seasons:

```bash
# Winter (shorter days, lower light)
python utils/calibrate_thresholds.py --video data/winter_sample.mp4

# Summer (longer days, brighter)
python utils/calibrate_thresholds.py --video data/summer_sample.mp4
```

### 3. Day/Night Handling

For 24/7 monitoring:

```bash
# Analyze full 24-hour cycle
python utils/calibrate_thresholds.py \
  --video data/24h_timelapse.mp4 \
  --sample_size 200 \
  --brightness_percentile 10  # Keeps nighttime frames
```

### 4. Video Quality Tiers

Different quality levels need different thresholds:

```bash
# HD camera (higher quality expectations)
python utils/calibrate_thresholds.py \
  --video data/hd_camera.mp4 \
  --blur_percentile 25  # Stricter blur filtering

# SD camera (lower quality expectations)
python utils/calibrate_thresholds.py \
  --video data/sd_camera.mp4 \
  --blur_percentile 10  # Looser blur filtering
```

---

## Interpreting Results

### Brightness Statistics

- **Low mean (<50)**: Dark environment (indoor, nighttime)
  - Use lower `min_brightness` (20-30)
- **High mean (>100)**: Bright environment (outdoor, daytime)
  - Use higher `min_brightness` (30-40)
- **High std (>40)**: Variable lighting (day/night cycles)
  - Use percentile-based threshold

### Laplacian Variance Statistics

- **Low mean (<100)**: Generally blurry video
  - Use lower `min_laplacian_var` (30-40)
- **High mean (>150)**: Sharp video
  - Use higher `min_laplacian_var` (50-70)
- **High std (>80)**: Variable sharpness
  - Use percentile-based threshold

---

## Best Practices

1. **Sample representative footage**: Use 24-hour timelapse or typical day's footage

2. **Adjust percentiles based on goals**:
   - High recall (keep more frames): Lower percentiles (5-10)
   - High precision (keep only best): Higher percentiles (20-30)

3. **Validate results**: Run preprocessing with calibrated thresholds and inspect output

4. **Re-calibrate periodically**: Lighting/camera conditions change over time

5. **Per-camera calibration**: Different cameras need different thresholds

---

## Integration with Pipeline

### Step 1: Calibrate

```bash
python utils/calibrate_thresholds.py --video data/video.mp4
```

### Step 2: Update Config

Edit `config.yaml`:
```yaml
preprocessing:
  modes:
    balanced:
      min_brightness: 28.3      # From calibration
      min_laplacian_var: 45.7   # From calibration
```

### Step 3: Run Pipeline

```bash
./runnable/run_pipeline.sh balanced data/video.mp4
```

### Step 4: Validate

Check `traceables/frames/` to ensure quality is acceptable.

---

## Troubleshooting

### Too many frames filtered out

**Symptom**: Very few frames extracted

**Solution**: Lower percentiles or check video quality
```bash
python utils/calibrate_thresholds.py \
  --video data/video.mp4 \
  --brightness_percentile 5 \
  --blur_percentile 10
```

### Too many low-quality frames kept

**Symptom**: Many dark/blurry frames in output

**Solution**: Increase percentiles
```bash
python utils/calibrate_thresholds.py \
  --video data/video.mp4 \
  --brightness_percentile 20 \
  --blur_percentile 25
```

### Inconsistent results

**Symptom**: Different traceables give different thresholds

**Solution**: Increase sample size
```bash
python utils/calibrate_thresholds.py \
  --video data/video.mp4 \
  --sample_size 500
```

---

## Technical Details

### Why Percentiles?

Fixed thresholds fail because:
- Camera A (HD): Laplacian variance 50-200 (sharp)
- Camera B (SD): Laplacian variance 20-80 (softer)
- Fixed threshold of 50 would reject all Camera B frames

Percentile-based thresholds adapt:
- Camera A: 15th percentile = 65 (keeps 85% of frames)
- Camera B: 15th percentile = 28 (keeps 85% of frames)

### Computational Cost

- Sample 100 frames: ~5-10 seconds
- Sample 500 frames: ~20-30 seconds
- One-time cost per video/camera

---

