# Static Mask Guide - Removing Black Polygons

This guide explains how to use static masking to remove black polygons or unwanted regions from video frames during preprocessing.

## Overview

The preprocessing pipeline now supports **static masking** to remove black polygons, camera overlays, or other unwanted regions from frames before detection.

**Key benefits:**
- Removes distracting regions (timestamps, logos, black borders)
- Reduces false positives in masked areas
- Improves detection focus on relevant regions
- Applied efficiently via bitwise AND operation

---

## Example: Before and After

### Before Masking
```
┌─────────────────────────────────────┐
│  ╱╲                                 │  ← Black polygon (camera artifact)
│ ╱  ╲    [Person detected]           │
│╱    ╲                               │
│      ╲   [Car detected]             │
│       ╲                             │
│        ╲  [False detection in       │  ← Unwanted detection
│         ╲  black region]            │     in masked area
│          ╲                          │
└─────────────────────────────────────┘
```

### After Masking
```
┌─────────────────────────────────────┐
│  ███                                │  ← Masked region (black)
│ ████    [Person detected]           │
│█████                                │
│      █   [Car detected]             │
│       █                             │
│        █                            │  ← No false detections
│         █                           │     in masked area
│          █                          │
└─────────────────────────────────────┘
```

**Result**: Clean frames with black polygons removed, reducing false positives and improving annotation quality.

**Visual workflow**:
1. **Original frame**: Contains black polygon artifact
2. **Generated mask**: White (keep) and black (remove) regions
3. **Masked frame**: Black polygon replaced with black pixels
4. **Detection output**: Only relevant objects detected

---

## Quick Start

### Step 1: Create a Mask

**Option A: Automatic (recommended for black regions)**
```bash
python utils/create_mask.py \
  --video data/timelapse_test.mp4 \
  --output masks/polygon_mask.png \
  --auto \
  --threshold 10
```

**Option B: Interactive (for custom polygons)**
```bash
python utils/create_mask.py \
  --video data/timelapse_test.mp4 \
  --output masks/polygon_mask.png
```

Interactive controls:
- **Left-click**: Add polygon vertex
- **Right-click**: Remove last vertex
- **Press 'q'**: Finish and save
- **Press 'r'**: Reset polygon

### Step 2: Configure Pipeline

Add mask path to `config.yaml`:

```yaml
preprocessing:
  video_path: "${data_dir}/timelapse_test.mp4"
  output_dir: "${output_root}/frames"
  mask_path: "masks/polygon_mask.png"  # Add this line
  target_width: 960
  target_height: 544
```

### Step 3: Run Preprocessing

```bash
python 01_data_preprocessing.py --video_path data/timelapse_test.mp4 --mode balanced
```

Or with command-line override:
```bash
python 01_data_preprocessing.py \
  --video_path data/timelapse_test.mp4 \
  --mask_path masks/polygon_mask.png \
  --mode balanced
```

---

## How It Works

### Mask Format

Masks are **grayscale images** (PNG or JPG):
- **White pixels (255)**: Keep these regions
- **Black pixels (0)**: Remove these regions (set to black)

**Example mask structure**:
```
┌─────────────────────────┐
│ 255 255 255 255 255 255 │  ← Keep (white)
│ 255 255 255 255 255 255 │
│ 255   0   0   0   255   │  ← Remove center (black polygon)
│ 255   0   0   0   255   │
│ 255 255 255 255 255 255 │
└─────────────────────────┘
```

**Real-world example**:
```
Original Frame          Mask                Masked Result
┌──────────┐           ┌──────────┐         ┌──────────┐
│ ╱╲       │           │ ██       │         │ ██       │
│╱  ╲ 🚶   │    AND    │███ ░░░░░ │    =    │███ 🚶   │
│    ╲     │           │  █ ░░░░░ │         │  █       │
│     ╲ 🚗 │           │   █░░░░░ │         │   █ 🚗   │
└──────────┘           └──────────┘         └──────────┘
  (Black polygon)       (Black=remove)       (Clean frame)
```

### Processing Pipeline

```python
# 1. Load video frame
frame = cv2.VideoCapture(video_path).read()

# 2. Resize to target dimensions (960×544)
resized = cv2.resize(frame, (960, 544))

# 3. Apply static mask (bitwise AND)
masked = cv2.bitwise_and(resized, resized, mask=static_mask)

# 4. Save masked frame
cv2.imwrite("frame_00000.jpg", masked)
```

The mask is loaded **once** at the start and reused for all frames (efficient).

---

## Use Cases

### 1. Remove Black Polygons (Camera Artifacts)

Some cameras add black polygons to mask privacy zones or crop regions.

```bash
# Auto-detect black regions
python utils/create_mask.py \
  --video data/camera_feed.mp4 \
  --output masks/black_polygon.png \
  --auto \
  --threshold 10
```

### 2. Remove Timestamp Overlays

```bash
# Interactive: click around timestamp region
python utils/create_mask.py \
  --video data/cctv_feed.mp4 \
  --output masks/no_timestamp.png
```

### 3. Focus on Specific Region (ROI)

```bash
# Interactive: define region of interest
python utils/create_mask.py \
  --video data/wide_angle.mp4 \
  --output masks/roi_only.png
```

### 4. Per-Camera Masks

For multi-camera deployments, create camera-specific masks:

```
masks/
├── camera_001_mask.png
├── camera_002_mask.png
└── camera_003_mask.png
```

Configure in `config.yaml`:
```yaml
preprocessing:
  mask_path: "masks/camera_mask.png"
```

---

## Advanced Options

### Custom Threshold (Auto Mode)

Adjust threshold based on lighting conditions:

```bash
# Low threshold (5): Very dark regions only
python utils/create_mask.py --video data/video.mp4 --output mask.png --auto --threshold 5

# Medium threshold (10): Default, good for most cases
python utils/create_mask.py --video data/video.mp4 --output mask.png --auto --threshold 10

# High threshold (30): Remove darker regions
python utils/create_mask.py --video data/video.mp4 --output mask.png --auto --threshold 30
```

### Sample Different Frame

If frame 100 doesn't show the polygon clearly:

```bash
python utils/create_mask.py \
  --video data/video.mp4 \
  --output mask.png \
  --frame_idx 500  # Sample frame 500 instead
```

### Custom Resolution

Match your video's native resolution:

```bash
python utils/create_mask.py \
  --video data/video.mp4 \
  --output mask.png \
  --width 1920 \
  --height 1080
```

---

## Validation

### Verify Mask Quality

After creating a mask, check the preview window:
- **Left panel**: Original frame
- **Middle panel**: Mask (white=keep, black=remove)
- **Right panel**: Masked result

Ensure:
- ✅ Black polygons are fully covered by black mask regions
- ✅ Important areas (people, equipment) are white in mask
- ✅ No unintended masking of relevant regions

### Test on Sample Frames

Run preprocessing on a short clip:

```bash
python 01_data_preprocessing.py \
  --video_path data/test_clip.mp4 \
  --mask_path masks/polygon_mask.png \
  --mode balanced
```

Inspect output frames in `traceables/frames/` to verify masking works correctly.

---

## Performance Impact

**Computational cost**: Negligible
- Mask loaded once (not per-frame)
- Bitwise AND is extremely fast (~0.1ms per frame)
- No impact on overall pipeline speed

**Storage**: No change
- Masked regions saved as black pixels
- JPEG compression handles black regions efficiently

---

## Troubleshooting

### Issue: Mask doesn't align with video

**Cause**: Resolution mismatch

**Solution**: Ensure mask resolution matches target dimensions:
```bash
python utils/create_mask.py \
  --video data/video.mp4 \
  --output mask.png \
  --width 960 \
  --height 544  # Match config.yaml
```

### Issue: Mask removes too much

**Cause**: Threshold too high (auto mode) or polygon too large (interactive)

**Solution**: 
- Auto mode: Lower threshold (`--threshold 5`)
- Interactive: Redraw polygon more precisely

### Issue: Mask doesn't remove polygon completely

**Cause**: Threshold too low or polygon not fully covered

**Solution**:
- Auto mode: Increase threshold (`--threshold 15`)
- Interactive: Ensure polygon fully covers black region

---

## Integration with Pipeline

### Full Pipeline with Masking

```bash
# 1. Create mask
python utils/create_mask.py \
  --video data/timelapse_test.mp4 \
  --output masks/polygon_mask.png \
  --auto

# 2. Run full pipeline
./runnable/run_pipeline.sh balanced data/timelapse_test.mp4

# Or with explicit mask
python 01_data_preprocessing.py \
  --video_path data/timelapse_test.mp4 \
  --mask_path masks/polygon_mask.png \
  --mode balanced

python 02_data_pretagging.py --mode balanced
python 03_pretag_cleanup.py --mode balanced
python 04_generate_samples.py --num_samples 20
python 05_generate_report.py
```

### Disable Masking

To disable masking temporarily:

```bash
# Option 1: Remove from config
# Comment out mask_path in config.yaml

# Option 2: Override with empty value
python 01_data_preprocessing.py \
  --video_path data/video.mp4 \
  --mask_path ""
```

---

## Best Practices

1. **Create masks on representative frames**: Choose a frame that clearly shows the polygon/region to mask

2. **Use auto mode for black regions**: Faster and more consistent than manual polygon drawing

3. **Use interactive mode for complex shapes**: Better control for irregular regions or multiple polygons

4. **Version control masks**: Store masks in `masks/` directory and commit to git

5. **Document mask purpose**: Name masks descriptively:
   - `camera_001_black_polygon.png`
   - `timestamp_overlay_mask.png`
   - `roi_construction_zone.png`

6. **Test before production**: Always validate masks on sample videos before deploying to production

---

## Example Workflow

```bash
# 1. Inspect video to identify black polygon
ffplay data/timelapse_test.mp4

# 2. Create mask automatically
python utils/create_mask.py \
  --video data/timelapse_test.mp4 \
  --output masks/polygon_mask.png \
  --auto \
  --threshold 10

# 3. Preview mask (window shows original | mask | result)
# Press any key to close preview

# 4. Update config.yaml
echo "  mask_path: masks/polygon_mask.png" >> config.yaml

# 5. Run preprocessing
python 01_data_preprocessing.py --mode balanced

# 6. Verify output frames
ls traceables/frames/
# Check that black polygons are removed
```

---
