# Pipeline Steps Detail

## 1. Preprocessing

**Transformation**: Raw video → Filtered frames

**Applied Filters**:
- FPS sampling: 0.8
- Min brightness: 20.0
- Min sharpness: 30.0
- Perceptual hash deduplication

**Statistics**:
- total_video_frames: 4077
- saved_frames: 27
- reduction_pct: 99.3

## 2. Pre-tagging

**Transformation**: Filtered frames → COCO detections

**Statistics**:
- raw_detections: 1009
- filtered_low_conf: 881
- kept_detections: 128

## 3. Cleanup

**Transformation**: Raw detections → Cleaned detections

**Applied Filters**:
- Min area: 1000.0 px²
- Min score: 0.6
- Person area: 300.0 px²

**Statistics**:
- original: 128
- removed_small: 83
- removed_score: 0
- removed_masked: 0
- kept: 45

