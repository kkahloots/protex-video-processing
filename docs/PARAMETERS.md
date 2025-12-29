# Protex AI - Parameter Control Panel

## Business Trade-offs Guide

This document explains how each parameter affects **annotation cost**, **temporal coverage**, and **detection quality**.

---

## 🎯 Core Parameters

### `MODE` (fast / balanced / accurate)
**Business Impact**: Controls the overall speed vs quality trade-off across the entire pipeline.

- **fast**: Minimal dataset size, lowest annotation cost, good for rapid iteration
- **balanced**: Good default for production annotation pipelines
- **accurate**: Maximum coverage and sensitivity, higher cost but better for critical areas

---

## 📹 Stage 1: Preprocessing (Video → Frames)

### `DESIRED_FPS`
**What it controls**: How many frames per second we extract from video.

**Business trade-off**:
- **Higher FPS** (e.g., 3.0):
  - ✅ Better temporal coverage (catch rare events)
  - ❌ More frames = higher annotation cost
  - ❌ More storage and processing time
  
- **Lower FPS** (e.g., 0.5):
  - ✅ Lower annotation cost
  - ✅ Faster processing
  - ❌ May miss brief events

**Manager guidance**: 
- Busy sites with frequent activity → 0.5-1.0 FPS
- Critical areas or rare event detection → 2.0-3.0 FPS

### `DUPLICATE_THRESHOLD`
**What it controls**: Filters out static/duplicate frames (no motion detected).

**Business impact**: Reduces annotation workload by 30-60% in typical timelapse footage without losing signal.

### `MIN_BRIGHTNESS` & `MIN_LAPLACIAN_VAR`
**What it controls**: Quality filters for frame brightness and sharpness.

**Business trade-off**:
- **Higher thresholds**: Only keep well-lit, sharp frames (better annotation quality)
- **Lower thresholds**: Keep more frames including marginal quality (better coverage)

**Manager guidance**: 
- Night-shift monitoring → lower brightness threshold (25-30)
- Daytime-only sites → higher brightness threshold (35+)
- Blurry cameras → increase laplacian threshold to force sharper frames

---

## 🤖 Stage 2: Pre-tagging (Detection)

### `MIN_CONFIDENCE`
**What it controls**: Minimum model confidence to keep a detection.

**Business trade-off**:
- **Higher threshold** (e.g., 0.6):
  - ✅ Fewer false positives (less annotator correction)
  - ❌ May miss real objects (lower recall)
  
- **Lower threshold** (e.g., 0.4):
  - ✅ Catch more objects (higher recall)
  - ❌ More false positives (more annotator work)

**Manager guidance**: Start at 0.5, adjust based on annotator feedback.

---

## 🧹 Stage 3: Cleanup

### `MIN_AREA` & `MIN_AREA_PERSON`
**What it controls**: Minimum bounding box size (pixels²) to keep.

**Business trade-off**:
- **Higher threshold** (e.g., 2000):
  - ✅ Removes tiny noise detections
  - ❌ May filter out distant but important objects
  
- **Lower threshold** (e.g., 500):
  - ✅ Keeps small objects
  - ❌ More noise for annotators to review

**Class-aware filtering**: People get lower area threshold (300 vs 1000) because any human presence near equipment is safety-critical.

**Manager guidance**: 
- 1000 is a good default for equipment at 960×544 resolution
- People: 200-500 depending on camera distance
- Adjust based on camera distance and object sizes

---

## 📊 Stage 4: Samples

### `NUM_SAMPLES`
**What it controls**: How many annotated sample images to generate for QA.

**Manager guidance**: 20-50 samples gives good coverage for quality checks without overwhelming reviewers.

---

## 💡 Quick Reference Table

| Scenario | MODE | FPS | MIN_CONFIDENCE | MIN_AREA | MIN_AREA_PERSON | MIN_BRIGHTNESS |
|----------|------|-----|----------------|----------|------------------|----------------|
| Pilot / Quick iteration | fast | 0.5 | 0.6 | 2000 | 500 | 35.0 |
| Production annotation | balanced | 1.0 | 0.5 | 1000 | 300 | 30.0 |
| Critical area / rare events | accurate | 3.0 | 0.4 | 500 | 200 | 25.0 |
| High-traffic area | fast | 0.5 | 0.6 | 2000 | 500 | 35.0 |
| Incident review | accurate | 3.0 | 0.4 | 500 | 200 | 25.0 |

**Note**: See docs/FEATURES.md for details on unified brightness filtering (removed redundant BLACK_THRESHOLD parameter).

---

