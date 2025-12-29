# Protex AI - Dataset Report

## Pipeline Summary

### 1. Preprocessing
- **Input**: Raw video
- **Output**: Filtered frames
- **Filters**:
  - FPS sampling: 0.8
  - Min brightness: 20.0
  - Min sharpness: 30.0
  - Perceptual hash deduplication
- **Statistics**: 27/4077 frames (99.3% reduction)

### 2. Pre-tagging
- **Input**: Filtered frames
- **Output**: COCO detections
- **Statistics**: 128/1009 detections kept

### 3. Cleanup
- **Input**: Raw detections
- **Output**: Cleaned detections
- **Filters**:
  - Min area: 1000.0 px²
  - Min score: 0.6
  - Person area: 300.0 px²
- **Statistics**: 45/128 kept (35%)

## Dataset Overview

- Total Images: 27
- Total Annotations: 45
- Avg Annotations/Image: 1.67
- High Activity Frames: 0

## Class Distribution

| Class | Count | % |
|-------|-------|---|
| person | 23 | 51.1% |
| car | 16 | 35.6% |
| truck | 3 | 6.7% |
| bus | 3 | 6.7% |

## Known Issues & Edge Cases

The following challenges are present in this dataset:

- **Low-confidence detections**: Some boxes have confidence < 0.6 (require review)
- **Occlusion**: Partial visibility of objects behind others
- **Lighting variations**: Dawn/dusk transitions may affect detection quality
- **Small objects**: Objects < 1000px² may be at detection limits
- **Crowded scenes**: Overlapping objects may have merged bounding boxes

**Recommendation**: Review orange-boxed (low-confidence) detections first. These represent the model's uncertainty and require careful annotation.
