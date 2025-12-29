#!/usr/bin/env python
"""
Visualize 10 consecutive frames at 3 points (beginning, middle, end) during video processing.
Shows quality filtering in action.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from utils.config_loader import get_config, get_mode_config


def get_mode_defaults(mode: str):
    mode = mode.lower()
    min_brightness = get_mode_config("preprocessing", mode, "min_brightness", 30.0)
    min_laplacian_var = get_mode_config("preprocessing", mode, "min_laplacian_var", 50.0)
    return min_brightness, min_laplacian_var


def is_quality_acceptable(frame, min_brightness, min_laplacian_var):
    small = cv2.resize(frame, (320, 180))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    
    brightness = float(np.mean(gray))
    if brightness < min_brightness:
        return False, "dark"
    
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < min_laplacian_var:
        return False, "blurry"
    
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float((edges > 0).mean())
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    hist = hist / (hist.sum() + 1e-8)
    entropy = float(-(hist * np.log2(hist + 1e-8)).sum())
    
    is_noise = lap_var > min_laplacian_var * 3.0 and edge_density > 0.20 and entropy > 4.0
    if is_noise:
        return False, "noise"
    
    return True, None


def create_frame_grid(frames, labels, thumb_w=160, thumb_h=90):
    """Create 2x5 grid of frames with labels."""
    grid = []
    for i in range(2):
        row = []
        for j in range(5):
            idx = i * 5 + j
            if idx < len(frames):
                thumb = cv2.resize(frames[idx], (thumb_w, thumb_h))
                # Add label
                label = labels[idx]
                color = (0, 255, 0) if label == "KEEP" else (0, 0, 255)
                cv2.putText(thumb, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            else:
                thumb = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
            row.append(thumb)
        grid.append(np.hstack(row))
    return np.vstack(grid)


def visualize_processing(video_path, mode, output_path):
    min_brightness, min_laplacian_var = get_mode_defaults(mode)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define 3 buffer positions: beginning, middle, end
    positions = [
        ("BEGINNING", 0),
        ("MIDDLE", total_frames // 2 - 5),
        ("END", max(0, total_frames - 10))
    ]
    
    all_grids = []
    
    for title, start_frame in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        labels = []
        
        for i in range(10):
            ret, frame = cap.read()
            if not ret:
                break
            
            quality_ok, reason = is_quality_acceptable(frame, min_brightness, min_laplacian_var)
            frames.append(frame)
            labels.append("KEEP" if quality_ok else reason.upper())
        
        if frames:
            grid = create_frame_grid(frames, labels)
            # Add title
            title_img = np.zeros((30, grid.shape[1], 3), dtype=np.uint8)
            cv2.putText(title_img, f"{title} (frames {start_frame}-{start_frame+9})", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            all_grids.append(np.vstack([title_img, grid]))
    
    cap.release()
    
    # Stack all 3 grids vertically
    final = np.vstack(all_grids)
    cv2.imwrite(output_path, final)
    print(f"✅ Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default=get_config("preprocessing.video_path", "data/timelapse_test.mp4"))
    parser.add_argument("--mode", type=str, default=get_config("defaults.mode", "balanced"))
    parser.add_argument("--output", type=str, default="traceables/processing_visualization.jpg")
    args, _ = parser.parse_known_args()
    
    Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    visualize_processing(args.video_path, args.mode, args.output)


if __name__ == "__main__":
    main()
