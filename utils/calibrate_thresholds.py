#!/usr/bin/env python3
"""
Auto-calibrate brightness and blur thresholds based on video analysis.

Usage:
    python utils/calibrate_thresholds.py --video data/video.mp4 --sample_size 100
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


def compute_frame_metrics(frame):
    """Compute brightness and Laplacian variance for a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return brightness, laplacian_var


def sample_video_metrics(video_path, sample_size=100, sample_strategy='uniform'):
    """
    Sample frames from video and compute quality metrics.
    
    Args:
        video_path: Path to video file
        sample_size: Number of frames to sample
        sample_strategy: 'uniform' or 'random'
    
    Returns:
        brightness_values: List of brightness values
        laplacian_values: List of Laplacian variance values
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if sample_size >= total_frames:
        frame_indices = list(range(total_frames))
    elif sample_strategy == 'uniform':
        step = total_frames // sample_size
        frame_indices = list(range(0, total_frames, step))[:sample_size]
    else:  # random
        frame_indices = sorted(np.random.choice(total_frames, sample_size, replace=False))
    
    brightness_values = []
    laplacian_values = []
    
    print(f"Sampling {len(frame_indices)} frames from {total_frames} total frames...")
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        brightness, laplacian_var = compute_frame_metrics(frame)
        brightness_values.append(brightness)
        laplacian_values.append(laplacian_var)
    
    cap.release()
    return brightness_values, laplacian_values


def compute_thresholds(brightness_values, laplacian_values, 
                    brightness_percentile=10, blur_percentile=15):
    """
    Compute thresholds based on percentiles.
    
    Args:
        brightness_values: List of brightness values
        laplacian_values: List of Laplacian variance values
        brightness_percentile: Percentile for min_brightness (lower = stricter)
        blur_percentile: Percentile for min_laplacian_var (lower = stricter)
    
    Returns:
        dict with recommended thresholds
    """
    min_brightness = np.percentile(brightness_values, brightness_percentile)
    min_laplacian_var = np.percentile(laplacian_values, blur_percentile)
    
    # Compute statistics
    brightness_stats = {
        'mean': np.mean(brightness_values),
        'std': np.std(brightness_values),
        'min': np.min(brightness_values),
        'max': np.max(brightness_values),
        'p10': np.percentile(brightness_values, 10),
        'p25': np.percentile(brightness_values, 25),
        'p50': np.percentile(brightness_values, 50),
    }
    
    laplacian_stats = {
        'mean': np.mean(laplacian_values),
        'std': np.std(laplacian_values),
        'min': np.min(laplacian_values),
        'max': np.max(laplacian_values),
        'p10': np.percentile(laplacian_values, 10),
        'p25': np.percentile(laplacian_values, 25),
        'p50': np.percentile(laplacian_values, 50),
    }
    
    return {
        'min_brightness': float(min_brightness),
        'min_laplacian_var': float(min_laplacian_var),
        'brightness_stats': brightness_stats,
        'laplacian_stats': laplacian_stats,
    }


def main():
    parser = argparse.ArgumentParser(description='Auto-calibrate quality thresholds')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--sample_size', type=int, default=100, 
                        help='Number of frames to sample (default: 100)')
    parser.add_argument('--brightness_percentile', type=float, default=10,
                        help='Percentile for brightness threshold (default: 10)')
    parser.add_argument('--blur_percentile', type=float, default=15,
                        help='Percentile for blur threshold (default: 15)')
    parser.add_argument('--strategy', choices=['uniform', 'random'], default='uniform',
                        help='Sampling strategy (default: uniform)')
    parser.add_argument('--output', help='Output JSON file (optional)')
    
    args = parser.parse_args()
    
    # Sample video
    brightness_values, laplacian_values = sample_video_metrics(
        args.video, args.sample_size, args.strategy
    )
    
    # Compute thresholds
    results = compute_thresholds(
        brightness_values, laplacian_values,
        args.brightness_percentile, args.blur_percentile
    )
    
    # Print results
    print("\n" + "="*60)
    print("CALIBRATION RESULTS")
    print("="*60)
    print(f"\nRecommended Thresholds:")
    print(f"  min_brightness: {results['min_brightness']:.1f}")
    print(f"  min_laplacian_var: {results['min_laplacian_var']:.1f}")
    
    print(f"\nBrightness Statistics:")
    for key, val in results['brightness_stats'].items():
        print(f"  {key}: {val:.1f}")
    
    print(f"\nLaplacian Variance Statistics:")
    for key, val in results['laplacian_stats'].items():
        print(f"  {key}: {val:.1f}")
    
    print("\n" + "="*60)
    print("USAGE:")
    print("="*60)
    print("\nAdd to config.yaml:")
    print(f"""
preprocessing:
  modes:
    balanced:
      min_brightness: {results['min_brightness']:.1f}
      min_laplacian_var: {results['min_laplacian_var']:.1f}
""")
    
    # Save to file if requested
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
