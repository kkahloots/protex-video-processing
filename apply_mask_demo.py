#!/usr/bin/env python
"""
Apply masking and resizing to a single frame.
Demonstrates removal of invalid camera regions (black polygons, occlusions).
"""

import cv2
import numpy as np
from pathlib import Path


def create_demo_mask(width, height):
    """Create a demo mask with black polygons on left and right sides."""
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # Left polygon (simulating camera occlusion)
    left_polygon = np.array([
        [0, 0],
        [width // 3, 0],
        [width // 4, height // 2],
        [width // 5, height],
        [0, height]
    ], dtype=np.int32)
    
    # Right polygon (simulating camera occlusion)
    right_polygon = np.array([
        [width, 0],
        [width * 2 // 3, 0],
        [width * 3 // 4, height // 2],
        [width * 4 // 5, height],
        [width, height]
    ], dtype=np.int32)
    
    cv2.fillPoly(mask, [left_polygon, right_polygon], 0)
    return mask


def apply_mask_and_resize(input_path, output_path, target_width=960, target_height=544):
    """Apply mask and resize to frame."""
    
    # Read frame
    frame = cv2.imread(input_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read: {input_path}")
    
    orig_h, orig_w = frame.shape[:2]
    print(f"Original size: {orig_w}x{orig_h}")
    
    # Resize first
    resized = cv2.resize(frame, (target_width, target_height))
    print(f"Resized to: {target_width}x{target_height}")
    
    # Create mask
    mask = create_demo_mask(target_width, target_height)
    
    # Apply mask (masked regions become black)
    masked = cv2.bitwise_and(resized, resized, mask=mask)
    
    # Save results
    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save masked result
    cv2.imwrite(str(output_path), masked)
    print(f"✅ Saved masked: {output_path}")
    
    # Save mask visualization
    mask_vis_path = output_dir / f"{output_path.stem}_mask.jpg"
    cv2.imwrite(str(mask_vis_path), mask)
    print(f"✅ Saved mask: {mask_vis_path}")
    
    # Save original resized for comparison
    orig_path = output_dir / f"{output_path.stem}_original.jpg"
    cv2.imwrite(str(orig_path), resized)
    print(f"✅ Saved original: {orig_path}")
    
    return masked, mask, resized


if __name__ == "__main__":
    input_path = "traceables/frames/frame_00022.jpg"
    output_path = "traceables/frames/frame_00022_masked.jpg"
    
    print("=" * 80)
    print(" Masking & Resizing Demo")
    print("=" * 80)
    print()
    
    if not Path(input_path).exists():
        print(f"❌ Input file not found: {input_path}")
        print("Run the pipeline first: ./runnable/run_pipeline.sh balanced data/timelapse_test.mp4")
        exit(1)
    
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print()
    
    masked, mask, original = apply_mask_and_resize(input_path, output_path)
    
    print()
    print("=" * 80)
    print("Results:")
    print("  • Original (resized): frame_00022_masked_original.jpg")
    print("  • Mask: frame_00022_masked_mask.jpg (white=keep, black=remove)")
    print("  • Masked Result: frame_00022_masked.jpg (polygon removed)")
    print("=" * 80)
