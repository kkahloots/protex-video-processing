#!/usr/bin/env python
"""
Generate Stage 3 cleanup comparison: raw vs cleaned boxes.
Shows class-aware filtering (people vs vehicles).
"""

import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def load_annotations():
    """Load raw and cleaned COCO annotations."""
    raw_path = Path("traceables/pre_tags/pre_tags_raw.json")
    cleaned_path = Path("traceables/pre_tags/pre_tags_cleaned.json")
    
    with open(raw_path) as f:
        raw = json.load(f)
    with open(cleaned_path) as f:
        cleaned = json.load(f)
    
    return raw, cleaned


def draw_boxes(img, annotations, categories, color_map):
    """Draw bounding boxes on image."""
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        cat_name = categories[ann["category_id"]]
        conf = ann.get("score", 1.0)
        
        # Color based on confidence
        if conf < 0.6:
            color = "orange"
        else:
            color = color_map.get(cat_name, "green")
        
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        
        # Label
        label = f"{cat_name} {conf:.2f}"
        draw.text((x, y - 20), label, fill=color, font=font)
    
    return img


def create_comparison():
    """Create side-by-side comparison of raw vs cleaned."""
    
    raw, cleaned = load_annotations()
    
    # Get categories
    categories = {cat["id"]: cat["name"] for cat in raw["categories"]}
    
    # Find image with good example (multiple objects, some filtered)
    raw_anns_by_img = {}
    for ann in raw["annotations"]:
        raw_anns_by_img.setdefault(ann["image_id"], []).append(ann)
    
    cleaned_anns_by_img = {}
    for ann in cleaned["annotations"]:
        cleaned_anns_by_img.setdefault(ann["image_id"], []).append(ann)
    
    # Find image with filtering (raw > cleaned)
    best_img_id = None
    max_diff = 0
    for img_id in raw_anns_by_img:
        diff = len(raw_anns_by_img[img_id]) - len(cleaned_anns_by_img.get(img_id, []))
        if diff > max_diff:
            max_diff = diff
            best_img_id = img_id
    
    if best_img_id is None:
        best_img_id = list(raw_anns_by_img.keys())[0]
    
    # Get image info
    img_info = next(img for img in raw["images"] if img["id"] == best_img_id)
    img_path = Path("traceables/frames") / img_info["file_name"]
    
    # Load image
    frame = cv2.imread(str(img_path))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create two copies
    raw_img = Image.fromarray(frame_rgb.copy())
    cleaned_img = Image.fromarray(frame_rgb.copy())
    
    # Color map
    color_map = {"person": "red", "car": "blue", "truck": "cyan", "bus": "magenta"}
    
    # Draw boxes
    raw_anns = raw_anns_by_img.get(best_img_id, [])
    cleaned_anns = cleaned_anns_by_img.get(best_img_id, [])
    
    raw_img = draw_boxes(raw_img, raw_anns, categories, color_map)
    cleaned_img = draw_boxes(cleaned_img, cleaned_anns, categories, color_map)
    
    # Create side-by-side
    width, height = raw_img.size
    comparison = Image.new("RGB", (width * 2 + 100, height + 200), (20, 20, 40))
    
    # Paste images
    comparison.paste(raw_img, (50, 150))
    comparison.paste(cleaned_img, (width + 50, 150))
    
    # Add labels
    draw = ImageDraw.Draw(comparison)
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        title_font = label_font = text_font = ImageFont.load_default()
    
    # Title
    draw.text((50, 30), "Stage 3: Cleanup - Class-Aware Filtering", fill=(0, 200, 255), font=title_font)
    
    # Labels
    draw.text((width // 2 - 100, 100), f"RAW ({len(raw_anns)} boxes)", fill=(255, 100, 100), font=label_font)
    draw.text((width + width // 2 - 100, 100), f"CLEANED ({len(cleaned_anns)} boxes)", fill=(100, 255, 100), font=label_font)
    
    # Legend
    legend_y = height + 160
    draw.text((50, legend_y), "Red=Person (300px² min) | Blue=Car (1000px² min) | Orange=Low confidence (<0.6)", 
              fill=(200, 200, 200), font=text_font)
    
    # Save
    output_path = Path("docs/imgs/stage3_comparison.png")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    comparison.save(output_path)
    
    print(f"✅ Saved: {output_path}")
    print(f"   Raw boxes: {len(raw_anns)}")
    print(f"   Cleaned boxes: {len(cleaned_anns)}")
    print(f"   Filtered: {len(raw_anns) - len(cleaned_anns)} boxes removed")
    
    return output_path


if __name__ == "__main__":
    print("=" * 80)
    print(" Stage 3 Cleanup Comparison Generator")
    print("=" * 80)
    print()
    
    output = create_comparison()
    
    print()
    print("=" * 80)
