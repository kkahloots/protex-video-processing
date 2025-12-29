#!/usr/bin/env python
"""
Detect imminent collisions in annotated sample images.
Analyzes bounding boxes for proximity and relative positions.
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import shutil


def load_coco_data():
    """Load COCO annotations."""
    coco_path = Path("traceables/pre_tags/pre_tags_cleaned.json")
    with open(coco_path) as f:
        return json.load(f)


def get_box_center(bbox):
    """Get center point of bounding box [x, y, w, h]."""
    return (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)


def get_distance(box1, box2):
    """Calculate distance between box centers."""
    c1 = get_box_center(box1)
    c2 = get_box_center(box2)
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def boxes_overlap(box1, box2, margin=50):
    """Check if boxes overlap or are very close (with margin)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    return not (x1 + w1 + margin < x2 or x2 + w2 + margin < x1 or
                y1 + h1 + margin < y2 or y2 + h2 + margin < y1)


def detect_collisions():
    """Detect potential collisions in sample images."""
    
    coco = load_coco_data()
    images = {img["id"]: img for img in coco["images"]}
    
    # Group annotations by image
    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)
    
    # Category mapping
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    
    # Collision detection
    collisions = []
    
    for img_id, anns in anns_by_img.items():
        if len(anns) < 2:
            continue
        
        img_info = images[img_id]
        
        # Check all pairs
        for i, ann1 in enumerate(anns):
            for ann2 in anns[i + 1:]:
                cat1 = categories[ann1["category_id"]]
                cat2 = categories[ann2["category_id"]]
                
                # Check for collision scenarios
                distance = get_distance(ann1["bbox"], ann2["bbox"])
                overlap = boxes_overlap(ann1["bbox"], ann2["bbox"], margin=30)
                
                # Collision criteria
                reason = None
                
                # Person near vehicle
                if ("person" in cat1 and "car" in cat2) or ("car" in cat1 and "person" in cat2):
                    if distance < 150 or overlap:
                        reason = f"Person near vehicle - distance: {distance:.0f}px"
                
                # Vehicle-vehicle proximity
                elif "car" in cat1 and "car" in cat2:
                    if distance < 100 or overlap:
                        reason = f"Vehicle-vehicle proximity - distance: {distance:.0f}px"
                
                # Person-person crowding
                elif cat1 == "person" and cat2 == "person":
                    if distance < 80 or overlap:
                        reason = f"Person crowding - distance: {distance:.0f}px"
                
                # Bus/truck near person
                elif ("person" in cat1 and ("bus" in cat2 or "truck" in cat2)) or \
                     (("bus" in cat1 or "truck" in cat1) and "person" in cat2):
                    if distance < 200 or overlap:
                        reason = f"Large vehicle near person - distance: {distance:.0f}px"
                
                if reason:
                    collisions.append({
                        "image_id": img_id,
                        "image_file": img_info["file_name"],
                        "objects": [cat1, cat2],
                        "reason": reason,
                        "distance": distance,
                        "boxes": [ann1["bbox"], ann2["bbox"]]
                    })
                    break  # One collision per image is enough
            if reason:
                break
    
    return collisions


def create_collision_image(collision, output_path):
    """Create annotated image highlighting collision."""
    samples_dir = Path("traceables/samples")
    frames_dir = Path("traceables/frames")
    
    # Find corresponding sample image
    img_file = collision["image_file"]
    frame_path = frames_dir / img_file
    
    if not frame_path.exists():
        return False
    
    img = Image.open(frame_path)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Draw collision boxes in red
    for bbox in collision["boxes"]:
        x, y, w, h = bbox
        draw.rectangle([x, y, x + w, y + h], outline="red", width=4)
    
    # Draw collision warning
    draw.rectangle([10, 10, 600, 80], fill=(255, 0, 0, 200))
    draw.text((20, 20), "⚠ COLLISION RISK", fill="white", font=font)
    draw.text((20, 50), collision["reason"], fill="white", font=font)
    
    img.save(output_path)
    return True


def main():
    """Main collision detection."""
    
    output_dir = Path("collision_detections")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print(" Collision Detection Analysis")
    print("=" * 80)
    print()
    
    collisions = detect_collisions()
    
    if not collisions:
        print("✅ No imminent collisions detected")
        return
    
    print(f"⚠ Found {len(collisions)} potential collision scenarios:\n")
    
    results = []
    
    for i, collision in enumerate(collisions, 1):
        print(f"{i}. Image: {collision['image_file']}")
        print(f"   Objects: {' + '.join(collision['objects'])}")
        print(f"   Reason: {collision['reason']}")
        print()
        
        # Create annotated image
        output_path = output_dir / f"collision_{i:02d}_{collision['image_file']}"
        if create_collision_image(collision, output_path):
            print(f"   ✅ Saved: {output_path}")
        print()
        
        results.append({
            "id": i,
            "image": collision["image_file"],
            "objects": collision["objects"],
            "reason": collision["reason"],
            "output": str(output_path)
        })
    
    # Save JSON report
    report_path = output_dir / "collision_report.json"
    with open(report_path, 'w') as f:
        json.dump({"total_collisions": len(collisions), "detections": results}, f, indent=2)
    
    print(f"📊 Report saved: {report_path}")
    print(f"📁 Images saved in: {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
