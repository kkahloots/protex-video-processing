# Cell 6: Processing Video with Moving Window Buffer
# Visualize 10 consecutive frames at 3 points (beginning, middle, end)

import numpy as np

# Get total frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define 3 buffer positions
positions = [
    ("BEGINNING", 0),
    ("MIDDLE", total_frames // 2 - 5),
    ("END", max(0, total_frames - 10))
]

# Get mode defaults
min_brightness, min_laplacian_var = get_mode_defaults("balanced")

def is_quality_acceptable(frame, min_brightness, min_laplacian_var):
    small = cv2.resize(frame, (320, 180))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    
    brightness = float(np.mean(gray))
    if brightness < min_brightness:
        return False, "DARK"
    
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < min_laplacian_var:
        return False, "BLUR"
    
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float((edges > 0).mean())
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    hist = hist / (hist.sum() + 1e-8)
    entropy = float(-(hist * np.log2(hist + 1e-8)).sum())
    
    is_noise = lap_var > min_laplacian_var * 3.0 and edge_density > 0.20 and entropy > 4.0
    if is_noise:
        return False, "NOISE"
    
    return True, "KEEP"

# Process each buffer
for title, start_frame in positions:
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Collect 10 frames
    frames = []
    labels = []
    
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        
        quality_ok, reason = is_quality_acceptable(frame, min_brightness, min_laplacian_var)
        
        # Resize for display (smaller)
        thumb = cv2.resize(frame, (160, 90))
        
        # Add label
        color = (0, 255, 0) if quality_ok else (0, 0, 255)
        cv2.putText(thumb, reason, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        frames.append(thumb)
        labels.append(reason)
    
    # Create 2x5 grid
    if len(frames) >= 10:
        row1 = np.hstack(frames[:5])
        row2 = np.hstack(frames[5:10])
        grid = np.vstack([row1, row2])
        
        # Add title
        title_img = np.zeros((30, grid.shape[1], 3), dtype=np.uint8)
        cv2.putText(title_img, f"{title} (frames {start_frame}-{start_frame+9})", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display
        plt.figure(figsize=(15, 4))
        plt.imshow(cv2.cvtColor(np.vstack([title_img, grid]), cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"{title} Buffer")
        plt.tight_layout()
        plt.show()
        
        # Print stats
        keep_count = sum(1 for l in labels if l == "KEEP")
        print(f"{title}: {keep_count}/10 frames kept")

print("✓ Visualization complete")
