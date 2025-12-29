#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing - Interactive Notebook
# **Protex AI - Computer Vision Ops Pipeline**
# 
# This notebook implements video preprocessing with:
# - Moving window buffer for adaptive threshold computation
# - Perceptual hashing for duplicate detection (applied first)
# - Granular visualization at each step

# ## 1. Configuration Variables

# In[1]:


# ============= CONFIG =============
from pathlib import Path

# INPUT/OUTPUT
VIDEO_PATH = "../data/timelapse_test.mp4"  # Relative to script/notebook
OUTPUT_DIR = "frames"

# SAMPLING
# For streaming-style we process every frame (frame_step=1),
# but we still keep DESIRED_FPS for metadata/logging.
DESIRED_FPS = None  # Will be set equal to original_fps after opening the video

# RESOLUTION
TARGET_WIDTH = 960
TARGET_HEIGHT = 544
total_pixels = TARGET_WIDTH * TARGET_HEIGHT

# BUFFER SETTINGS
BUFFER_SIZE = 50  # Number of frames in moving window
BUFFER_OVERLAP = 10  # Frames to overlap between buffers

# PERCEPTUAL HASH / DEDUP
HASH_SIZE = 16
HASH_THRESHOLD = 5  # small Hamming distance allowed
TIME_WINDOW_FRAMES = 200  # only compare to last N frames for dedup
MIN_FRAMES_BETWEEN_KEPT = 5  # min spacing between kept frames (in original index)

# ADAPTIVE THRESHOLDS
BRIGHTNESS_PERCENTILE = 10  # Use 10th percentile as min brightness
BLUR_PERCENTILE = 15  # Use 15th percentile as min sharpness

# NOISE / TEXTURE FILTERS
USE_NOISE_FILTER = False  # <--- start with False to ensure we keep frames
NOISE_THRESHOLD = 3000  # baseline for combined noise score (if used)
COLOR_VAR_THRESHOLD = 2000  # threshold for color variance
MIN_EDGE_PIXELS = 500  # very low edges => flat / empty
MAX_EDGE_PIXELS = int(total_pixels * 0.30)  # too many edges => likely static noise

# VISUALIZATION
SHOW_SAMPLES = 3  # For preview
FIGSIZE = (15, 5)

DISPLAY_WIDTH = 160  # Smaller frame size for grid display
DISPLAY_HEIGHT = 90
BUFFERS_TO_SHOW = 3  # Show only first 3 buffers in grid
FRAMES_PER_SECTION = 10  # 3 sections × 10 frames = 30 frames per grid

MAX_DEBUG_BUFFERS = 5  # Only log/visualise 5 random buffers
MAX_REJECTED_EXAMPLES = 3  # Show 3 random rejected frames


# ## 2. Imports

# In[2]:


# ============= IMPORTS =============
import random
from collections import deque

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imagehash
from PIL import Image
import json


# In[3]:


# ============= SETUP =============
# Clean up old frames from previous run
import shutil
if Path(OUTPUT_DIR).exists():
    shutil.rmtree(OUTPUT_DIR)
    print(f"✓ Cleaned up old frames from {OUTPUT_DIR}")

Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

print("✓ Imports loaded")
print(f"✓ Output directory: {OUTPUT_DIR}")


# ## 3. Helper Functions

# In[4]:


# ============= HELPER FUNCTIONS =============


def compute_perceptual_hash(frame):
    """Compute perceptual hash using pHash (more robust for timelapse)."""
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil_img, hash_size=HASH_SIZE)


def compute_brightness(frame):
    """Compute mean brightness of frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def compute_sharpness(frame):
    """Compute sharpness using Laplacian variance."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_edge_count(frame):
    """Simple texture measure: count edge pixels via Canny."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return int(np.count_nonzero(edges))


def compute_entropy(frame):
    """Compute entropy of grayscale image (measure of randomness)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten().astype(np.float32)
    hist_sum = np.sum(hist)
    if hist_sum == 0:
        return 0.0
    hist /= hist_sum  # normalize
    nonzero = hist[hist > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def compute_noise_score(frame):
    """
    Improved noise score combining variance and entropy.

    - High variance: pixel intensities very spread out
    - High entropy: distribution is very random
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    var = float(np.var(gray))
    ent = compute_entropy(frame)
    return var + ent * 100.0  # entropy weighted to similar scale


def compute_color_variance(frame):
    """Compute average variance across RGB channels."""
    b, g, r = cv2.split(frame)
    var_b = float(np.var(b))
    var_g = float(np.var(g))
    var_r = float(np.var(r))
    return (var_b + var_g + var_r) / 3.0


def show_frames(frames, titles, suptitle=""):
    """Display multiple frames side by side."""
    n = len(frames)
    if n == 0:
        print(f"⚠️ No frames to display for: {suptitle}")
        return

    fig, axes = plt.subplots(1, n, figsize=FIGSIZE)
    if n == 1:
        axes = [axes]

    for ax, frame, title in zip(axes, frames, titles):
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# In[5]:


def visualize_buffer_before_after(
    buffer_frames, buffer_indices, kept_frames_in_buffer, buffer_id, max_buffers=BUFFERS_TO_SHOW
):
    """
    Show buffer timeline before and after filtering in one image.
    Top row: All frames in buffer (sampled)
    Bottom row: Kept frames only
    """
    if buffer_id > max_buffers:
        return
    
    # Sample 10 frames evenly from buffer
    n_buffer = len(buffer_frames)
    if n_buffer == 0:
        return
    
    sample_indices = np.linspace(0, n_buffer - 1, min(10, n_buffer), dtype=int)
    
    # Get kept frame indices within this buffer
    kept_indices_set = {idx for _, idx, _ in kept_frames_in_buffer}
    
    fig, axes = plt.subplots(2, 10, figsize=(20, 5))
    
    # Top row: all frames (with red border if rejected)
    for col in range(10):
        ax = axes[0, col]
        if col < len(sample_indices):
            idx = sample_indices[col]
            frame = buffer_frames[idx]
            global_idx = buffer_indices[idx]
            small = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            ax.imshow(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            
            # Red border if rejected, green if kept
            if global_idx in kept_indices_set:
                ax.set_title(f"{global_idx}", fontsize=8, color='green', fontweight='bold')
                for spine in ax.spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(3)
            else:
                ax.set_title(f"{global_idx}", fontsize=8, color='red')
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)
        ax.axis("off")
    
    # Bottom row: kept frames only
    for col in range(10):
        ax = axes[1, col]
        if col < len(kept_frames_in_buffer):
            frame, idx, _ = kept_frames_in_buffer[col]
            small = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            ax.imshow(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{idx}", fontsize=8, color='green', fontweight='bold')
        ax.axis("off")
    
    axes[0, 0].text(-0.5, 0.5, 'BEFORE\n(all)', transform=axes[0, 0].transAxes,
                    fontsize=10, va='center', ha='right', fontweight='bold')
    axes[1, 0].text(-0.5, 0.5, 'AFTER\n(kept)', transform=axes[1, 0].transAxes,
                    fontsize=10, va='center', ha='right', fontweight='bold', color='green')
    
    fig.suptitle(f"Buffer {buffer_id} – Before/After Filtering", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# In[6]:


def maybe_store_rejected_example(
    frame, idx, reason, rejected_examples, rejection_state
):
    """
    Keep a uniform random sample of rejected frames (up to MAX_REJECTED_EXAMPLES)
    using reservoir sampling over all rejected frames.
    """
    rejection_state["total"] += 1
    total = rejection_state["total"]
    k = MAX_REJECTED_EXAMPLES

    sample = {
        "frame": frame.copy(),  # defensive copy
        "idx": idx,
        "reason": reason,
    }

    if len(rejected_examples) < k:
        rejected_examples.append(sample)
    else:
        # Replace existing sample with probability k / total
        if random.random() < k / total:
            replace_i = random.randrange(k)
            rejected_examples[replace_i] = sample


# ![flitering01](./dta_processing01.png)
# 

# ![flitering02](./dta_processing02.png)
# 

# In[7]:


def process_buffer(
    buffer_frames,
    buffer_indices,
    buffer_id,
    stats,
    seen_hashes,  # deque[(hash, idx)]
    all_kept_frames,
    debug_buffers,
    rejection_state,
    rejected_examples,
):
    """
    Process one buffer with:
    - time-windowed perceptual hash dedup
    - adaptive brightness / sharpness thresholds
    - optional noise filter (static/snow)
    - minimum spacing between kept frames
    """
    is_debug = buffer_id in debug_buffers

    print(f"\n{'='*60}")
    print(f"📦 BUFFER {buffer_id} | Frames: {len(buffer_frames)}")
    print(f"{'='*60}")
    # Note: visualize_buffer_before_after will be called at the end after filtering

    # ==========================
    # STEP 1: Perceptual hash (time-windowed)
    # ==========================
    if is_debug:
        print("\n[STEP 1] Perceptual Hash Duplicate Detection (time-windowed)")
        print(f"   Time window: last {TIME_WINDOW_FRAMES} frames")
        print(f"   HASH_THRESHOLD: {HASH_THRESHOLD}")

    buffer_with_hash = []
    hash_removed = 0

    for frame, idx in zip(buffer_frames, buffer_indices):
        frame_hash = compute_perceptual_hash(frame)

        # Only compare with hashes from last TIME_WINDOW_FRAMES frames
        recent_hashes = [
            h for (h, fidx) in seen_hashes if idx - fidx <= TIME_WINDOW_FRAMES
        ]
        is_duplicate = any(frame_hash - h <= HASH_THRESHOLD for h in recent_hashes)

        if is_duplicate:
            hash_removed += 1
            stats["hash_duplicates"] += 1
            maybe_store_rejected_example(
                frame, idx, "duplicate", rejected_examples, rejection_state
            )
        else:
            buffer_with_hash.append((frame, idx, frame_hash))
            seen_hashes.append((frame_hash, idx))  # bounded by deque maxlen

    if is_debug:
        print(f"   Removed as duplicates: {hash_removed}")
        print(f"   Remaining after dedup: {len(buffer_with_hash)} frames")

    if not buffer_with_hash:
        if is_debug:
            print("   ⚠️ Buffer empty after dedup")
        return all_kept_frames

    # ==========================
    # STEP 2: Adaptive thresholds
    # ==========================
    if is_debug:
        print("\n[STEP 2] Computing Adaptive Thresholds")

    brightness_values = [compute_brightness(f[0]) for f in buffer_with_hash]
    sharpness_values = [compute_sharpness(f[0]) for f in buffer_with_hash]
    noise_values = [compute_noise_score(f[0]) for f in buffer_with_hash]

    if brightness_values:
        min_brightness = np.percentile(brightness_values, BRIGHTNESS_PERCENTILE)

        if len(brightness_values) >= 10:
            max_brightness = np.percentile(
                brightness_values, 100 - BRIGHTNESS_PERCENTILE
            )
        else:
            max_brightness = 255.0  # no upper cut for tiny buffers

        min_sharpness = np.percentile(sharpness_values, BLUR_PERCENTILE)
    else:
        min_brightness = 30.0
        max_brightness = 240.0
        min_sharpness = 50.0

    adaptive_noise_thr = (
        np.percentile(noise_values, 99) if noise_values else NOISE_THRESHOLD
    )

    if is_debug:
        print(
            f"   Brightness range: [{min_brightness:.2f}, {max_brightness:.2f}] "
            f"(p{BRIGHTNESS_PERCENTILE}–p{100 - BRIGHTNESS_PERCENTILE} or default)"
        )
        print(f"   Sharpness threshold: {min_sharpness:.2f} (p{BLUR_PERCENTILE})")
        if noise_values:
            print(
                f"   Noise stats: min={np.min(noise_values):.2f}, "
                f"mean={np.mean(noise_values):.2f}, "
                f"max={np.max(noise_values):.2f}, "
                f"adaptive_thr(p99)={adaptive_noise_thr:.2f}"
            )
        else:
            print("   Noise stats: no values (empty buffer_with_hash)")

    # ==========================
    # STEP 3: Quality filtering
    # ==========================
    if is_debug:
        print("\n[STEP 3] Quality Filtering")

    dark_removed = 0
    bright_removed = 0
    blur_removed = 0
    texture_removed = 0
    noise_removed = 0
    spacing_removed = 0

    last_kept_idx = stats.get("last_kept_idx", -999999)

    for frame, idx, frame_hash in buffer_with_hash:
        brightness = compute_brightness(frame)
        sharpness = compute_sharpness(frame)
        edge_count = compute_edge_count(frame)

        noise_score = compute_noise_score(frame)
        color_var = compute_color_variance(frame)

        # --- NOISE / STATIC SNOW FILTER (optional) ---
        if USE_NOISE_FILTER and (
            (noise_score > adaptive_noise_thr)
            or (edge_count > MAX_EDGE_PIXELS)
            or (color_var > COLOR_VAR_THRESHOLD)
        ):
            noise_removed += 1
            stats["noisy_frames"] += 1
            maybe_store_rejected_example(
                frame, idx, "noise", rejected_examples, rejection_state
            )
            continue

        # Too dark
        if brightness < min_brightness:
            dark_removed += 1
            stats["dark_frames"] += 1
            maybe_store_rejected_example(
                frame, idx, "dark", rejected_examples, rejection_state
            )
            continue

        # Too bright (if we actually computed an upper bound < 255)
        if brightness > max_brightness and max_brightness < 255.0:
            bright_removed += 1
            stats["bright_frames"] += 1
            maybe_store_rejected_example(
                frame, idx, "bright", rejected_examples, rejection_state
            )
            continue

        # Too blurry
        if sharpness < min_sharpness:
            blur_removed += 1
            stats["blurry_frames"] += 1
            maybe_store_rejected_example(
                frame, idx, "blurry", rejected_examples, rejection_state
            )
            continue

        # Too little texture (very flat)
        if edge_count < MIN_EDGE_PIXELS:
            texture_removed += 1
            stats["low_texture_frames"] += 1
            maybe_store_rejected_example(
                frame, idx, "low_texture", rejected_examples, rejection_state
            )
            continue

        # Enforce minimum spacing between kept frames
        if idx - last_kept_idx < MIN_FRAMES_BETWEEN_KEPT:
            spacing_removed += 1
            maybe_store_rejected_example(
                frame, idx, "too_close_in_time", rejected_examples, rejection_state
            )
            continue

        # ✅ Keep frame
        all_kept_frames.append((frame, idx, frame_hash))
        stats["kept"] += 1
        last_kept_idx = idx

    stats["last_kept_idx"] = last_kept_idx

    if is_debug:
        print(
            f"   Removed: {dark_removed} dark, "
            f"{bright_removed} bright, "
            f"{blur_removed} blurry, "
            f"{texture_removed} low-texture, "
            f"{noise_removed} noisy (noise filter {'ON' if USE_NOISE_FILTER else 'OFF'}), "
            f"{spacing_removed} spacing"
        )
        kept_from_buffer = (
            len(buffer_with_hash)
            - dark_removed
            - bright_removed
            - blur_removed
            - texture_removed
            - noise_removed
            - spacing_removed
        )
        print(f"   Kept from buffer: {kept_from_buffer}")

    # Collect kept frames from this buffer for visualization
    kept_frames_in_buffer = [
        (frame, idx, frame_hash) for frame, idx, frame_hash in all_kept_frames
        if idx in buffer_indices
    ]

    # Visualize before/after filtering
    visualize_buffer_before_after(
        buffer_frames, buffer_indices, kept_frames_in_buffer, buffer_id
    )

    return all_kept_frames


# ## 4. Open Video and Extract Metadata

# In[8]:


cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

original_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Set DESIRED_FPS to original_fps for metadata
DESIRED_FPS = original_fps

# Streaming-style: we do not skip frames by FPS
frame_step = 1

print(f"📹 Video: {VIDEO_PATH}")
print(f"   Original FPS: {original_fps:.2f}")
print(f"   Total frames: {total_frames}")
print(f"   Processing every frame (streaming-style)")


# ## 5. Sample Initial Frames (Preview)

# In[9]:


preview_frames = []
preview_indices = [0, total_frames // 4, total_frames // 2, 3 * total_frames // 4]

for idx in preview_indices[:SHOW_SAMPLES]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        preview_frames.append(frame)

if len(preview_frames) > 0:
    show_frames(
        preview_frames,
        [f"Frame {idx}" for idx in preview_indices[: len(preview_frames)]],
        "Raw Video Preview",
    )
else:
    print("⚠️ Could not read preview frames")


# ## 6. Process Video with Moving Window Buffer

# In[10]:


# Reset video to start for processing
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Storage
all_kept_frames = []  # list of (frame, idx, hash)
buffer_frames = []
buffer_indices = []

# time-windowed seen hashes: deque[(hash, frame_idx)]
seen_hashes = deque(maxlen=TIME_WINDOW_FRAMES * 2)

# For debugging + rejected examples
debug_buffers = []  # buffer_ids chosen for printing/visualizing
rejection_state = {"total": 0}  # for reservoir sampling
rejected_examples = []  # list of {frame, idx, reason}

# Statistics
stats = {
    "total_read": 0,
    "sampled": 0,
    "hash_duplicates": 0,
    "dark_frames": 0,
    "bright_frames": 0,
    "blurry_frames": 0,
    "low_texture_frames": 0,
    "noisy_frames": 0,
    "kept": 0,
    "last_kept_idx": -999999,
}

frame_idx = 0
buffer_count = 0

print("🔄 Starting video processing...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    stats["total_read"] += 1
    frame_idx += 1

    # Streaming-style: consider each frame
    stats["sampled"] += 1

    # Resize once to target resolution
    resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

    # Add to buffer
    buffer_frames.append(resized)
    buffer_indices.append(frame_idx)

    # Process buffer when full
    if len(buffer_frames) >= BUFFER_SIZE:
        buffer_count += 1

        # Reservoir sampling to pick random buffers to debug
        if len(debug_buffers) < MAX_DEBUG_BUFFERS:
            debug_buffers.append(buffer_count)
        else:
            if random.random() < MAX_DEBUG_BUFFERS / buffer_count:
                replace_i = random.randrange(MAX_DEBUG_BUFFERS)
                debug_buffers[replace_i] = buffer_count

        current_buffer_frames = list(buffer_frames)
        current_buffer_indices = list(buffer_indices)

        # Process this buffer
        all_kept_frames = process_buffer(
            current_buffer_frames,
            current_buffer_indices,
            buffer_count,
            stats,
            seen_hashes,
            all_kept_frames,
            debug_buffers,
            rejection_state,
            rejected_examples,
        )

        # Slide the buffer for the NEXT iteration
        if BUFFER_OVERLAP > 0:
            buffer_frames = buffer_frames[-BUFFER_OVERLAP:]
            buffer_indices = buffer_indices[-BUFFER_OVERLAP:]
        else:
            buffer_frames = []
            buffer_indices = []

        # Show some kept frames and stats
        kept_this_buffer = [
            f for f in all_kept_frames if f[1] in current_buffer_indices
        ]
        if kept_this_buffer:
            print(f"\n✅ Buffer {buffer_count}: {len(kept_this_buffer)} frames kept")
            # show_frames(
            #     [f[0] for f in kept_this_buffer[:3]],
            #     [f"Kept {f[1]}" for f in kept_this_buffer[:3]],
            #     f"Buffer {buffer_count} – Kept Sample",
            # )
        else:
            print(f"\n⚠️ Buffer {buffer_count}: no frames kept")

        print("\n📊 Stats so far:")
        print(
            json.dumps(
                {
                    k: (int(v) if isinstance(v, (int, np.integer)) else v)
                    for k, v in stats.items()
                    if k != "last_kept_idx"
                },
                indent=2,
            )
        )

        print(
            f"\n✓ Buffer {buffer_count} processed. "
            f"Overlap (for next buffer): {BUFFER_OVERLAP} frames\n"
        )


# In[ ]:


# Process remaining frames in buffer (if any)
if len(buffer_frames) > 0:
    buffer_count += 1
    if len(debug_buffers) < MAX_DEBUG_BUFFERS:
        debug_buffers.append(buffer_count)
    else:
        if random.random() < MAX_DEBUG_BUFFERS / buffer_count:
            replace_i = random.randrange(MAX_DEBUG_BUFFERS)
            debug_buffers[replace_i] = buffer_count

    print(f"\n📦 Processing final buffer ({len(buffer_frames)} frames)...")
    all_kept_frames = process_buffer(
        buffer_frames,
        buffer_indices,
        buffer_count,
        stats,
        seen_hashes,
        all_kept_frames,
        debug_buffers,
        rejection_state,
        rejected_examples,
    )

cap.release()
print("\n✅ Video processing complete!")
print("📊 Stats:")
print(
    json.dumps(
        {
            k: (int(v) if isinstance(v, (int, np.integer)) else v)
            for k, v in stats.items()
            if k != "last_kept_idx"
        },
        indent=2,
    )
)
print(f"🔍 Debug buffers visualised: {sorted(debug_buffers)}")


# In[ ]:


# --- Show sample rejected frames ---
if rejected_examples:
    print("\n🧹 Sample rejected frames:")
    for ex in rejected_examples:
        print(f"  Frame {ex['idx']}: {ex['reason']}")
    show_frames(
        [ex["frame"] for ex in rejected_examples],
        [f"{ex['idx']} – {ex['reason']}" for ex in rejected_examples],
        "Rejected frame examples",
    )
else:
    print("\n🧹 No rejected frames collected for examples.")


# In[ ]:


xxx


# ## 7. Statistics Summary

# In[ ]:


print("\n" + "=" * 60)
print("📊 PROCESSING STATISTICS")
print("=" * 60)
print(f"Total frames read:        {stats['total_read']:>6}")
print(f"Sampled frames:           {stats['sampled']:>6}")
print(f"Hash duplicates removed:  {stats['hash_duplicates']:>6}")
print(f"Dark frames removed:      {stats['dark_frames']:>6}")
print(f"Blurry frames removed:    {stats['blurry_frames']:>6}")
print(f"Final kept frames:        {stats['kept']:>6}")
print("=" * 60)
if stats["sampled"] > 0:
    print(f"Reduction: {100 * (1 - stats['kept'] / stats['sampled']):.1f}%")
print("=" * 60)

# Visualization
labels = ["Sampled", "Hash Dup", "Dark", "Blurry", "Kept"]
values = [
    stats["sampled"],
    stats["hash_duplicates"],
    stats["dark_frames"],
    stats["blurry_frames"],
    stats["kept"],
]
colors = ["#3498db", "#e74c3c", "#95a5a6", "#f39c12", "#2ecc71"]

plt.figure(figsize=(10, 6))
plt.bar(labels, values, color=colors)
plt.title("Frame Processing Pipeline", fontsize=14, fontweight="bold")
plt.ylabel("Frame Count")
plt.grid(axis="y", alpha=0.3)
for i, v in enumerate(values):
    plt.text(i, v + max(values) * 0.02, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.show()


# ## 8. Save Frames to Disk

# In[ ]:


print("\n💾 Saving frames to disk...")

frames_metadata = []

for save_idx, (frame, orig_idx, frame_hash) in enumerate(all_kept_frames):
    filename = f"frame_{save_idx:05d}.jpg"
    filepath = Path(OUTPUT_DIR) / filename
    cv2.imwrite(str(filepath), frame)

    frames_metadata.append(
        {
            "id": save_idx,
            "file_name": filename,
            "original_frame_idx": int(orig_idx),
            "width": TARGET_WIDTH,
            "height": TARGET_HEIGHT,
            "hash": str(frame_hash),
        }
    )

    if save_idx < 5 or save_idx % 100 == 0:
        print(f"   Saved: {filename} (original frame {orig_idx})")

print(f"\n✓ Saved {len(all_kept_frames)} frames to {OUTPUT_DIR}/")


# ## 9. Save Metadata

# In[ ]:


metadata = {
    "video_path": VIDEO_PATH,
    "original_fps": float(original_fps),
    "desired_fps": float(DESIRED_FPS),
    "target_resolution": f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
    "buffer_size": BUFFER_SIZE,
    "buffer_overlap": BUFFER_OVERLAP,
    "hash_size": HASH_SIZE,
    "hash_threshold": HASH_THRESHOLD,
    "time_window_frames": TIME_WINDOW_FRAMES,
    "min_frames_between_kept": MIN_FRAMES_BETWEEN_KEPT,
    "brightness_percentile": BRIGHTNESS_PERCENTILE,
    "blur_percentile": BLUR_PERCENTILE,
    "use_noise_filter": USE_NOISE_FILTER,
    "noise_threshold": NOISE_THRESHOLD,
    "color_var_threshold": COLOR_VAR_THRESHOLD,
    "statistics": {
        k: (int(v) if isinstance(v, (int, np.integer)) else v)
        for k, v in stats.items()
        if k != "last_kept_idx"
    },
    "num_saved_frames": len(frames_metadata),
    "frames": frames_metadata,
}

metadata_path = Path(OUTPUT_DIR) / "frames_metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Metadata saved to {metadata_path}")


# ## 10. Final Visualization - Sample Output

# In[ ]:


if len(all_kept_frames) > 0:
    n_samples = min(6, len(all_kept_frames))
    indices = np.linspace(0, len(all_kept_frames) - 1, n_samples, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        frame, orig_idx, _ = all_kept_frames[idx]
        axes[i].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Output #{idx} (orig frame {orig_idx})", fontsize=10)
        axes[i].axis("off")

    fig.suptitle("Final Output Samples", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()

print("\n✅ Preprocessing complete!")
print(f"📁 Output: {OUTPUT_DIR}/")
print(f"📊 Frames: {len(all_kept_frames)}")

