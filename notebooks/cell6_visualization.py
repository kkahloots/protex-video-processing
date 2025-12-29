# Visualization code for showing 3 sets of 10 consecutive frames per buffer

# Group samples by buffer
from collections import defaultdict
buffer_groups = defaultdict(list)
for buf_idx, pos, frame in buffer_samples:
    buffer_groups[buf_idx].append((pos, frame))

# Select first 3 buffers for visualization
buffers_to_show = sorted(buffer_groups.keys())[:3]

for buf_num in buffers_to_show:
    frames_in_buffer = buffer_groups[buf_num]
    
    # Get 10 consecutive frames from beginning, mid, end
    # Beginning: frames 0-9
    # Mid: frames around BUFFER_SIZE//2
    # End: frames around BUFFER_SIZE-10 to BUFFER_SIZE-1
    
    print(f"\n=== Buffer {buf_num} ===")
    
    # Create figure with 3 rows (beginning, mid, end) x 10 columns
    fig, axes = plt.subplots(3, 10, figsize=(20, 6))
    fig.suptitle(f'Buffer {buf_num}: Beginning, Mid, End (10 frames each)', fontsize=14, fontweight='bold')
    
    sections = ['Beginning', 'Mid', 'End']
    
    for row_idx, section in enumerate(sections):
        for col_idx in range(10):
            ax = axes[row_idx, col_idx]
            
            # Calculate which frame to show
            if section == 'Beginning':
                frame_pos = col_idx
            elif section == 'Mid':
                frame_pos = BUFFER_SIZE//2 - 5 + col_idx
            else:  # End
                frame_pos = BUFFER_SIZE - 10 + col_idx
            
            # Find frame at this position
            frame_found = False
            for pos, frame in frames_in_buffer:
                if pos == frame_pos:
                    # Resize frame to smaller size for display
                    small_frame = cv2.resize(frame, (160, 90))
                    ax.imshow(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))
                    ax.set_title(f'{section[:3]}-{col_idx}', fontsize=8)
                    frame_found = True
                    break
            
            if not frame_found:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=10)
                ax.set_title(f'{section[:3]}-{col_idx}', fontsize=8)
            
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

print(f"\n\u2713 Displayed {len(buffers_to_show)} buffers")
