# Add this at the END of _filter_quality_and_noise function, before the return statement:

    # Ensure at least one frame is kept from buffer
    if not kept_this_buffer and buffer_with_hash:
        # Force-keep the first frame that passed dedup
        frame, idx, frame_hash = buffer_with_hash[0]
        kept_this_buffer.append((frame, idx, frame_hash))
        stats["kept"] += 1
        stats["forced_kept_quality"] = stats.get("forced_kept_quality", 0) + 1
        last_kept_idx = idx
        if is_debug:
            print(f"   ⚠️ No frames passed quality filters. Forcibly kept frame {idx}.")
    
    return kept_this_buffer
