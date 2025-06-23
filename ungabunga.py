def detect_note_events(chromagram, t, threshold=0.5):
    detections = []
    active_notes = [None] * 12  # Stores on-times, or None if inactive

    for frame_idx in range(chromagram.shape[1]):
        for pitch_class in range(12):
            strength = chromagram[pitch_class, frame_idx]

            if strength > threshold:
                if active_notes[pitch_class] is None:
                    # Note-on detected
                    active_notes[pitch_class] = t[frame_idx]
            else:
                if active_notes[pitch_class] is not None:
                    # Note-off detected
                    on_time = active_notes[pitch_class]
                    off_time = t[frame_idx]
                    detections.append((on_time, off_time, pitch_class))
                    active_notes[pitch_class] = None

    # Handle notes that were still active at the end
    for pitch_class, on_time in enumerate(active_notes):
        if on_time is not None:
            detections.append((on_time, t[-1], pitch_class))

    return detections

def group_close_detections(detections, merge_gap=0.1):
    """
    Merge note events that are close in time (within merge_gap seconds)
    and have the same pitch class.
    """
    from collections import defaultdict

    # Group detections by pitch class
    grouped = defaultdict(list)
    for start, end, pitch in detections:
        grouped[pitch].append((start, end))

    merged_detections = []

    for pitch, events in grouped.items():
        # Sort by start time
        events.sort()
        current_start, current_end = events[0]

        for start, end in events[1:]:
            if start - current_end <= merge_gap:
                # Extend current event
                current_end = max(current_end, end)
            else:
                # Finalize current event
                merged_detections.append((current_start, current_end, pitch))
                current_start, current_end = start, end

        # Final event
        merged_detections.append((current_start, current_end, pitch))

    # Sort merged results by start time
    merged_detections.sort()
    return merged_detections

def detect_chords_unga_bunga(detections, min_duration=0.1):
    MAJOR = {0, 4, 7}
    MINOR = {0, 3, 7}
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Step 1: Flatten detections into note on/off events
    events = []
    note_durations = {}  # (start, end) per note
    for on, off, pitch in detections:
        events.append((on, 'on', pitch))
        events.append((off, 'off', pitch))
        note_durations[(on, pitch)] = off - on
    events.sort()

    # Step 2: Track active notes with start time
    active_notes = {}  # pitch_class -> start_time
    found_chords = []
    last_time = None

    for time, kind, pitch in events:
        if kind == 'on':
            active_notes[pitch] = time
        elif kind == 'off':
            active_notes.pop(pitch, None)

        # Only try chord detection when note state changes
        if last_time is not None and time != last_time:
            if len(active_notes) >= 3:
                # Convert to list of (pitch, duration)
                note_info = []
                for p, start in active_notes.items():
                    dur = note_durations.get((start, p), 0)
                    note_info.append((p, dur))

                # Pick top 3 longest
                top_notes = sorted(note_info, key=lambda x: -x[1])[:3]
                pitch_classes = sorted(set(p % 12 for p, _ in top_notes))

                # Try all as root
                for root in pitch_classes:
                    intervals = sorted((p - root) % 12 for p in pitch_classes)
                    if set(intervals) == MAJOR:
                        chord_name = f"{NOTE_NAMES[root]} major"
                        if time - last_time >= min_duration:
                            found_chords.append((last_time, time, chord_name))
                        break
                    elif set(intervals) == MINOR:
                        chord_name = f"{NOTE_NAMES[root]} minor"
                        if time - last_time >= min_duration:
                            found_chords.append((last_time, time, chord_name))
                        break
        last_time = time

    return found_chords

