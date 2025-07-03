
def detect_chords_pattern_matching(detections, min_duration=0.1):
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

