import numpy as np


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

def soft_detect_note_events(chromagram, t, upper_threshold=0.5, lower_threshold=0.2):
    detections = []
    active_notes = [None] * 12  # Stores on-times, or None if inactive
    activation_certainty = [0] * 12

    for frame_idx in range(chromagram.shape[1]):
        for pitch_class in range(12):
            strength = chromagram[pitch_class, frame_idx]

            if strength > upper_threshold:
                if active_notes[pitch_class] is None:
                    # Note-on surely detected
                    active_notes[pitch_class] = t[frame_idx]
                    activation_certainty[pitch_class] = 2
            elif strength > lower_threshold:
                if active_notes[pitch_class] is None:
                    # Note-on not surely detected
                    active_notes[pitch_class] = t[frame_idx]
                    activation_certainty[pitch_class] = 1
            else:
                if active_notes[pitch_class] is not None:
                    # Note-off detected
                    on_time = active_notes[pitch_class]
                    off_time = t[frame_idx]
                    detections.append((on_time, off_time, pitch_class, activation_certainty[pitch_class]))
                    active_notes[pitch_class] = None

    # Handle notes that were still active at the end
    for pitch_class, on_time in enumerate(active_notes):
        if on_time is not None:
            detections.append((on_time, t[-1], pitch_class, activation_certainty[pitch_class]))

    return detections



def filter_events(events, max_notes=5, min_notes = 2):
    return [row for row in events if (sum(row) <= max_notes and sum(row) >= min_notes)]

def notes_to_event_matrix(detections, step=0.1):
    if not detections:
        return []


    # Determine time range
    start_time = min(note[0] for note in detections)
    end_time = max(note[1] for note in detections)
    num_steps = int(np.ceil((end_time - start_time) / step))

    # Generate full matrix
    full_matrix = [[0] * 12 for _ in range(num_steps)]
    for onset, offset, pitch in detections:
        start_idx = int((onset - start_time) / step)
        end_idx = int((offset - start_time) / step)
        for i in range(start_idx, end_idx + 1):
            if 0 <= i < num_steps:
                full_matrix[i][pitch] = 1

    # Remove consecutive duplicates
    event_matrix = []
    prev_row = None
    for row in full_matrix:
        if row != prev_row:
            event_matrix.append(row)
            prev_row = row

    return event_matrix


def soft_notes_to_event_matrix(detections, step=0.01):
    if not detections:
        return []

    # Determine time range
    start_time = min(note[0] for note in detections)
    end_time = max(note[1] for note in detections)
    num_steps = int(np.ceil((end_time - start_time) / step))

    # Generate full matrix
    full_matrix = [[0] * 12 for _ in range(num_steps)]
    for onset, offset, pitch, activation_certainty in detections:
        start_idx = int((onset - start_time) / step)
        end_idx = int((offset - start_time) / step)
        for i in range(start_idx, end_idx + 1):
            if 0 <= i < num_steps:
                full_matrix[i][pitch] = 0.5 * activation_certainty

    # Remove consecutive duplicates
    event_matrix = []
    prev_row = None
    for row in full_matrix:
        if row != prev_row:
            event_matrix.append(row)
            prev_row = row

    return event_matrix