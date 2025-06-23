import numpy as np

def detections_to_pitch_matrix(detections, t, num_classes=12):
    """
    Convert a list of note detections (on, off, pitch) into
    a binary pitch_class * time matrix aligned to t.
    """
    pitch_matrix = np.zeros((num_classes, len(t)))
    frame_duration = t[1] - t[0]

    for on, off, pitch in detections:
        pitch_class = pitch % 12
        start_idx = int(np.searchsorted(t, on))
        end_idx = int(np.searchsorted(t, off))
        pitch_matrix[pitch_class, start_idx:end_idx] = 1
    return pitch_matrix
