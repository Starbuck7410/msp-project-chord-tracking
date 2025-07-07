import numpy as np

def get_chord_names(chord_data):
    chord_dict = {
        0: "C",
        1: "C#",
        2: "D",
        3: "D#",
        4: "E",
        5: "F",
        6: "F#",
        7: "G",
        8: "G#",
        9: "A",
        10: "A#",
        11: "B",
        12: "Cm",
        13: "C#m",
        14: "Dm",
        15: "D#m",
        16: "Em",
        17: "Fm",
        18: "F#m",
        19: "Gm",
        20: "G#m",
        21: "Am",
        22: "A#m",
        23: "Bm"
    }
    chord_indices, _ = chord_data
    unique_indices = np.unique(chord_indices)
    return [chord_dict[i] for i in unique_indices]