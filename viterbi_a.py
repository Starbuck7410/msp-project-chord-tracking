import numpy as np


def get_note_vector(note, alpha):
    C = [1+alpha+alpha**3+alpha**7, 0, 0, 0, alpha**4, 0, 0, alpha**2+alpha**7, 0, 0, alpha**6, 0]
    return C[note:] + C[:note]

def get_major_chord(note, alpha):
    a = get_note_vector(note, alpha)
    b = get_note_vector(note + 4, alpha)
    c = get_note_vector(note + 7, alpha)
    return [x + y + z for x, y, z in zip(a, b, c)]

def get_minor_chord(note, alpha):
    a = get_note_vector(note, alpha)
    b = get_note_vector(note + 3, alpha)
    c = get_note_vector(note + 7, alpha)
    return [x + y + z for x, y, z in zip(a, b, c)]


def get_emission_coeff(chord, detection, alpha = 0.75):
    chord_vec = []
    if chord < 12:
        chord_vec = get_major_chord(chord, alpha)
    else:
        chord_vec = get_minor_chord(chord - 12, alpha)
    return sum(a * b for a, b in zip(chord_vec, detection))


def get_transition_matrix(test=False):
    if test:
        return [[1/24]*24 for _ in range(24)]
    # Chords are numbered 0–23:
    #   0 - 11: Major chords (C to B)
    #  12 - 23: Minor chords (C to B)
    transition_matrix = [
        # Major chords
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #  0
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #  1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #  2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #  3
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #  4
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #  5
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #  6
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #  7
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #  8
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #  9
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 11
        # Minor chords
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 12
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 13
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 14
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 15
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 16
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 17
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 18
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 19
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 20
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 21
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 22
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 23
    ]
    return transition_matrix



def get_initial_prob_array():
    return np.full(24, 1/24)


def viterbi(detection_matrix, test = False):
    initial_prob_array = get_initial_prob_array()
    transition_matrix = get_transition_matrix()
    n = len(transition_matrix[0])
    m = len(detection_matrix)
    V = np.zeros((m, n))
    backpointer = np.zeros((m, n))
    for i in range(n):
        V[0][i] = initial_prob_array[i] * get_emission_coeff(i, detection_matrix[0])
        backpointer[0][i] = None
    for t in range(1, m):
        for i in range(n):
            max_prob = -1
            best_prev = None
            for j in range(n):
                prob = V[t-1][j] * transition_matrix[j][i] * get_emission_coeff(i, detection_matrix[t])
                if prob > max_prob:
                    max_prob = prob
                    best_prev = j
            V[t][i] = max_prob
            backpointer[t][i] = best_prev
    max_final_prob = np.max(V[m-1])
    last_state = np.argmax(V[m-1])
    best_path = np.array([last_state])
    counter = 0
    for t in range(m-1, 0, -1):
        print(best_path)
        best_path = np.append(best_path, backpointer[t][best_path[counter]].astype(int))
        counter += 1
    best_path = np.flip(best_path)
    return best_path, max_final_prob


song = [
#    C  C# D  D# E  F  F# G  G# A  A# B
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # D# + A
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],  # D# + F# + A  (F#m7-ish)
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # D + F + A (D minor)
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # C + F (maybe F major over C)
    [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # C# + E + G (C# diminished)
    [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],  # E + F# + G# (E major add6/9 flavor)
    [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0],  # F + G + A (F6 or Fmaj9 no root)
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],  # D + E + A# (spicy altered chord)
]

print(viterbi(song, test = True))