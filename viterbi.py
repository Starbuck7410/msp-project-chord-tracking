import numpy as np

def get_note_vector(note, alpha):
    C = [1+alpha+alpha**3+alpha**7, 0, 0, 0, alpha**4, 0, 0, alpha**2+alpha**7, 0, 0, alpha**6, 0]
    if note > 11:
        note -= 12
    return C[12 - note:] + C[:12 - note]

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

def get_emission_coeff(chord, detection, alpha = 0.5):
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
    #  12 - 23: Minor chords (c to b)
    transition_matrix = [
        # Major -> Major                         Major -> Minor 
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,     0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  #  0
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #  1
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #  2
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #  3
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #  4
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #  5
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,     0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #  6
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #  7
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #  8
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,     1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #  9
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 11
        # Minor -> Major                         Minor -> Minor
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 12
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 13
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 14
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 15
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 16
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 17
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 18
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 19
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 20
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 21
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 22
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 23
    ]
    return transition_matrix



def get_initial_prob_array():
    return np.full(24, 1/24)


def viterbi(detection_matrix, test = False):
    initial_prob_array = get_initial_prob_array()
    transition_matrix = get_transition_matrix(test)
    n = len(transition_matrix[0])
    m = len(detection_matrix)

    V = np.zeros((m, n))
    backpointer = np.zeros((m, n), dtype=int)  # integer type here!

    for i in range(n):
        V[0][i] = initial_prob_array[i] * get_emission_coeff(i, detection_matrix[0])
        backpointer[0][i] = -1  # No previous state for t=0

    for t in range(1, m):
        for i in range(n):
            max_prob = -1
            best_prev = -1
            for j in range(n):
                prob = V[t-1][j] * transition_matrix[j][i] * get_emission_coeff(i, detection_matrix[t])
                if prob > max_prob:
                    max_prob = prob
                    best_prev = j
            V[t][i] = max_prob
            backpointer[t][i] = best_prev

    max_final_prob = np.max(V[m-1])
    last_state = np.argmax(V[m-1])

    best_path = [last_state]
    for t in range(m-1, 0, -1):
        last_state = backpointer[t][last_state]
        best_path.append(last_state)
    best_path.reverse()

    return np.array(best_path), max_final_prob


