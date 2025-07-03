#!/usr/bin/python
from itertools import combinations
from chromagram import extract_chromagram
from pattern_matching import *
from viterbi import *
from note_detector import *



chroma, t = extract_chromagram('vit2.wav', plot = True)



soft_detections = soft_detect_note_events(chroma, t)

for detection in soft_detections:
    print(detection)

detections = detect_note_events(chroma, t)

song = notes_to_event_matrix(detections)

song = filter_events(song)

for line in song:
    print(line)

soft_song = soft_notes_to_event_matrix(soft_detections)

soft_song = filter_events(soft_song)

for line in soft_song:
    print(line)


print(viterbi(soft_song, test = False))
print(viterbi(song, test = False))

# print("STD")

# chords = group_close_detections(detect_chords_unga_bunga(detections))


# for chord in chords:
#     print(chord)



# soft_song = [
# #    C  C# D  D# E  F  F# G  G# A  A# B
#     [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # D# + A
#     [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],  # D# + F# + A  (F#m7-ish)
#     [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # D + F + A (D minor)
#     [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # C + F (maybe F major over C)
#     [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # C# + E + G (C# diminished)
#     [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],  # E + F# + G# (E major add6/9 flavor)
#     [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0],  # F + G + A (F6 or Fmaj9 no root)
#     [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],  # D + E + A# (spicy altered chord)
# ]
