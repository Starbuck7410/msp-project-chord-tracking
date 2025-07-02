#!/usr/bin/python
from itertools import combinations
from chroma import extract_chromagram
from ungabunga import *
from viterbi_a import *


chroma, t = extract_chromagram('vit2.wav', plot = False)

threshold = [0.1, 0.7] # Low threshold and high threshold



detections = detect_note_events(chroma, t, threshold[0])

for detection in detections:
    print(detection)

song = notes_to_event_matrix(detections)

song = filter_events(song)

for line in song:
    print(line)


print(viterbi(song, test = False))

print("STD")

chords = group_close_detections(detect_chords_unga_bunga(detections))


for chord in chords:
    print(chord)



# song = [
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
