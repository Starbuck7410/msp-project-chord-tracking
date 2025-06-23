#!/usr/bin/python
from itertools import combinations
from chroma import extract_chromagram
from ungabunga import *


chroma, t = extract_chromagram('test2.wav', plot = True)

threshold = [0.1, 0.7] # Low threshold and high threshold



detections = detect_note_events(chroma, t, threshold[0])

for detection in detections:
    print(detection)


chords = group_close_detections(detect_chords_unga_bunga(detections))


for chord in chords:
    print(chord)
