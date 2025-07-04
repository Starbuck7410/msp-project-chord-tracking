#!/usr/bin/python

import sys
from chromagram import extract_chromagram
from pattern_matching import *
from note_detector import *

chroma, t = extract_chromagram(sys.argv[1], plot = False)

detections = detect_note_events(chroma, t)

chords = group_close_detections(detect_chords_pattern_matching(detections))

for chord in chords:
    print(chord)


