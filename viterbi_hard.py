#!/usr/bin/python

import sys
from chromagram import extract_chromagram
from pattern_matching import *
from viterbi import *
from note_detector import *

chroma, t = extract_chromagram(sys.argv[1], plot = False)

detections = detect_note_events(chroma, t)

song = notes_to_event_matrix(detections)

song = filter_events(song)

print(viterbi(song))