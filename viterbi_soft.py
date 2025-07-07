#!/usr/bin/python

import sys
from chromagram import extract_chromagram
from pattern_matching import *
from viterbi import *
from note_detector import *
from chord_names import get_chord_names

chroma, t = extract_chromagram(sys.argv[1], plot = False)

detections = soft_detect_note_events(chroma, t)

song = soft_notes_to_event_matrix(detections)

song = filter_events(song)

print(get_chord_names(viterbi(song)))

