import os, sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)   # .../mpe
sys.path.append(parent)