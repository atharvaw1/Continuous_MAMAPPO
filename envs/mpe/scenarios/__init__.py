import os, sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)   # .../mpe
sys.path.append(parent)

import imp
import os.path as osp

def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    return imp.load_source('', pathname)
