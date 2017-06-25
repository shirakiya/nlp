import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
basepath = os.path.normpath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, basepath)
