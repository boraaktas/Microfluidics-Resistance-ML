
import sys
import os
if not sys.stdout or sys.stdout.fileno() < 0:
    sys.stdout = open(os.devnull, 'w')
if not sys.stderr or sys.stderr.fileno() < 0:
    sys.stderr = open(os.devnull, 'w')
