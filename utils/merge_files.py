import os.path as op
from glob import glob

filenames = sorted(glob(op.join('..', 'datasets',  \
    '*.txt')))

with open('urls.txt', 'w') as wf:
    for fil in filenames:
        with open(fil, 'r') as f:
            for line in f.readlines():
                wf.write(line)
