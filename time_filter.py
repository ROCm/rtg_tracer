#!/usr/bin/env python
"""

Parse chrome://tracing JSON file, report first and last timestamps, and ask user to filter out events based on time.

"""

import json
import os
import sys

import numpy as np


def get_input(msg):
    if sys.version_info[0] < 3:
        return raw_input(msg)
    else:
        return input(msg)

def crappyhist(a, bins=50, width=140):
    h, b = np.histogram(a, bins)

    for i in range (0, bins):
        print('{:12.5f}  | {:{width}s} {}'.format(
            b[i],
            '#'*int(width*h[i]/np.amax(h)),
            h[i],
            width=width))
    print('{:12.5f}  |'.format(b[bins]))

blob = json.load(open(sys.argv[1]))

ts_all = [int(event['ts']) for event in blob['traceEvents'] if 'ts' in event]
np_all = np.asarray(ts_all)

ts_begin = np_all.min()
ts_end = np_all.max()

print("Events are in the microsecond range %s : %s, diff = %s" % (ts_begin, ts_end, ts_end-ts_begin))
crappyhist(np_all, width=80)
answer = get_input("Retain events only after which timestamp? ")

ts_prune = int(float(answer))

name,ext = os.path.splitext(sys.argv[1])

out = open("%s_prune_after_%s%s" % (name,ts_prune,ext), 'w')
out.write('{"traceEvents":[\n')

for event in blob['traceEvents']:
    if 'ts' in event:
        ts = int(event['ts'])
        if ts > ts_prune:
            out.write('%s,\n' % json.dumps(event))
    else:
        out.write('%s,\n' % json.dumps(event))

out.write('{}\n]\n}')
