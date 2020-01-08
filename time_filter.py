#!/usr/bin/env python
"""

Parse chrome://tracing JSON file, report first and last timestamps, and ask user to filter out events based on time.

"""

import json
import mmap
import os
import re
import sys

import numpy as np


def get_num_lines(filename):
    with open(filename, "r+") as f:
        buf = mmap.mmap(f.fileno(), 0)
        lines = 0
        readline = buf.readline
        while readline():
            lines += 1
        return lines

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

use_streaming_parser = False
if len(sys.argv) < 2:
    print("missing filename")
    sys.exit(1)
if len(sys.argv) > 2:
    use_streaming_parser = True
filename = sys.argv[1]

# things we might need later
blob = None
linecount = get_num_lines(filename)
toolbar_width = 100
fraction = 0.01
if use_streaming_parser and linecount > 1000:
    portion = int(linecount*fraction)
    def parse_line(index,line):
        try:
            event = json.loads(line[:-2])
        except:
            print("line='%s'"%line)
            raise
        if index%portion == 0:
            sys.stdout.write('-')
            sys.stdout.flush()
        return int(event['ts'])
    with open(filename, 'r') as f:
        print("Streaming JSON file %s" % filename)
        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
        ts_all = [parse_line(index,line) for index,line in enumerate(f) if '"ts":' in line]
        print() # newline
else:
    blob = json.load(open(filename))
    print("Loaded JSON file %s" % filename)
    ts_all = [int(event['ts']) for event in blob['traceEvents'] if 'ts' in event]

np_all = np.asarray(ts_all)
ts_begin = np_all.min()
ts_end = np_all.max()
name,ext = os.path.splitext(filename)

print("Events are in the microsecond range %s : %s, diff = %s" % (ts_begin, ts_end, ts_end-ts_begin))
crappyhist(np_all, width=80)

answer = get_input("Retain events only after which timestamp? ")
ts_start = int(float(answer))

answer = get_input("Do you wish to enter a stopping timestamp, as well? [y/N] ")
if 'y' in answer or 'Y' in answer:
    answer = get_input("Retain events before which timestamp? ")
    ts_stop = int(float(answer))
    outname = "%s_prune_between_%s_%s%s" % (name,ts_start,ts_stop,ext)
else:
    ts_stop = ts_end
    outname = "%s_prune_after_%s%s" % (name,ts_start,ext)

out = open(outname, 'w')

if use_streaming_parser:
    portion = int(linecount*fraction)
    def parse_line(index,line):
        try:
            event = json.loads(line[:-2])
        except:
            print("line='%s'"%line)
            raise
        if index%portion == 0:
            sys.stdout.write('-')
            sys.stdout.flush()
        ts = int(event['ts'])
        if ts >= ts_start and ts <= ts_stop:
            out.write('%s,\n' % json.dumps(event))
    with open(filename, 'r') as f:
        print("Streaming JSON file %s again, to prune output" % filename)
        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
        for index,line in enumerate(f):
            if '"ts":' in line:
                parse_line(index,line)
            else:
                out.write(line)
        print() # newline
else:
    out.write('{"traceEvents":[\n')
    for event in blob['traceEvents']:
        if 'ts' in event:
            ts = int(event['ts'])
            if ts >= ts_start and ts <= ts_stop:
                out.write('%s,\n' % json.dumps(event))
        else:
            out.write('%s,\n' % json.dumps(event))
    out.write('{}\n]\n}')

out.close()
