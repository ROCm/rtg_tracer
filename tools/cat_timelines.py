#!/usr/bin/env python
"""
Copyright (c) 2022 Advanced Micro Devices, Inc.

"""

# concatenate all given files, skipping first 2 lines and last 2 lines
# adding comma between files

from __future__ import print_function
import sys

original_lines = open(sys.argv[1]).readlines()
print(original_lines[0], end=" ")
print(original_lines[1], end=" ")
for filename in sys.argv[1:]:
    lines = open(filename).readlines()
    for line in lines[2:-2]:
        print(line, end=" ")
    if filename != sys.argv[-1]:
        print(',', end=" ")
print(original_lines[-2], end=" ")
print(original_lines[-1], end=" ")


