#!/usr/bin/env python
"""

Parse files looking for HCC and HIP profile or trace output.
Generate a chrome://tracing JSON file.

HCC print statement:

#define LOG_PROFILE(op, start, end, type, tag, msg) \
{\
    std::stringstream sstream;\
    sstream << "profile: " << std::setw(7) << type << ";\t" \
       << std::setw(40) << tag\
       << ";\t" << std::fixed << std::setw(6) << std::setprecision(1) << (end-start)/1000.0 << " us;";\
    if (HCC_PROFILE_VERBOSE & (HCC_PROFILE_VERBOSE_TIMESTAMP)) {\
       sstream << "\t" << start << ";\t" << end << ";";\
    }\
    if (HCC_PROFILE_VERBOSE & (HCC_PROFILE_VERBOSE_OPSEQNUM)) {\
       sstream << "\t" << *op << ";";\
    }\
    sstream <<  msg << "\n";\
}

Example output:

    profile: barrier;  depcnt=0,acq=none,rel=none;     19.0 us;  83825272529988; 83825272549028; #0.0.1;
    profile:  kernel;  _ZN12_GLOBAL__N_110hip_fill_nILj256EPjmjEEvT0_T1_T2_;     11.2 us;  83831226310426; 83831226321626; #0.0.2;
    profile: barrier;  depcnt=0,acq=sys,rel=sys;     10.2 us;  83831226329626; 83831226339866; #0.0.3;
    profile: barrier;  depcnt=0,acq=none,rel=none;     18.1 us;  83825592607250; 83825592625330; #0.1.1;
    profile: barrier;  depcnt=1,acq=none,rel=none;     25.8 us;  83831227971384; 83831227997144; #0.1.10; deps=#0.1.9
    profile:    copy;  DeviceToDevice_async_fast;     13.1 us;  83831227978264; 83831227991384; #0.1.9; 4004 bytes; 0.0 MB; 0.3 GB/s;

HIP mapping TID print statement:

    tprintf(DB_API, "HIP initialized short_tid#%d (maps to full_tid: 0x%s)\n", _shortTid, tid_ss.str().c_str());

Example output:
    ^[[32mhip-api @83825270582601 pid:602 tid:1:HIP initialized short_tid#1 (maps to full_tid: 0x7f26aaffb700)

HIP opening print statement:

    fprintf(stderr, "%s<<hip-api pid:%d tid:%s @%lu%s\n",
            API_COLOR, tls_tidInfo.pid(), fullStr->c_str(), apiStartTick, API_COLOR_END);

Example output:

    ESC[0m<<hip-api pid:602 tid:1.1 hipInit (0) @83825270592479
    <<hip-api pid:602 tid:1.2 hipGetDeviceCount (0x7ffe3b52ac7c) @83825270603650
    <<hip-api pid:602 tid:1.2 hipGetDeviceCount (0x7ffe3b52ac7c) @83825270603650

HIP closing print statement:

    fprintf(stderr, "  %ship-api pid:%d tid:%d.%lu %-30s ret=%2d (%s)>> +%lu ns%s\n",
	    (localHipStatus == 0) ? API_COLOR : KRED, tls_tidInfo.pid(), tls_tidInfo.tid(),
	    tls_tidInfo.apiSeqNum(), __func__, localHipStatus,
	    ihipErrorString(localHipStatus), ticks, API_COLOR_END);

Example output:

    hip-api pid:602 tid:1.1 hipInit                        ret= 0 (hipSuccess)>> +3196 ns
    hip-api pid:602 tid:1.2 hipGetDeviceCount              ret= 0 (hipSuccess)>> +2585 ns
    hip-api pid:602 tid:1.3 hipGetDeviceCount              ret= 0 (hipSuccess)>> +2234 ns
    hip-api pid:602 tid:1.4 hipDeviceGet                   ret= 0 (hipSuccess)>> +4739 ns

"""

from __future__ import print_function
import os
import re
import sys

RE_HIP_TID        = re.compile(r"HIP initialized short_tid#(\d+)\s*\(maps to full_tid: 0x(\w+)\)")
RE_HIP_OPEN       = re.compile(r"<<hip-api pid:(\d+) tid:(\d+\.\d+) (.*) @(\d+)")
RE_HIP_CLOSE      = re.compile(r"hip-api pid:(\d+) tid:(\d+\.\d+) (.*) ret=\s?(\d+) \((\w+)\)>> \+(\d+) ns")
RE_HCC_PROF_TS_OP = re.compile(r"profile:\s+(\w+);\s+(.*);\s+(.*) us;\s+(\d+);\s+(\d+);\s+(.*);")
RE_HCC_PROF_TS    = re.compile(r"profile:\s+(\w+);\s+(.*);\s+(.*) us;\s+(\d+);\s+(\d+);")
RE_HCC_PROF_OP    = re.compile(r"profile:\s+(\w+);\s+(.*);\s+(.*) us;\s+(.*);")
RE_HCC_PROF       = re.compile(r"profile:\s+(\w+);\s+(.*);\s+(.*) us;")

count_skipped = 0
count_hip_tid = 0
count_hip_open = 0
count_hip_close = 0
count_hip_missed = 0
count_hcc_prof_ts_op = 0
count_hcc_prof_ts = 0
count_hcc_prof_op = 0
count_hcc_prof = 0
count_hcc_missed = 0

hip_events = {}

out = open("out.json", "w")

for filename in sys.argv[1:]:
    if not os.path.isfile(filename):
        print("Skipping '%s': not a file" % filename)
        continue
    with open(filename) as input_file:
        for line in input_file:
            match = None

            match = RE_HIP_TID.search(line)
            if match:
                count_hip_tid += 1
                short_tid,hex_tid = match.groups()
                continue

            match = RE_HIP_OPEN.search(line)
            if match:
                count_hip_open += 1
                pid,tid,msg,ts = match.groups()
                hip_events[(pid,tid)] = msg
                continue

            match = RE_HIP_CLOSE.search(line)
            if match:
                count_hip_close += 1
                pid,tid,msg,retcode,retstr,ns = match.groups()
                if (pid,tid) not in hip_events:
                    print("HIP event close without open: (%s,%s)"%(pid,tid))
                    sys.exit(1)
                continue

            # look for most specific HCC profile first

            match = RE_HCC_PROF_TS_OP.search(line)
            if match:
                count_hcc_prof_ts_op += 1
                optype,msg,us,start,stop,extra = match.groups()
                continue

            match = RE_HCC_PROF_TS.search(line)
            if match:
                count_hcc_prof_ts += 1
                optype,msg,us,start,stop = match.groups()
                continue

            match = RE_HCC_PROF_OP.search(line)
            if match:
                count_hcc_prof_op += 1
                optype,msg,us,extra = match.groups()
                continue

            match = RE_HCC_PROF.search(line)
            if match:
                count_hcc_prof += 1
                optype,msg,us = match.groups()
                continue

            if 'hip-api' in line:
                count_hip_missed += 1
                continue

            if 'profile:' in line:
                count_hcc_missed += 1
                continue

            count_skipped += 1

print(" total skipped lines: %d"%count_skipped)
print("       tid hip lines: %d"%count_hip_tid)
print("      open hip lines: %d"%count_hip_open)
print("     close hip lines: %d"%count_hip_close)
print("    missed hip lines: %d"%count_hip_missed)
print("prof ts op hcc lines: %d"%count_hcc_prof_ts_op)
print("   prof ts hcc lines: %d"%count_hcc_prof_ts)
print("   prof op hcc lines: %d"%count_hcc_prof_op)
print("      prof hcc lines: %d"%count_hcc_prof)
print("    missed hcc lines: %d"%count_hcc_missed)
