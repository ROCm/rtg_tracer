#!/usr/bin/env python
"""

Parse files looking for HCC and HIP profile or trace output.
Generate a chrome://tracing JSON file.
TensorFlow JSON output is parsed, removing the Tensor and Memory categories.

HCC GPU-to-Host timestamp:
    hcc-ts-ref, prof_name gpu_host_ts, unix_ts 1552667119747642, gpu_ts 83823572514501

HCC sample output:

    profile: barrier;  depcnt=0,acq=none,rel=none;     19.0 us;  83825272529988; 83825272549028; #0.0.1;
    profile:  kernel;  _ZN12_GLOBAL__N_110hip_fill_nILj256EPjmjEEvT0_T1_T2_;     11.2 us;  83831226310426; 83831226321626; #0.0.2;
    profile: barrier;  depcnt=0,acq=sys,rel=sys;     10.2 us;  83831226329626; 83831226339866; #0.0.3;
    profile: barrier;  depcnt=0,acq=none,rel=none;     18.1 us;  83825592607250; 83825592625330; #0.1.1;
    profile: barrier;  depcnt=1,acq=none,rel=none;     25.8 us;  83831227971384; 83831227997144; #0.1.10; deps=#0.1.9
    profile:    copy;  DeviceToDevice_async_fast;     13.1 us;  83831227978264; 83831227991384; #0.1.9; 4004 bytes; 0.0 MB; 0.3 GB/s;

HIP mapping TID sample output:

    ^[[32mhip-api @83825270582601 pid:602 tid:1:HIP initialized short_tid#1 (maps to full_tid: 0x7f26aaffb700)

HIP opening sample output:

    ESC[0m<<hip-api pid:602 tid:1.1 hipInit (0) @83825270592479
    <<hip-api pid:602 tid:1.2 hipGetDeviceCount (0x7ffe3b52ac7c) @83825270603650
    <<hip-api pid:602 tid:1.2 hipGetDeviceCount (0x7ffe3b52ac7c) @83825270603650

HIP closing sample output:

    hip-api pid:602 tid:1.1 hipInit                        ret= 0 (hipSuccess)>> +3196 ns
    hip-api pid:602 tid:1.2 hipGetDeviceCount              ret= 0 (hipSuccess)>> +2585 ns
    hip-api pid:602 tid:1.3 hipGetDeviceCount              ret= 0 (hipSuccess)>> +2234 ns
    hip-api pid:602 tid:1.4 hipDeviceGet                   ret= 0 (hipSuccess)>> +4739 ns

"""

from __future__ import print_function
import getopt
import json
import os
import re
import subprocess
import sys

RE_HCC_TS_REF     = re.compile(r"hcc-ts-ref, prof_name gpu_host_ts, unix_ts (\d+), gpu_ts (\d+)")
RE_HIP_TID        = re.compile(r"hip-api pid:(\d+) tid:(\d+):HIP initialized short_tid#(\d+)\s*\(maps to full_tid: 0x(\w+)\)")
RE_HIP_OPEN       = re.compile(r"<<hip-api pid:(\d+) tid:(\d+)\.(\d+) (.*) @(\d+)")
RE_HIP_CLOSE      = re.compile(r"hip-api pid:(\d+) tid:(\d+)\.(\d+) (.*) ret=\s?(\d+) \((\w+)\)>> \+(\d+) ns")
RE_HCC_PROF_TS_OP = re.compile(r"profile:\s+(\w+);\s+(.*);\s+(.*) us;\s+(\d+);\s+(\d+);\s+(.*)")
RE_HCC_PROF_TS    = re.compile(r"profile:\s+(\w+);\s+(.*);\s+(.*) us;\s+(\d+);\s+(\d+);")
RE_HCC_PROF_OP    = re.compile(r"profile:\s+(\w+);\s+(.*);\s+(.*) us;\s+(.*)")
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
hcc_ts_ref = 0
hip_pid = 0
count_json = 0
count_json_skipped = 0

# HIP can nest its calls, so the opnum is not reliable (bug in HIP trace).
# Also, hipLaunchKernel is printed without a closing HIP print.
# The hipLaunchKernel message should amend the preceeding kernel launch info.
hip_events = {}

# The set of GPUs is maintained for pretty-printing metadata in the chrome://tracing output.
devices = {}

try:
    opts,non_opt_args = getopt.gnu_getopt(sys.argv[1:], "o:")
    output_filename = None
    for o,a in opts:
        if o == "-o":
            output_filename = a
        else:
            assert False, "unhandled option"
except getopt.getoptError as err:
    print(err)
    sys.exit(2)

if not output_filename:
    output_filename = "out.json"
    print("Writing chrome://tracing output to '%s' (use -o to change name)" % output_filename)
else:
    print("Writing chrome://tracing output to '%s'" % output_filename)

out = open(output_filename, "w")
out.write("""{
"traceEvents": [
""")

def kern_to_json(full_string):
    """Parse HIP kernel information into an 'args' JSON object."""
    parts = full_string.strip().split()
    name = parts[1][1:-1]
    gridDim = parts[2].split(':')[1]
    groupDim = parts[3].split(':')[1]
    sharedMem = parts[4].split(':')[1]
    stream = parts[5].split(':')[1]
    return '{"name":"%s", "gridDim":"%s", "groupDim":"%s", "sharedMem":"%s", "stream":"%s"}'%(
            name, gridDim, groupDim, sharedMem, stream)

def get_system_ticks():
    global hcc_ts_ref
    if 0 == hcc_ts_ref:
        print("HCC to Unix timestamp reference not found prior to first attempt to output HCC event")
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "get_system_ticks")
        if not os.path.exists(os.path.join(path, "get_system_ticks")):
            print("attempting to compile get_system_ticks program")
            print("cd '%s'; make clean all" % path)
            make_process = subprocess.Popen("make clean all", shell=True, cwd=path)
            if make_process.wait() != 0:
                print("failed to compile get_system_ticks")
                sys.exit(2)
        print("attempting to run get_system_ticks program")
        ticks_process = subprocess.Popen("./get_system_ticks", shell=True, cwd=path, stdout=subprocess.PIPE)
        if ticks_process.wait() != 0:
            print("failed to run get_system_ticks program")
            print(path)
            sys.exit(2)
        try:
            output = ticks_process.stdout.readlines()
            offset = output[2].split()[1]
            out.write("\n")
            hcc_ts_ref = int(offset)
        except:
            print("Failed to parse output of get_system_ticks:")
            for line in output: print(line)
            sys.exit(2)

for filename in non_opt_args:
    if not os.path.isfile(filename):
        print("Skipping '%s': not a file" % filename)
        continue
    if output_filename == filename:
        # skip the output filename we just created; it's not an input file
        continue

    with open(filename) as input_file:

        # attempt to read file as a JSON object. If successful, assume it is TF output.
        try:
            obj = json.load(input_file)
            for event in obj["traceEvents"]:
                # skip Tensor and Memory categories
                if "cat" in event and event["cat"] in ["Tensor","Memory"]:
                    count_json_skipped += 1
                    continue
                elif ("name" in event and event["name"] == "process_name" and
                        ("Allocators" in event["args"]["name"] or
                            "Tensors" in event["args"]["name"])):
                    count_json_skipped += 1
                    continue
                else:
                    count_json += 1
                    out.write("%s,\n"%json.dumps(event))

            continue # filename loop
        except:
            print("Could not parse '%s' as JSON object, assuming HCC/HIP output" % filename)

    # the JSON attempt above consumes the open file lines, so re-open
    with open(filename) as input_file:
        for line in input_file:
            match = None

            match = RE_HCC_TS_REF.search(line)
            if match:
                unix_ts,gpu_ts = match.groups()
                hcc_ts_ref = int(unix_ts) - (int(gpu_ts)/1000)
                print("hcc_ts_ref=%d" % hcc_ts_ref)

            match = RE_HIP_TID.search(line)
            if match:
                count_hip_tid += 1
                pid,tid,short_tid,hex_tid = match.groups()
                hip_pid = pid
                if short_tid in hip_events:
                    print("Duplicate short_tid found in HIP event %s" % short_tid)
                    sys.exit(1)
                hip_events[(pid,tid)] = []
                continue

            match = RE_HIP_OPEN.search(line)
            if match:
                count_hip_open += 1
                pid,tid,opnum,msg,ts = match.groups()
                if (pid,tid) not in hip_events:
                    print("HIP event open before HIP init: (%s,%s)"%(pid,tid))
                    sys.exit(1)
                if msg.startswith('hip'):
                    hip_events[(pid,tid)].append((msg,ts))
                elif 'hipLaunchKernel' in msg:
                    # hipLaunchKernel doesn't print a closing HIP event, so
                    # last item in hip event stack gets an updated msg
                    count_hip_open -= 1
                    old_msg,old_ts = hip_events[(pid,tid)][-1]
                    if 'LaunchKernel' not in old_msg:
                        print("hipLaunchKernel didn't nest as expected into '%s'" % old_msg)
                        sys.exit(1)
                    hip_api = old_msg.split()[0]
                    assert hip_api.startswith('hip')
                    new_msg = " ".join(msg.strip().split()[3:])
                    hip_events[(pid,tid)][-1] = ("%s %s" % (hip_api,new_msg),old_ts)
                else:
                    print("Unrecognized HIP event message: '%s'" % msg)
                    sys.exit(1)
                continue

            match = RE_HIP_CLOSE.search(line)
            if match:
                get_system_ticks()
                count_hip_close += 1
                pid,tid,opnum,new_msg,retcode,retstr,ns = match.groups()
                if (pid,tid) not in hip_events:
                    print("HIP event close before HIP init: (%s,%s)"%(pid,tid))
                    sys.exit(1)
                msg,ts = hip_events[(pid,tid)].pop()
                new_msg = new_msg.strip()
                if not msg.startswith(new_msg):
                    print("event mismatch: '%s'.startswith('%s')" % (item,msg))
                    print(opnum)
                    sys.exit(1)
                if 'Kernel' in new_msg:
                    out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%s, "tid":%s, "args":%s},\n'%(
                        new_msg, (int(ts)/1000)+hcc_ts_ref, int(ns)/1000, pid, tid, kern_to_json(msg)))
                else:
                    out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%s, "tid":%s},\n'%(
                        new_msg, (int(ts)/1000)+hcc_ts_ref, int(ns)/1000, pid, tid))
                continue

            # look for most specific HCC profile first

            match = RE_HCC_PROF_TS_OP.search(line)
            if match:
                get_system_ticks()
                count_hcc_prof_ts_op += 1
                optype,msg,us,start,stop,extra = match.groups()
                if not extra.startswith('#'):
                    print("HCC event extra message string not recognized '%s'" % extra)
                    sys.exit(1)
                extra_parts = extra.split(';')
                opnum = extra_parts[0].strip()
                if not opnum.startswith('#'):
                    print("HCC event sequence number not recognized '%s'" % opnum)
                    sys.exit(1)
                opnum_parts = opnum[1:].split('.')
                if len(opnum_parts) != 3:
                    print("HCC event sequence number not recognized '%s'" % opnum)
                    sys.exit(1)
                # these are fake -- pid is GPU device ID, tid is stream ID
                # we use negative offsets for GPU device ID so they don't collide with TensorFlow
                pid = -int(opnum_parts[0])-1
                devices[pid] = None
                tid = opnum_parts[1]
                out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%s, "tid":%s},\n'%(
                    msg, (int(start)/1000)+hcc_ts_ref, (int(stop)-int(start))/1000, pid, tid))
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

if hip_pid:
    out.write('{"name":"process_name", "ph":"M", "pid":%s, "args":{"name":"HIP"}},\n'%hip_pid)

for fake in devices:
    dev = -(fake + 1)
    out.write('{"name":"process_name", "ph":"M", "pid":%s, "args":{"name":"HCC GPU %s"}},\n'%(fake,dev))


# write an empty event so we don't need to clean up the last extra comma
out.write("""{}
]
}
""")

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
print("          JSON lines: %d"%count_json)
print("  skipped JSON lines: %d"%count_json_skipped)
