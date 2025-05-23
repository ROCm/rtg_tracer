#!/usr/bin/env python
"""
Copyright (c) 2022 Advanced Micro Devices, Inc.

Parse files generated by HSA_TOOLS_LIB rtg_tracer.so and generate a chrome://tracing JSON file.

HSA sampel output
-----------------
    HSA: pid:4905 tid:140302773952640 hsa_agent_get_info (agent_94179622800240, 17, 0x7ffd086dfebc) ret=0 @1811961275577532 +152

HIP sample output
-----------------
    HIP: pid:4905 tid:140302773952640 hipFree  ret=0 @1811962118575233 +34010

roctx sample output
-----------------
ranges
    RTX: pid:4905 tid:140302773952640 some message ret=0 @1811962118575233 +34010
markers
    RTX: pid:4905 tid:140302773952640 some message ret=0 @1811962118578989

strace/ltrace -tttT sample output {beta}
----------------------------------------

    1555950992.253210 fclose(0x7f1d799ab620 <unfinished ...>
    1555950992.253296 free(0x6d0c90)                 = <void> <0.000067>
    1555950992.253390 <... fclose resumed> )         = 0 <0.000176>
    1555950992.253415 __fpending(0x7f1d799ab540, 0, 0x7f1d799ac780, 0) = 0 <0.000078>
    1555950992.253523 fileno(0x7f1d799ab540)         = 2 <0.000064>

HCC sample output (Deprecated, HCC_PROFILE=2 mode, useful for the old 'rpt' reporting tool)
-------------------------------------------------------------------------------------------

    profile: barrier;  depcnt=0,acq=none,rel=none;     19.0 us;  83825272529988; 83825272549028; #0.0.1;
    profile:  kernel;  _ZN12_GLOBAL__N_110hip_fill_nILj256EPjmjEEvT0_T1_T2_;     11.2 us;  83831226310426; 83831226321626; #0.0.2;
    profile: barrier;  depcnt=0,acq=sys,rel=sys;     10.2 us;  83831226329626; 83831226339866; #0.0.3;
    profile: barrier;  depcnt=0,acq=none,rel=none;     18.1 us;  83825592607250; 83825592625330; #0.1.1;
    profile: barrier;  depcnt=1,acq=none,rel=none;     25.8 us;  83831227971384; 83831227997144; #0.1.10; deps=#0.1.9
    profile:    copy;  DeviceToDevice_async_fast;     13.1 us;  83831227978264; 83831227991384; #0.1.9; 4004 bytes; 0.0 MB; 0.3 GB/s;

HIP AMD_LOG_LEVEL=4 output
--------------------------

:3:hip_device_runtime.cpp   :688 : 1233197572 us: [pid:24631 tid: 0x155555409740] ^[[32m hipSetDevice ( 0 ) ^[[0m
:3:hip_device_runtime.cpp   :692 : 1233197574 us: [pid:24631 tid: 0x155555409740] hipSetDevice: Returned hipSuccess :
:3:hip_module.cpp           :533 : 1233197579 us: [pid:24631 tid: 0x155555409740] ^[[32m hipExtModuleLaunchKernel ( 0x0x55610b387b70, 294912, 1, 1, 256, 1, 1, 0, stream:<null>, char array:<null>, 0x7fffffff0d90, event:0, event:0, 0 ) ^[[0m
:4:command.cpp              :359 : 1233197604 us: [pid:24631 tid: 0x155555409740] Command (KernelExecution) enqueued: 0x55611d09a700
:3:rocvirtual.cpp           :733 : 1233197615 us: [pid:24631 tid: 0x155555409740] Arg0:  dst.coerce = ptr:0x152dcaafa000 obj:[0x152d43200000-0x152e87200000]
:3:rocvirtual.cpp           :733 : 1233197627 us: [pid:24631 tid: 0x155555409740] Arg1:  src.coerce = ptr:0x152dcdb1f800 obj:[0x152d43200000-0x152e87200000]
:3:rocvirtual.cpp           :809 : 1233197631 us: [pid:24631 tid: 0x155555409740] Arg2:  height = val:384
:3:rocvirtual.cpp           :809 : 1233197635 us: [pid:24631 tid: 0x155555409740] Arg3:  width = val:96
:3:rocvirtual.cpp           :809 : 1233197638 us: [pid:24631 tid: 0x155555409740] Arg4:  dim_stride = val:1152
:3:rocvirtual.cpp           :809 : 1233197650 us: [pid:24631 tid: 0x155555409740] Arg5:  dim_total = val:1152
:3:rocvirtual.cpp           :809 : 1233197651 us: [pid:24631 tid: 0x155555409740] Arg6:  magic_h = val:1431655766
:3:rocvirtual.cpp           :809 : 1233197655 us: [pid:24631 tid: 0x155555409740] Arg7:  shift_h = val:4
:3:rocvirtual.cpp           :809 : 1233197657 us: [pid:24631 tid: 0x155555409740] Arg8:  magic_w = val:1431655766
:3:rocvirtual.cpp           :809 : 1233197668 us: [pid:24631 tid: 0x155555409740] Arg9:  shift_w = val:2
:3:rocvirtual.cpp           :3198: 1233197669 us: [pid:24631 tid: 0x155555409740] ShaderName : batched_transpose_32x32_half
:3:rocvirtual.cpp           :3412: 1233197673 us: [pid:24631 tid: 0x155555409740] KernargSegmentByteSize = 48 KernargSegmentAlignment = 128

"""

from __future__ import print_function
import getopt
import json
import os
import re
import subprocess
import sys

RE_HSA              = re.compile(r"HSA: pid:(\d+) tid:(\d+) (.*) (\(.*\)) ret=(.*) @(\d+) \+(\d+)")
RE_HSA_DISPATCH_HOST= re.compile(r"HSA: pid:(\d+) tid:(\d+) dispatch queue:(.*) agent:(\d+) signal:(\d+) name:'(.*)' tick:(\d+) id:(\d+) workgroup:{(\d+),(\d+),(\d+)} grid:{(\d+),(\d+),(\d+)}")
RE_HSA_DISPATCH     = re.compile(r"HSA: pid:(\d+) tid:(\d+) dispatch queue:(.*) agent:(\d+) signal:(\d+) name:'(.*)' start:(\d+) stop:(\d+) id:(\d+)")
RE_HSA_BARRIER_HOST = re.compile(r"HSA: pid:(\d+) tid:(\d+) barrier queue:(.*) agent:(\d+) signal:(\d+) dep1:(\d+) dep2:(\d+) dep3:(\d+) dep4:(\d+) dep5:(\d+) tick:(\d+) id:(\d+)")
RE_HSA_BARRIER      = re.compile(r"HSA: pid:(\d+) tid:(\d+) barrier queue:(.*) agent:(\d+) signal:(\d+) start:(\d+) stop:(\d+) dep1:(\d+) dep2:(\d+) dep3:(\d+) dep4:(\d+) dep5:(\d+) id:(\d+)")
RE_HSA_COPY         = re.compile(r"HSA: pid:(\d+) tid:(\d+) copy agent:(\d+) signal:(\d+) start:(\d+) stop:(\d+) dep1:(\d+) dep2:(\d+) dep3:(\d+) dep4:(\d+) dep5:(\d+)")
RE_HIP              = re.compile(r"HIP: pid:(\d+) tid:(\d+) (.*) ret=(.*) @(\d+) \+(\d+)")
RE_ROCTX            = re.compile(r"RTX: pid:(\d+) tid:(\d+) (.*) @(\d+) \+(\d+)")
RE_ROCTX_MARKER     = re.compile(r"RTX: pid:(\d+) tid:(\d+) (.*) @(\d+)")
RE_HCC_PROF_TS_OPX  = re.compile(r"profile:\s+(\w+);\s+(.*);\s+(.*) us;\s+(\d+);\s+(\d+);\s+#(\d+\.\d+\.\d+);\s+(.*)")
RE_HCC_PROF_TS_OP   = re.compile(r"profile:\s+(\w+);\s+(.*);\s+(.*) us;\s+(\d+);\s+(\d+);\s+#(\d+\.\d+\.\d+);")
RE_HCC_PROF_TS      = re.compile(r"profile:\s+(\w+);\s+(.*);\s+(.*) us;\s+(\d+);\s+(\d+);")
RE_HCC_PROF_OP      = re.compile(r"profile:\s+(\w+);\s+(.*);\s+(.*) us;\s+#(\d+\.\d+\.\d+);")
RE_HCC_PROF         = re.compile(r"profile:\s+(\w+);\s+(.*);\s+(.*) us;")
RE_STRACE_UNFINISED = re.compile(r"(\d+)\.(\d+) (.*) <unfinished \.\.\.>")
RE_STRACE_RESUMED   = re.compile(r'(\d+)\.(\d+) <\.\.\. (\w+) resumed> .* = <?(["\w/\.-]+)>? <(.*)>')
RE_STRACE_COMPLETE  = re.compile(r"(\d+)\.(\d+) (.*) = (-?<?\w+>?) <(.*)>")
RE_HIP_LOG          = re.compile(r":(.*):(.*):(.*): (.*) us: \[pid:(.*) tid: 0x(.*)\] (.*)")
RE_ANSI_ESCAPE      = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

count_skipped = 0
count_hip_api = 0
count_hip_missed = 0
count_hcc_prof_ts_opx = 0
count_hcc_prof_ts_op = 0
count_hcc_prof_ts = 0
count_hcc_prof_op = 0
count_hcc_prof = 0
count_hcc_prof_missed = 0
count_gap_duplicate_ts = 0
count_gap_wrapped = 0
count_gap_okay = 0
count_strace_unfinished = 0
count_strace_resumed = 0
count_strace_complete = 0
count_hsa_api = 0
count_hsa_dispatch_host = 0
count_hsa_dispatch = 0
count_hsa_barrier_host = 0
count_hsa_barrier = 0
count_hsa_copy = 0
count_hsa_missed = 0
count_roctx_api = 0
count_roctx_missed = 0
count_hip_log = 0
all_pids = {}

workgroups = {}

# do we replace hipModuleLaunchKernel et al with the actual kernel names
replace_kernel_launch_with_name = False

# HCC_PROFILE=2 stuff
# The set of GPUs is maintained for pretty-printing metadata in the chrome://tracing output.
devices = {}
# Per device HCC timestamps, for gap analysis
gaps = {}
gap_names = {}

# strace/ltrace need to track unfinished calls by name
strace_unfinished = {}

def print_help():
    print("""
rocm-timeline-generator.py

arguments:
    -f              show flow events from host to HSA queue
    -g              add gap events to output
    -h              this help message
    -k              replace HIP kernel launch function with actual kernel names
    -o filename     output JSON to given filename
    -p              group by process ID
    -v              verbose console output (extra debugging)
    -w              print workgroups sizes, sorted
""")
    sys.exit(0)

try:
    opts,non_opt_args = getopt.gnu_getopt(sys.argv[1:], "fghko:pvw")
    group_by_process = False
    output_filename = None
    show_gaps = False
    show_flow = True
    verbose = False
    print_workgroups = False
    for o,a in opts:
        if o == "-f":
            show_flow = False
        elif o == "-g":
            show_gaps = True
        elif o == "-h":
            print_help()
        elif o == "-k":
            replace_kernel_launch_with_name = True
        elif o == "-o":
            output_filename = a
        elif o == "-p":
            group_by_process = True
        elif o == "-v":
            verbose = True
        elif o == "-w":
            print_workgroups = True
        else:
            assert False, "unhandled option"
except getopt.GetoptError as err:
    print(err)
    sys.exit(2)

name_map = {}

agent_counter = 0
agent_to_index = {}
agent_to_queue_map = {}
agent_to_queue_counter = {}
def get_gpu_pid_tid(pid, queue, agent):
    global agent_counter
    global agent_to_index
    global agent_to_queue_map
    global agent_to_queue_counter
    key = (pid,agent)
    if key not in agent_to_index:
        agent_to_index[key] = agent_counter
        agent_to_queue_map[key] = {}
        agent_to_queue_counter[key] = 0
        agent_counter += 1
    agent_index = agent_to_index[key]
    queue_map = agent_to_queue_map[key]
    if queue not in queue_map:
        queue_map[queue] = agent_to_queue_counter[key]
        agent_to_queue_counter[key] += 1
    queue = queue_map[queue]
    tid = agent_index * 1000 + int(queue) + 1
    if group_by_process:
        label = "GPU %s.%s" % (agent_index,queue)
        name_map[(pid,tid)] = label
        return (int(pid),tid)
    else:
        label = "%s.%s" % (agent_index,queue)
        name_map[(0,tid)] = label
        return (0,tid)

copy_to_index = {}
copy_counter = 0
def get_gpu_copy_pid_tid(pid, agent):
    global copy_to_index
    global copy_counter
    key = (pid,agent)
    if key not in copy_to_index:
        copy_to_index[key] = copy_counter
        copy_counter += 1
    copy_index = copy_to_index[key]
    tid = copy_index * 1000
    if group_by_process:
        label = "GPU %s.copy" % copy_index
        name_map[(pid,tid)] = label
        return (int(pid),tid)
    else:
        label = "%s.copy" % copy_index
        name_map[(0,tid)] = label
        return (0,tid)

hsa_to_index = {}
hsa_counter = 1
def get_hsa_pid_tid(pid, tid):
    global hsa_to_index
    global hsa_counter
    key = (pid,tid)
    if key not in hsa_to_index:
        hsa_to_index[key] = hsa_counter
        hsa_counter += 1
    hsa_index = hsa_to_index[key]
    label = "%s.%s" % (pid,tid)
    tid = 100000 + hsa_index
    if group_by_process:
        label = "HSA " + label
        name_map[(pid,tid)] = label
        return (int(pid),tid)
    else:
        name_map[(1,tid)] = label
        return (1,tid)

hip_to_index = {}
hip_counter = 1
def get_hip_pid_tid(pid, tid):
    global hip_to_index
    global hip_counter
    key = (pid,tid)
    if key not in hip_to_index:
        hip_to_index[key] = hip_counter
        hip_counter += 1
    hip_index = hip_to_index[key]
    label = "%s.%s" % (pid,tid)
    tid = 10000000 + hip_index
    if group_by_process:
        label = "HIP " + label
        name_map[(pid,tid)] = label
        return (int(pid),tid)
    else:
        name_map[(2,tid)] = label
        return (2,tid)

hip_log_counter = 0
hip_log_buffer = {}
def append_hip_log_buffer(pid, tid, msg):
    global hip_log_buffer
    key = (pid,tid)
    if key not in hip_log_buffer:
        hip_log_buffer[key] = msg
    else:
        hip_log_buffer[key] += msg

def get_hip_log_buffer(pid, tid):
    global hip_log_buffer
    key = (pid,tid)
    ret = hip_log_buffer[key]
    del hip_log_buffer[key]
    return ret

hip_log_ts = {}
def set_hip_log_ts(pid, tid, ts):
    global hip_log_ts
    key = (pid,tid)
    hip_log_ts[key] = ts

def get_hip_log_ts(pid, tid):
    global hip_log_ts
    key = (pid,tid)
    return hip_log_ts[key]

if not output_filename:
    output_filename = "out.json"
    print("Writing chrome://tracing output to '%s' (use -o to change name)" % output_filename)
else:
    print("Writing chrome://tracing output to '%s'" % output_filename)

def vprint(msg):
    if verbose:
        print(msg)

out = open(output_filename, "w")
out.write("""{
"traceEvents": [
""")

def hash_pid_agent(pid, agent):
    return int(agent)/int(pid)

for filename in non_opt_args:
    if not os.path.isfile(filename):
        print("Skipping '%s': not a file" % filename)
        continue
    if output_filename == filename:
        # skip the output filename we just created; it's not an input file
        continue

    # the JSON attempt above consumes the open file lines, so re-open
    print("Parsing '%s'" % filename)
    with open(filename) as input_file:
        for line in input_file:
            match = None

            #compile(r"  hip-api pid:(\d+) tid:(\d+) (.*) ret=(.*)>> \+(\d+)")
            match = RE_HIP.search(line)
            if match:
                count_hip_api += 1
                pid,tid,func,retcode,ts,dur = match.groups()
                all_pids[pid] = pid
                pid,tid = get_hip_pid_tid(pid, tid)
                args = None
                kernname = None
                if ' [' in func: # look for kernname
                    func,kernname = func.split(' [',1)
                    kernname = kernname[:-1] # strip off ']'
                if '(' in func:
                    func,args = func.split('(',1)
                    args = args[:-1] # strip off ')'
                    args = args.replace('"', '') # remove any quotes, would conflict with JSON format
                if replace_kernel_launch_with_name and kernname is not None:
                    func = kernname
                ts = int(ts)/1000
                dur = int(dur)/1000
                if args:
                    if not replace_kernel_launch_with_name and kernname is not None:
                        out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%d, "tid":%s, "args":{"name":"%s","params":"%s"}},\n'%(
                            func, ts, dur, pid, tid, kernname, args))
                    else:
                        out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%d, "tid":%s, "args":{"params":"%s"}},\n'%(
                            func, ts, dur, pid, tid, args))
                else:
                    if not replace_kernel_launch_with_name and kernname is not None:
                        out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%d, "tid":%s, "args":{"name":"%s"}},\n'%(
                            func, ts, dur, pid, tid, kernname))
                    else:
                        out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%d, "tid":%s},\n'%(
                            func, ts, dur, pid, tid))
                continue

            if 'HIP:' in line:
                count_hip_missed += 1
                continue

            match = RE_HSA.search(line)
            if match:
                count_hsa_api += 1
                pid,tid,func,args,retcode,ts,dur = match.groups()
                all_pids[pid] = pid
                pid,tid = get_hsa_pid_tid(pid, tid)
                ts = int(ts)/1000
                dur = int(dur)/1000
                if '\\' in args:
                    print("HSA event with bad escape character")
                    out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%d, "tid":%s},\n'%(
                        func, ts, dur, pid, tid))
                else:
                    out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%d, "tid":%s, "args":{"params":"%s"}},\n'%(
                        func, ts, dur, pid, tid, args))
                continue

            match = RE_HSA_DISPATCH_HOST.search(line)
            if match:
                count_hsa_dispatch_host += 1
                pid,tid,queue,agent,signal,name,tick,did,wgx,wgy,wgz,gx,gy,gz = match.groups()
                all_pids[pid] = pid
                # we use get_hip_pid_tid here because we want HSA host dispatches to group with HIP
                #pid,tid = get_hsa_pid_tid(pid, tid)
                pid,tid = get_hip_pid_tid(pid, tid)
                workgroups[(int(wgx)*int(wgy)*int(wgz),(wgx,wgy,wgz),name)] = None
                tick = int(tick)/1000
                out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":1, "pid":%s, "tid":%s},\n'%(
                    name, tick, pid, tid))
                if show_flow:
                    out.write('{"name":"%s", "cat":"dispatch", "ph":"s", "ts":%s, "pid":%s, "tid":%s, "id":%s},\n'%(
                        name, tick, pid, tid, did))
                continue

            match = RE_HSA_DISPATCH.search(line)
            if match:
                count_hsa_dispatch += 1
                pid,tid,queue,agent,signal,name,start,stop,did = match.groups()
                all_pids[pid] = pid
                pid,tid = get_gpu_pid_tid(pid, queue, agent)
                ts = (int(start)/1000)
                dur = (int(stop)-int(start))/1000
                out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%s, "tid":%s},\n'%(
                    name, ts, dur, pid, tid))
                if show_flow:
                    out.write('{"name":"%s", "cat":"dispatch", "ph":"f", "ts":%s, "pid":%s, "tid":%s, "id":%s},\n'%(
                        name, ts, pid, tid, did))
                continue

            match = RE_HSA_BARRIER_HOST.search(line)
            if match:
                count_hsa_barrier_host += 1
                name = 'barrier'
                pid,tid,queue,agent,signal,dep1,dep2,dep3,dep4,dep5,tick,did = match.groups()
                all_pids[pid] = pid
                # we use get_hip_pid_tid here because we want HSA host dispatches to group with HIP
                #pid,tid = get_hsa_pid_tid(pid, tid)
                pid,tid = get_hip_pid_tid(pid, tid)
                tick = int(tick)/1000
                out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":1, "pid":%s, "tid":%s, "args":{"dep1":"%s","dep2":"%s","dep3":"%s","dep4":"%s","dep5":"%s"}},\n'%(
                    name, tick, pid, tid, dep1, dep2, dep3, dep4, dep5))
                if show_flow:
                    out.write('{"name":"%s", "cat":"barrier", "ph":"s", "ts":%s, "pid":%s, "tid":%s, "id":%s},\n'%(
                        name, tick, pid, tid, did))
                continue

            match = RE_HSA_BARRIER.search(line)
            if match:
                count_hsa_barrier += 1
                name = 'barrier'
                pid,tid,queue,agent,signal,start,stop,dep1,dep2,dep3,dep4,dep5,did = match.groups()
                all_pids[pid] = pid
                pid,tid = get_gpu_pid_tid(pid, queue, agent)
                ts = (int(start)/1000)
                dur = (int(stop)-int(start))/1000
                out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%s, "tid":%s, "args":{"dep1":"%s","dep2":"%s","dep3":"%s","dep4":"%s","dep5":"%s"}},\n'%(
                    name, ts, dur, pid, tid, dep1, dep2, dep3, dep4, dep5))
                if show_flow:
                    out.write('{"name":"%s", "cat":"barrier", "ph":"f", "ts":%s, "pid":%s, "tid":%s, "id":%s},\n'%(
                        name, ts, pid, tid, did))
                continue

            match = RE_HSA_COPY.search(line)
            if match:
                count_hsa_copy += 1
                name = 'copy'
                pid,tid,agent,signal,start,stop,dep1,dep2,dep3,dep4,dep5 = match.groups()
                all_pids[pid] = pid
                pid,tid = get_gpu_copy_pid_tid(pid, agent)
                ts = (int(start)/1000)
                dur = (int(stop)-int(start))/1000
                out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%s, "tid":%s, "args":{"dep1":"%s","dep2":"%s","dep3":"%s","dep4":"%s","dep5":"%s"}},\n'%(
                    name, ts, dur, pid, tid, dep1, dep2, dep3, dep4, dep5))
                continue

            if 'HSA:' in line:
                count_hsa_missed += 1
                if count_hsa_missed == 1:
                    print(line)
                continue

            # put roctx traces into same group as HIP APIs
            match = RE_ROCTX.search(line)
            if match:
                count_roctx_api += 1
                pid,tid,func,ts,dur = match.groups()
                all_pids[pid] = pid
                pid,tid = get_hip_pid_tid(pid, tid)
                args = None
                kernname = None
                ts = int(ts)/1000
                dur = int(dur)/1000
                out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%d, "tid":%s},\n'%(
                    func, ts, dur, pid, tid))
                continue

            match = RE_ROCTX_MARKER.search(line)
            if match:
                count_roctx_api += 1
                pid,tid,func,ts = match.groups()
                all_pids[pid] = pid
                pid,tid = get_hip_pid_tid(pid, tid)
                args = None
                kernname = None
                ts = int(ts)/1000
                out.write('{"name":"%s", "ph":"i", "ts":%s, "pid":%d, "tid":%s},\n'%(
                    func, ts, pid, tid))
                continue

            if 'RTX:' in line:
                count_roctx_missed += 1
                continue

            # look for most specific HCC profile first
            match = RE_HCC_PROF_TS_OPX.search(line)
            if match:
                count_hcc_prof_ts_opx += 1
                optype,msg,us,start,stop,opnum,extra = match.groups()
                extra = ' '.join(extra.strip().split()).strip() # normalize whitespace in extra text
                opnum_parts = opnum.split('.')
                if len(opnum_parts) != 3:
                    print("RE_HCC_PROF_TS_OPX: HCC event sequence number not recognized '%s'" % opnum)
                    sys.exit(1)
                # these are fake -- pid is GPU device ID, tid is stream ID
                # we use negative offsets for GPU device ID so they don't collide with TensorFlow
                pid = -int(opnum_parts[0])-1
                devices[pid] = None
                tid = opnum_parts[1]
                seqnum = opnum_parts[2]
                if extra:
                    args = '{"seqNum":%s, "extra":"%s"}' % (seqnum,extra)
                else:
                    args = '{"seqNum":%s}' % seqnum
                ts = int(start)
                dur = int(stop)-int(start)
                out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%s, "tid":%s, "args":%s},\n'%(
                    msg, ts, dur, pid, tid, args))
                # we do not log barriers for gap analysis
                if show_gaps and "depcnt" not in msg:
                    if pid not in gaps:
                        gaps[pid] = {}
                        gap_names[pid] = {}
                    if ts in gaps[pid]:
                        count_gap_duplicate_ts += 1
                        vprint("duplicate timestamp found in gap analysis, pid:%s ts:%s" % (pid,ts))
                        if dur > gaps[pid][ts]:
                            vprint("replacing with longer dur %s >  %s for name:%s" % (dur, gaps[pid][ts], msg))
                            gaps[pid][ts] = dur
                            gap_names[pid][ts] = msg
                        else:
                            vprint("keeping existing      dur %s >= %s for name:%s" % (gaps[pid][ts], dur, msg))
                    else:
                        gaps[pid][ts] = dur
                        gap_names[pid][ts] = msg
                continue

            match = RE_HCC_PROF_TS_OP.search(line)
            if match:
                count_hcc_prof_ts_op += 1
                optype,msg,us,start,stop,opnum = match.groups()
                opnum_parts = opnum.split('.')
                if len(opnum_parts) != 3:
                    print("RE_HCC_PROF_TS_OP: HCC event sequence number not recognized '%s'" % opnum)
                    sys.exit(1)
                # these are fake -- pid is GPU device ID, tid is stream ID
                # we use negative offsets for GPU device ID so they don't collide with TensorFlow
                pid = -int(opnum_parts[0])-1
                devices[pid] = None
                tid = opnum_parts[1]
                seqnum = opnum_parts[2]
                args = '{"seqNum":%s}' % seqnum
                out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%s, "tid":%s, "args":%s},\n'%(
                    msg, int(start), int(stop)-int(start), pid, tid, args))
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

            if 'profile:' in line:
                count_hcc_prof_missed += 1
                continue

            match = RE_STRACE_UNFINISED.search(line)
            if match:
                count_strace_unfinished += 1
                ts,ms,text = match.groups()
                text = text.strip()
                if '(' not in text:
                    print("strace unfinished cannot parse function name: %s" % text)
                name = text.split('(')[0]
                strace_unfinished[name] = (text,ts,ms)
                continue

            match = RE_STRACE_RESUMED.search(line)
            #compile(r'(\d+)\.(\d+) <\.\.\. (\w+) resumed> .* = <?(["\w/\.-]+)>? <(.*)>')
            if match:
                count_strace_resumed += 1
                ts,ms,text,retval,dur = match.groups()
                if text not in strace_unfinished:
                    print("strace resumed cannot find corresponding unfinished: %s" % text)
                otext,ots,oms = strace_unfinished[text]
                ts = int(ots)*1000000 + int(oms) # convert s.us to us
                dur = int(float(dur)*1000000)
                otext = "%s)" % otext # adds the closing function paren back in
                escaped_call = json.dumps(otext)
                out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%s, "args":{"name":%s}},\n'%(
                    text,ts,dur,-2000,escaped_call))
                continue

            match = RE_STRACE_COMPLETE.search(line)
            #compile(r"(\d+)\.(\d+) (.*) = (-?<?\w+>?) <(.*)>")
            if match:
                count_strace_complete += 1
                ts,ms,text,retval,dur = match.groups()
                text = text.strip()
                ts = int(ts)*1000000 + int(ms) # convert s.us to us
                dur = int(float(dur)*1000000)
                name = text.split('(')[0]
                escaped_call = json.dumps(text)
                out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%s, "args":{"name":%s}},\n'%(
                    name,ts,dur,-2000,escaped_call))
                continue

            match = RE_HIP_LOG.search(line)
            if match:
              try:
                count_hip_log += 1
                _,_,_,ts,pid,tid,msg = match.groups()
                ts = int(ts)
                all_pids[pid] = pid
                pid,tid = get_hip_pid_tid(pid, tid)
                # scrub any ansi escape terminal color sequences
                msg = RE_ANSI_ESCAPE.sub('', msg)
                msg = msg.strip()
                # find hip api name, if present
                if msg.startswith('hip'):
                    name = msg.split()[0]
                    if name[-1] == ':':
                        name = name[:-1]
                    if 'Returned' in msg:
                        msg = get_hip_log_buffer(pid, tid)
                        if 'hipExtModuleLaunchKernel' in name:  # doesn't have corresponding start
                            old_ts = ts-10
                        else:
                            old_ts = get_hip_log_ts(pid, tid)
                        dur = ts - old_ts
                        if dur > 100000:
                            print("excessive duration at ", count_hip_log)
                            print(ts, old_ts, dur)
                            print(line)
                        if 'hipGetDevice' in name or 'hipSetDevice' in name:
                            continue
                        else:
                            out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%d, "tid":%s, "args":{"params":"%s"}},\n'%(
                                name, old_ts, dur, pid, tid, msg))
                    else:
                        set_hip_log_ts(pid, tid, ts)
                        append_hip_log_buffer(pid, tid, msg)
                else:
                    append_hip_log_buffer(pid, tid, msg)
                continue
              except:
                print("FAIL on line ", count_hip_log)
                print("")
                print(line)
                print("")
                sys.exit(1)

            vprint("unparsed line: %s" % line.strip())
            count_skipped += 1

if not group_by_process:
    out.write('{"name":"process_name", "ph":"M", "pid":0, "args":{"name":"GPU Activity"}},\n')
    out.write('{"name":"process_name", "ph":"M", "pid":1, "args":{"name":"HSA API Activity"}},\n')
    out.write('{"name":"process_name", "ph":"M", "pid":2, "args":{"name":"HIP API Activity"}},\n')
    out.write('{"name":"process_sort_index", "ph":"M", "pid":0, "args":{"sort_index":0}},\n')
    out.write('{"name":"process_sort_index", "ph":"M", "pid":1, "args":{"sort_index":1}},\n')
    out.write('{"name":"process_sort_index", "ph":"M", "pid":2, "args":{"sort_index":2}},\n')

if group_by_process:
    RE_LABEL = re.compile(r"(.*) (\d+)\.(\d+)")
else:
    RE_LABEL = re.compile(r"(\d+)\.(\d+)")

pids = {}
for (pid,tid) in sorted(name_map):
    label = name_map[(pid,tid)]
    if group_by_process:
        if pid not in pids:
            pids[pid] = None
            out.write('{"name":"process_name", "ph":"M", "pid":%s, "args":{"name":"Process %s"}},\n'%(pid,pid))
        if len(all_pids) == 1 and 'GPU' not in label:
            match = RE_LABEL.search(label)
            if match:
                a,b,c = match.groups()
                label = "%s %s" % (a,c)
        out.write('{"name":"thread_name", "ph":"M", "pid":%s, "tid":%d, "args":{"name":"%s"}},\n'%(pid,tid,label))
    elif len(all_pids) == 1 and pid != 0:
        # drop the PID since it is redundant, but don't drop GPU ordinal accidentally
        match = RE_LABEL.search(label)
        if match:
            b,c = match.groups()
            label = "%s" % c
        out.write('{"name":"thread_name", "ph":"M", "pid":%d, "tid":%d, "args":{"name":"%s"}},\n'%(pid,tid,label))
    else:
        out.write('{"name":"thread_name", "ph":"M", "pid":%d, "tid":%d, "args":{"name":"%s"}},\n'%(pid,tid,label))
    out.write('{"name":"thread_sort_index", "ph":"M", "pid":%s, "tid":%d, "args":{"sort_index":%d}},\n'%(pid,tid,tid))

#if count_hsa_dispatch or count_hsa_barrier or count_hsa_copy:
#    for pid,agent in hsa_queues:
#        out.write('{"name":"process_name", "ph":"M", "pid":%s, "args":{"name":"HSA Agent for pid %s agent %s"}},\n'%(hash_pid_agent(pid,agent),pid,agent))

if count_strace_resumed + count_strace_complete > 0:
    out.write('{"name":"process_name", "ph":"M", "pid":-2000, "args":{"name":"strace/ltrace"}},\n')

for fake in devices:
    dev = -(fake + 1)
    out.write('{"name":"process_name", "ph":"M", "pid":%s, "args":{"name":"HCC GPU %s"}},\n'%(fake,dev))

def get_gap_name(dur):
    gaps_default = [10, 20, 50, 100, 1000, 10000, 100000]
    for interval in gaps_default:
        if dur < interval:
            return "gap%s" % interval
    return "gapHUGE"

if show_gaps:
    for fake in sorted(gaps):
        dev = -(fake + 1)
        pid = fake-1000 # something unique from fake HCC GPU pid
        g = gaps[fake]
        out.write('{"name":"process_name", "ph":"M", "pid":%s, "args":{"name":"GAP GPU %s"}},\n'%(pid,dev))
        prev_ts = None
        prev_dur = None
        prev_end = None
        for ts in sorted(g):
            dur = g[ts]
            if not prev_end:
                prev_ts = ts
                prev_dur = dur
                prev_end = ts+dur
                continue
            # check whether the next duration fits inside the previous one, and skip if so
            if ts < prev_end and ts+dur < prev_end:
                count_gap_wrapped += 1
                vprint("event completely inside another ts:%s ts+dur:%s prev_end:%s name:%s wrapped_by:%s" % (
                    ts, ts+dur, prev_end, gap_names[fake][ts][0:40], gap_names[fake][prev_ts][0:40]))
                continue
            else:
                count_gap_okay += 1
            gap_dur = ts - prev_end
            out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%s, "tid":%s},\n'%(
                get_gap_name(gap_dur), prev_end, gap_dur, pid, 0))
            prev_ts = ts
            prev_dur = dur
            prev_end = ts+dur




# write an empty event so we don't need to clean up the last extra comma
out.write("""{}
]
}
""")

print("    total skipped lines: %d"%count_skipped)
print("          api hsa lines: %d"%count_hsa_api)
print("host dispatch hsa lines: %d"%count_hsa_dispatch_host)
print("     dispatch hsa lines: %d"%count_hsa_dispatch)
print(" host barrier hsa lines: %d"%count_hsa_barrier_host)
print("      barrier hsa lines: %d"%count_hsa_barrier)
print("         copy hsa lines: %d"%count_hsa_copy)
print("       missed hsa lines: %d"%count_hsa_missed)
print("          api hip lines: %d"%count_hip_api)
print("       missed hip lines: %d"%count_hip_missed)
print("          log hip lines: %d"%count_hip_log)
print("            roctx lines: %d"%count_roctx_api)
print("     missed roctx lines: %d"%count_roctx_missed)
print("  prof ts opx hcc lines: %d"%count_hcc_prof_ts_opx)
print("   prof ts op hcc lines: %d"%count_hcc_prof_ts_op)
print("      prof ts hcc lines: %d"%count_hcc_prof_ts)
print("      prof op hcc lines: %d"%count_hcc_prof_op)
print("         prof hcc lines: %d"%count_hcc_prof)
print("       missed hcc lines: %d"%count_hcc_prof_missed)
print("unfinished strace lines: %d"%count_strace_unfinished)
print("   resumed strace lines: %d"%count_strace_resumed)
print("  complete strace lines: %d"%count_strace_complete)
print("       duplicate gap ts: %d"%count_gap_duplicate_ts)
print("      wrapped gap event: %d"%count_gap_wrapped)
print("         okay gap event: %d"%count_gap_okay)

if print_workgroups:
    for size,(x,y,z),name in sorted(workgroups):
        print(size,(x,y,z),name)
