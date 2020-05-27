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

strace/ltrace -tttT sample output:

    1555950992.253210 fclose(0x7f1d799ab620 <unfinished ...>
    1555950992.253296 free(0x6d0c90)                 = <void> <0.000067>
    1555950992.253390 <... fclose resumed> )         = 0 <0.000176>
    1555950992.253415 __fpending(0x7f1d799ab540, 0, 0x7f1d799ac780, 0) = 0 <0.000078>
    1555950992.253523 fileno(0x7f1d799ab540)         = 2 <0.000064>

RCCL non-standard timestamps

rocm-framework-3:24888:25056 [1] 1558545003174220 NCCL INFO AllReduce: opCount 1 sendbuff 0x7f80b54bad00 recvbuff 0x7f81cce12200 count 1001 datatype 7 op 0 root 0 comm
0x7f80467b9770 [nranks=2] stream 0x7f804680c090

VDI (OLD)

:3:rocdevice.cpp            :434 : 17353046108778: Initializing HSA stack.
:3:hip_device_runtime.cpp   :455 : 17353103685127: [7f0eccf0a700] hipGetDeviceCount ( 0x7ffc162ea44c )
:3:hip_device_runtime.cpp   :457 : 17353103685235: [7f0eccf0a700] hipGetDeviceCount: Returned hipSuccess
:3:hip_stream.cpp           :121 : 17353103741889: ihipStreamCreate: 199ad4c0

VDI (NEW)

:3:comgrctx.cpp             :33  : 210788573126: Loading COMGR library.
:3:hip_device_runtime.cpp   :468 : 210790172306: 38985: [7fbe92cca180] hipGetDeviceCount ( 0x7ffef19aeeac )
:3:hip_device_runtime.cpp   :470 : 210790172512: 38985: [7fbe92cca180] hipGetDeviceCount: Returned hipSuccess
:3:hip_device_runtime.cpp   :494 : 210790172558: 38985: [7fbe92cca180] hipSetDevice ( 0 )
:3:hip_device_runtime.cpp   :499 : 210790172578: 38985: [7fbe92cca180] hipSetDevice: Returned hipSuccess
:3:hip_stream.cpp           :179 : 210790172669: 38985: [7fbe92cca180] hipStreamCreateWithFlags ( 0x7ffef19aeee8, 0 )
:3:hip_stream.cpp           :172 : 210790172694: ihipStreamCreate: 2241d40
:3:hip_stream.cpp           :181 : 210790172709: 38985: [7fbe92cca180] hipStreamCreateWithFlags: Returned hipSuccess
:3:hip_device_runtime.cpp   :453 : 210790172759: 38985: [7fbe92cca180] hipGetDevice ( 0x7ffef19aeec4 )

"""

from __future__ import print_function
import getopt
import json
import os
import re
import subprocess
import sys

RE_HCC_TS_REF       = re.compile(r"hcc-ts-ref, prof_name gpu_host_ts, unix_ts (\d+), gpu_ts (\d+)")
RE_HIP_TID          = re.compile(r"hip-api pid:(\d+) tid:(\d+):HIP initialized short_tid#(\d+)\s*\(maps to full_tid: 0x(\w+)\)")
RE_HIP_OPEN         = re.compile(r"<<hip-api pid:(\d+) tid:(\d+)\.(\d+) (.*) @(\d+)")
RE_HIP_CLOSE        = re.compile(r"hip-api pid:(\d+) tid:(\d+)\.(\d+) (.*) ret=\s?(\d+) \((\w+)\)>> \+(\d+) ns")
RE_HCC_PROF_TS_OPX  = re.compile(r"profile:\s+(\w+);\s+(.*);\s+(.*) us;\s+(\d+);\s+(\d+);\s+#(\d+\.\d+\.\d+);\s+(.*)")
RE_HCC_PROF_TS_OP   = re.compile(r"profile:\s+(\w+);\s+(.*);\s+(.*) us;\s+(\d+);\s+(\d+);\s+#(\d+\.\d+\.\d+);")
RE_HCC_PROF_TS      = re.compile(r"profile:\s+(\w+);\s+(.*);\s+(.*) us;\s+(\d+);\s+(\d+);")
RE_HCC_PROF_OP      = re.compile(r"profile:\s+(\w+);\s+(.*);\s+(.*) us;\s+#(\d+\.\d+\.\d+);")
RE_HCC_PROF         = re.compile(r"profile:\s+(\w+);\s+(.*);\s+(.*) us;")
RE_STRACE_UNFINISED = re.compile(r"(\d+)\.(\d+) (.*) <unfinished \.\.\.>")
RE_STRACE_RESUMED   = re.compile(r'(\d+)\.(\d+) <\.\.\. (\w+) resumed> .* = <?(["\w/\.-]+)>? <(.*)>')
RE_STRACE_COMPLETE  = re.compile(r"(\d+)\.(\d+) (.*) = (-?<?\w+>?) <(.*)>")
RE_HSA_OPEN         = re.compile(r"<<hsa-api pid:(\d+) tid:(\d+) (.*) (\(.*\)) @(\d+)")
RE_HSA_CLOSE        = re.compile(r"  hsa-api pid:(\d+) tid:(\d+) (.*) ret=(.*)>> \+(\d+)") # used to capture ' ns' at end, but units were wrong in HSA tracer library
RE_HSA_DISPATCH_HOST= re.compile(r"<<hsa-api pid:(\d+) tid:(\d+) dispatch queue:(.*) agent:(\d+) signal:(\d+) name:'(.*)' tick:(\d+) id:(\d+) >>")
RE_HSA_DISPATCH     = re.compile(r"<<hsa-api pid:(\d+) tid:(\d+) dispatch queue:(.*) agent:(\d+) signal:(\d+) name:'(.*)' start:(\d+) stop:(\d+) id:(\d+) >>")
RE_HSA_BARRIER_HOST = re.compile(r"<<hsa-api pid:(\d+) tid:(\d+) barrier queue:(.*) agent:(\d+) signal:(\d+) dep1:(\d+) dep2:(\d+) dep3:(\d+) dep4:(\d+) dep5:(\d+) tick:(\d+) id:(\d+) >>")
RE_HSA_BARRIER      = re.compile(r"<<hsa-api pid:(\d+) tid:(\d+) barrier queue:(.*) agent:(\d+) signal:(\d+) start:(\d+) stop:(\d+) dep1:(\d+) dep2:(\d+) dep3:(\d+) dep4:(\d+) dep5:(\d+) id:(\d+) >>")
RE_HSA_COPY         = re.compile(r"<<hsa-api pid:(\d+) tid:(\d+) copy agent:(\d+) signal:(\d+) start:(\d+) stop:(\d+) dep1:(\d+) dep2:(\d+) dep3:(\d+) dep4:(\d+) dep5:(\d+) >>")
RE_RCCL_ALLREDUCE   = re.compile(r"(.*):(\d+):(\d+) \[(\d+)\] (\d+) NCCL INFO AllReduce: opCount (.*) sendbuff (.*) recvbuff (.*) count (\d+) datatype (\d+) op (\d+) root 0 comm (.*) \[nranks=(\d+)\] stream (.*)")
#RE_VDI_OPEN         = re.compile(r":\d+:.*:.*:(.*): \[(.*)\] (.*) \( (.*) \)")
#RE_VDI_CLOSE        = re.compile(r":\d+:.*:.*:(.*): \[(.*)\] (.*): Returned (.*)")
RE_VDI_OPEN         = re.compile(r":\d+:.*:.*:(.*):(.*): \[(.*)\] (.*) \( (.*) \)")
RE_VDI_CLOSE        = re.compile(r":\d+:.*:.*:(.*):(.*): \[(.*)\] (.*): Returned (.*)")
RE_VDI_MSG          = re.compile(r":\d+:.*:.*:(.*): (.*)")

count_skipped = 0
count_hip_tid = 0
count_hip_open = 0
count_hip_close = 0
count_hip_missed = 0
count_vdi_open = 0
count_vdi_close = 0
count_vdi_message = 0
count_hcc_prof_ts_opx = 0
count_hcc_prof_ts_op = 0
count_hcc_prof_ts = 0
count_hcc_prof_op = 0
count_hcc_prof = 0
count_hcc_missed = 0
hip_pids = {}
count_json = 0
count_json_skipped = 0
count_gap_duplicate_ts = 0
count_gap_wrapped = 0
count_gap_okay = 0
count_strace_unfinished = 0
count_strace_resumed = 0
count_strace_complete = 0
count_hsa_open = 0
count_hsa_close = 0
count_hsa_dispatch_host = 0
count_hsa_dispatch = 0
count_hsa_barrier_host = 0
count_hsa_barrier = 0
count_hsa_copy = 0
count_hsa_missed = 0
count_rccl_info_allreduce = 0
count_rccl_missed = 0

hsa_pids = {}
hsa_queues = {}

# HIP can nest its calls, so the opnum is not reliable (bug in HIP trace).
# Also, hipLaunchKernel is printed without a closing HIP print.
# The hipLaunchKernel message should amend the preceeding kernel launch info.
hip_events = {}

# HIP's hipExtLaunchMultiKernelMultiDevice will nest many calls to hipLaunchKernel
hip_multikernel = {}

# VDI can nest its calls
# VDI HIP tracing has open/close events, but the messages are interleaved due to multithreading
vdi_events = {}
# some hip calls were missing their open lines
vdi_missing_names_open = ["hipInit","canAccessPeer"]
vdi_missing_names_close = ["ihipModuleLaunchKernel"]


# HSA can nest its calls.
hsa_events = {}

# The set of GPUs is maintained for pretty-printing metadata in the chrome://tracing output.
devices = {}

# Per device HCC timestamps, for gap analysis
gaps = {}
gap_names = {}

# strace/ltrace need to track unfinished calls by name
strace_unfinished = {}

# do we replace hipModuleLaunchKernel et al with the actual kernel names
replace_kernel_launch_with_name = False

# organizing HIP calls into stream groups needs a pid to name mapping
hip_stream_output = False
hip_stream_pids = {}

def print_help():
    print("""
rocm-timeline-generator.py

arguments:
    -f              show flow events from host to HSA queue
    -g              add gap events to output
    -h              this help message
    -k              replace HIP kernel launch function with actual kernel names
    -o filename     output JSON to given filename
    -s              HIP calls with hipStream_t args are grouped separately
    -v              verbose console output (extra debugging)
""")
    sys.exit(0)

try:
    opts,non_opt_args = getopt.gnu_getopt(sys.argv[1:], "fghko:sv")
    output_filename = None
    show_gaps = False
    show_flow = False
    verbose = False
    for o,a in opts:
        if o == "-f":
            show_flow = True
        elif o == "-g":
            show_gaps = True
        elif o == "-h":
            print_help()
        elif o == "-k":
            replace_kernel_launch_with_name = True
        elif o == "-o":
            output_filename = a
        elif o == "-s":
            hip_stream_output = True
        elif o == "-v":
            verbose = True
        else:
            assert False, "unhandled option"
except getopt.GetoptError as err:
    print(err)
    sys.exit(2)

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

def kern_name(full_string):
    """Parse HIP kernel name information."""
    return full_string.strip().split()[1][1:-1]

def kern_to_json(full_string, use_kernel_name=True):
    """Parse HIP kernel information into an 'args' JSON object."""
    parts = full_string.strip().split()
    name = parts[1][1:-1]
    gridDim = parts[2].split(':')[1]
    groupDim = parts[3].split(':')[1]
    sharedMem = parts[4].split(':')[1]
    stream = parts[5].split(':')[1]
    if use_kernel_name:
        return '{"name":"%s", "gridDim":"%s", "groupDim":"%s", "sharedMem":"%s", "stream":"%s"}'%(
                name, gridDim, groupDim, sharedMem, stream)
    else:
        return '{"gridDim":"%s", "groupDim":"%s", "sharedMem":"%s", "stream":"%s"}'%(
                gridDim, groupDim, sharedMem, stream)

def hip_args_to_json(full_string):
    """Parse HIP call parameters into an 'args' JSON object.

    Example:
        hipMemcpyHtoDAsync (0x7fb64a635100, 0x7fbf05503500, 4004, stream:0.2)

    """
    parts = full_string.strip().split()
    counter = 0
    ret = '{'
    for part in parts[1:]:
        if counter > 0:
            ret += ', '
        if part[0] == '(':
            part = part[1:]
        if part[-1] in [')',',']:
            part = part[0:-1]
        if ':' in part:
            try:
                name,value = part.split(':', 1) # hipCtx_t args have an extra ':'
                ret += '"%s":"%s"' % (name,value)
            except:
                print("uh oh")
                print(part)
                print(full_string)
                raise
        else:
            ret += '"arg%d":"%s"' % (counter,part)
        counter += 1
    ret += '}'
    return ret

def hip_get_stream(full_string):
    global hip_stream_pids
    pidstr = None
    parts = full_string.strip().split()
    for part in parts:
        if part[0] == '(':
            part = part[1:]
        if part[-1] in [')',',']:
            part = part[0:-1]
        if 'stream:' in part:
            pidstr = part.split(':')[1]
            break
    if pidstr is None:
        print("failed to retrieve stream ID from '%s'" % full_string)
        sys.exit(1)
    device,stream = pidstr.split('.')
    fake_tid = int(stream)
    fake_pid = -((int(device)+1)*10000)
    hip_stream_pids[fake_pid] = device
    return fake_pid,fake_tid

def hash_pid_agent(pid, agent):
    return int(agent)/int(pid)

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

            match = RE_HIP_TID.search(line)
            if match:
                count_hip_tid += 1
                pid,tid,short_tid,hex_tid = match.groups()
                hip_pids[pid] = None
                if short_tid in hip_events:
                    print("Duplicate short_tid found in HIP event %s" % short_tid)
                    sys.exit(1)
                hip_events[(pid,tid)] = []
                hip_multikernel[(pid,tid)] = False
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
                    if 'hipExtLaunchMultiKernelMultiDevice' in msg:
                        assert (pid,tid) in hip_multikernel
                        hip_multikernel[(pid,tid)] = True
                elif 'hipLaunchKernel' in msg:
                    # hipLaunchKernel doesn't print a closing HIP event
                    count_hip_open -= 1
                    # we might be inside a multi-kernel launch block
                    if hip_multikernel[(pid,tid)]:
                        msg = " ".join(msg.strip().split()[3:])
                        msg = "spacer %s" % msg # because kern_to_json expects a bigger string to split
                        if replace_kernel_launch_with_name:
                            out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%s, "tid":%s, "args":%s},\n'%(
                                kern_name(msg), (int(ts)/1000), int(ns)/1000, pid, tid, kern_to_json(msg,False)))
                        else:
                            out.write('{"name":"hipLaunchKernel", "ph":"X", "ts":%s, "dur":%s, "pid":%s, "tid":%s, "args":%s},\n'%(
                                (int(ts)/1000), int(ns)/1000, pid, tid, kern_to_json(msg)))
                    else:
                        # last item in hip event stack gets an updated msg
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
                count_hip_close += 1
                pid,tid,opnum,new_msg,retcode,retstr,ns = match.groups()
                if (pid,tid) not in hip_events:
                    print("HIP event close before HIP init: (%s,%s)"%(pid,tid))
                    sys.exit(1)
                msg,ts = hip_events[(pid,tid)].pop()
                new_msg = new_msg.strip()
                if not msg.startswith(new_msg):
                    print("event mismatch: '%s'.startswith('%s')" % (msg,new_msg))
                    print(opnum)
                    sys.exit(1)
                if 'hipExtLaunchMultiKernelMultiDevice' in msg:
                    assert (pid,tid) in hip_multikernel
                    hip_multikernel[(pid,tid)] = False
                # we treat hipSetDevice special, with its lone device ID argument becoming part of its name
                # so that the chrome://tracing treats them as distinct boxes
                if 'hipSetDevice' in msg:
                    new_msg = msg
                if 'Kernel' in new_msg and 'hipExtLaunchMultiKernelMultiDevice' not in new_msg:
                    if replace_kernel_launch_with_name:
                        out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%s, "tid":%s, "args":%s},\n'%(
                            kern_name(msg), (int(ts)/1000), int(ns)/1000, pid, tid, kern_to_json(msg,False)))
                    else:
                        out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%s, "tid":%s, "args":%s},\n'%(
                            new_msg, (int(ts)/1000), int(ns)/1000, pid, tid, kern_to_json(msg)))
                else:
                    out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%s, "tid":%s, "args":%s},\n'%(
                        new_msg, (int(ts)/1000), int(ns)/1000, pid, tid, hip_args_to_json(msg)))
                # If the stream argument is available, we create a new group based on stream ID.
                # This group will contain duplicates of HIP events, but organized as streams.
                if hip_stream_output and 'stream:' in msg:
                    pid,tid = hip_get_stream(msg)
                    if 'Kernel' in new_msg:
                        if replace_kernel_launch_with_name:
                            out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%s, "tid":%s, "args":%s},\n'%(
                                kern_name(msg), (int(ts)/1000), int(ns)/1000, pid, tid, kern_to_json(msg,False)))
                        else:
                            out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%s, "tid":%s, "args":%s},\n'%(
                                new_msg, (int(ts)/1000), int(ns)/1000, pid, tid, kern_to_json(msg)))
                    else:
                        out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%s, "tid":%s, "args":%s},\n'%(
                            new_msg, (int(ts)/1000), int(ns)/1000, pid, tid, hip_args_to_json(msg)))
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
                ts = (int(start)/1000)
                dur = (int(stop)-int(start))/1000
                #ts = int(start)
                #dur = int(stop)-int(start)
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
                    msg, (int(start)/1000), (int(stop)-int(start))/1000, pid, tid, args))
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

            #compile(r"<<hsa-api pid:(\d+) tid:(\d+) (.*) (\(.*\)) @(\d+)")
            match = RE_HSA_OPEN.search(line)
            if match:
                count_hsa_open += 1
                pid,tid,func,args,ts = match.groups()
                if (pid,tid) not in hsa_events:
                    hsa_events[(pid,tid)] = []
                if func.startswith('hsa'):
                    hsa_events[(pid,tid)].append((func,args,ts))
                else:
                    print("Unrecognized HSA event message: '%s'" % func)
                    sys.exit(1)
                continue

            #compile(r"  hsa-api pid:(\d+) tid:(\d+) (.*) ret=(.*)>> \+(\d+) ns")
            match = RE_HSA_CLOSE.search(line)
            if match:
                count_hsa_close += 1
                pid,tid,func,retcode,ns = match.groups()
                if (pid,tid) not in hsa_events:
                    print("HSA event close before HSA init: (%s,%s)"%(pid,tid))
                    sys.exit(1)
                func_orig,args,ts = hsa_events[(pid,tid)].pop()
                pid = -int(pid)
                hsa_pids[pid] = None
                ts = int(ts)/1000
                ns = int(ns)/1000
                if not func.startswith(func_orig):
                    print("event mismatch: '%s'.startswith('%s')" % (func,func_orig))
                    sys.exit(1)
                if '\\' in args:
                    print("HSA event with bad escape character")
                    out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%d, "tid":%s},\n'%(
                        func, ts, ns, pid, tid))
                else:
                    out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%d, "tid":%s, "args":{"params":"%s"}},\n'%(
                        func, ts, ns, pid, tid, args))
                continue

            match = RE_HSA_DISPATCH_HOST.search(line)
            if match:
                count_hsa_dispatch_host += 1
                pid,tid,queue,agent,signal,name,tick,did = match.groups()
                key = (pid,agent)
                if key not in hsa_queues:
                    hsa_queues[key] = {}
                if queue not in hsa_queues[key]:
                    index = len(hsa_queues[key])
                    hsa_queues[key][queue] = index
                tid = hsa_queues[key][queue]
                pid = -int(pid)
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
                new_pid = hash_pid_agent(pid,agent)
                key = (pid,agent)
                if key not in hsa_queues:
                    hsa_queues[key] = {}
                if queue not in hsa_queues[key]:
                    index = len(hsa_queues[key])
                    hsa_queues[key][queue] = index
                tid = hsa_queues[key][queue]
                ts = (int(start)/1000)
                dur = (int(stop)-int(start))/1000
                out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%s, "tid":%s},\n'%(
                    name, ts, dur, new_pid, tid))
                if show_flow:
                    out.write('{"name":"%s", "cat":"dispatch", "ph":"f", "ts":%s, "pid":%s, "tid":%s, "id":%s},\n'%(
                        name, ts, new_pid, tid, did))
                continue

            match = RE_HSA_BARRIER_HOST.search(line)
            if match:
                count_hsa_barrier_host += 1
                name = 'barrier'
                pid,tid,queue,agent,signal,dep1,dep2,dep3,dep4,dep5,tick,did = match.groups()
                key = (pid,agent)
                if key not in hsa_queues:
                    hsa_queues[key] = {}
                if queue not in hsa_queues[key]:
                    index = len(hsa_queues[key])
                    hsa_queues[key][queue] = index
                tid = hsa_queues[key][queue]
                pid = -int(pid)
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
                new_pid = hash_pid_agent(pid,agent)
                key = (pid,agent)
                if key not in hsa_queues:
                    hsa_queues[key] = {}
                if queue not in hsa_queues[key]:
                    index = len(hsa_queues[key])
                    hsa_queues[key][queue] = index
                tid = hsa_queues[key][queue]
                ts = (int(start)/1000)
                dur = (int(stop)-int(start))/1000
                out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%s, "tid":%s, "args":{"dep1":"%s","dep2":"%s","dep3":"%s","dep4":"%s","dep5":"%s"}},\n'%(
                    name, ts, dur, new_pid, tid, dep1, dep2, dep3, dep4, dep5))
                if show_flow:
                    out.write('{"name":"%s", "cat":"barrier", "ph":"f", "ts":%s, "pid":%s, "tid":%s, "id":%s},\n'%(
                        name, ts, new_pid, tid, did))
                continue

            match = RE_HSA_COPY.search(line)
            if match:
                count_hsa_copy += 1
                name = 'copy'
                pid,tid,agent,signal,start,stop,dep1,dep2,dep3,dep4,dep5 = match.groups()
                new_pid = hash_pid_agent(pid,agent)
                key = (pid,agent)
                if key not in hsa_queues:
                    hsa_queues[key] = {}
                #tid = hsa_queues[key][queue]
                tid = -1
                ts = (int(start)/1000)
                dur = (int(stop)-int(start))/1000
                out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":%s, "tid":%s, "args":{"dep1":"%s","dep2":"%s","dep3":"%s","dep4":"%s","dep5":"%s"}},\n'%(
                    name, ts, dur, new_pid, tid, dep1, dep2, dep3, dep4, dep5))
                continue

            if 'hsa-api' in line:
                count_hsa_missed += 1
                if count_hsa_missed == 1:
                    print(line)
                continue

            #compile(r"(\w+):(\d+):(\d+) \[(\d+)\] (\d+) NCCL INFO AllReduce: opCount (\w+) sendbuff (.*) recvbuff (.*) count (\d+) datatype (\d+) op (\d+) root 0 comm (\d+) \[nranks=(\d+)\] stream (.*)")
            match = RE_RCCL_ALLREDUCE.search(line)
            if match:
                count_rccl_info_allreduce += 1
                host,pid,tid,device,ts,opCount,sendbuff,recvbuff,count,datatype,op,comm,nranks,stream = match.groups()
                out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":-4000, "tid":%s, "args":{"stream":"%s","comm":"%s"}},\n'%(
                    "AllReduce count=%s datatype=%s op=%s"%(count,datatype,op),
                    ts, 10, tid, stream, comm))
                continue

            #RE_VDI_OPEN = re.compile(r":\d+:.*:.*:(.*): \[(.*)\] (.*) \( (.*) \)")
            match = RE_VDI_OPEN.search(line)
            if match:
                count_vdi_open += 1
                ts,pid,tid,hipname,args = [x.strip() for x in match.groups()]
                if tid not in vdi_events: vdi_events[tid] = []
                # some hip calls were missing their close lines
                if hipname in vdi_missing_names_close: continue
                vdi_events[tid].append((ts,hipname))
                continue

            # RE_VDI_CLOSE = re.compile(r":\d+:.*:.*:(.*): \[(.*)\] (.*): Returned (.*)")
            match = RE_VDI_CLOSE.search(line)
            if match:
                count_vdi_close += 1
                ts,pid,tid,hipname,args = [x.strip() for x in match.groups()]
                if tid not in vdi_events:
                    # some hip calls were missing their open lines
                    if hipname in vdi_missing_names_open: continue
                    print("missing tid in vdi events")
                    print(line)
                    sys.exit(1)
                # some hip calls were missing their open lines
                if hipname in vdi_missing_names_open: continue
                prev_ts,prev_hipname = vdi_events[tid].pop()
                if hipname != prev_hipname:
                    print("event name mismatch: %s != %s" % (hipname, prev_hipname))
                    print(line)
                    sys.exit(1)
                # VDI outputs NS / 100
                prev_ts = int(prev_ts) / 10
                ts = int(ts) / 10
                out.write('{"name":"%s", "ph":"X", "ts":%s, "dur":%s, "pid":-1234, "tid":%s},\n'%(
                    hipname, prev_ts, ts-prev_ts, int(tid, 16)))
                continue

            match = RE_VDI_MSG.search(line)
            if match:
                count_vdi_message += 1
                continue

            if 'NCCL INFO' in line:
                count_rccl_missed += 1
                continue

            vprint("unparsed line: %s" % line.strip())
            count_skipped += 1

if count_rccl_info_allreduce > 0:
    out.write('{"name":"process_name", "ph":"M", "pid":-4000, "args":{"name":"RCCL"}},\n')

if hsa_pids:
    for pid in hsa_pids:
        out.write('{"name":"process_name", "ph":"M", "pid":%d, "args":{"name":"HSA"}},\n'%pid)

if count_hsa_dispatch or count_hsa_barrier or count_hsa_copy:
    for pid,agent in hsa_queues:
        out.write('{"name":"process_name", "ph":"M", "pid":%s, "args":{"name":"HSA Agent for pid %s agent %s"}},\n'%(hash_pid_agent(pid,agent),pid,agent))

if count_vdi_close:
    out.write('{"name":"process_name", "ph":"M", "pid":-1234, "args":{"name":"HIP/VDI"}},\n')

if count_strace_resumed + count_strace_complete > 0:
    out.write('{"name":"process_name", "ph":"M", "pid":-2000, "args":{"name":"strace/ltrace"}},\n')

if hip_pids:
    for pid in hip_pids:
        out.write('{"name":"process_name", "ph":"M", "pid":%s, "args":{"name":"HIP"}},\n'%pid)

if hip_stream_output:
    for pid in hip_stream_pids:
        pidstr = hip_stream_pids[pid]
        out.write('{"name":"process_name", "ph":"M", "pid":%s, "args":{"name":"HIP Stream Device %s"}},\n'%(pid,pidstr))

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
print("         open hsa lines: %d"%count_hsa_open)
print("        close hsa lines: %d"%count_hsa_close)
print("host dispatch hsa lines: %d"%count_hsa_dispatch_host)
print("     dispatch hsa lines: %d"%count_hsa_dispatch)
print(" host barrier hsa lines: %d"%count_hsa_barrier_host)
print("      barrier hsa lines: %d"%count_hsa_barrier)
print("         copy hsa lines: %d"%count_hsa_copy)
print("       missed hsa lines: %d"%count_hsa_missed)
print("          tid hip lines: %d"%count_hip_tid)
print("         open hip lines: %d"%count_hip_open)
print("        close hip lines: %d"%count_hip_close)
print("       missed hip lines: %d"%count_hip_missed)
print("         open vdi lines: %d"%count_vdi_open)
print("        close vdi lines: %d"%count_vdi_close)
print("      message vdi lines: %d"%count_vdi_message)
print("  prof ts opx hcc lines: %d"%count_hcc_prof_ts_opx)
print("   prof ts op hcc lines: %d"%count_hcc_prof_ts_op)
print("      prof ts hcc lines: %d"%count_hcc_prof_ts)
print("      prof op hcc lines: %d"%count_hcc_prof_op)
print("         prof hcc lines: %d"%count_hcc_prof)
print("       missed hcc lines: %d"%count_hcc_missed)
print("        rccl info lines: %d"%count_rccl_info_allreduce)
print("      missed rccl lines: %d"%count_rccl_missed)
print("             JSON lines: %d"%count_json)
print("     skipped JSON lines: %d"%count_json_skipped)
print("unfinished strace lines: %d"%count_strace_unfinished)
print("   resumed strace lines: %d"%count_strace_resumed)
print("  complete strace lines: %d"%count_strace_complete)
print("       duplicate gap ts: %d"%count_gap_duplicate_ts)
print("      wrapped gap event: %d"%count_gap_wrapped)
print("         okay gap event: %d"%count_gap_okay)
