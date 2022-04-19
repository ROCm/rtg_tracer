# rocm-timeline-generator

## Quick Start
```bash
apt install libsqlite3-dev libunwind-dev libfmt-dev
git clone https://github.com/jeffdaily/rocm-timeline-generator
cd rocm-timeline-generator
make
HSA_TOOLS_LIB=/path/to/rocm-timeline-generator/rtg_tracer/rtg_tracer.so ./my_program
```

## Documentation
The rocm-timeline-generator project contains
- rtg_tracer – builds and links 'rtg_tracer.so' library for use with HSA (ROCR) profiling
- rocm-timeline-generator.py — python script for translating rtg_tracer output to chrome://tracing JSON input
- rpt – parses profiling data that follows the older HCC_PROFILE=2 format 
- test – nccl-example.cpp for verifying rtg_tracer behavior

Profiling output is produced using one or more of the following environment variables.

| Environment Variable   | Default Value         | Description                                                                                                                                                                                                                                                                 |
|------------------------|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| HSA_TOOLS_LIB          | unset                 | Tell HSA Runtime (ROCr) where to find rtg_tracer.so.                                                                                                                                                                                                                        |
| RTG_FILENAME           | rtg_trace.txt[.<pid>] | Output filename.  pid is always appended to avoid collisions in multi-process use cases.                                                                                                                                                                                    |
| RTG_HSA_API_FILTER     | unset                 | Trace specific HSA calls. Special case 'all', 'core', and 'ext', otherwise simple substring matching. Separate tokens with comma. Examples: RTG_HSA_API_FILTER=all RTG_HSA_API_FILTER=hsa_signal,hsa_agent_get_info # would match hsa_signal_create                         |
| RTG_HSA_API_FILTER_OUT | unset                 | Do not trace specific HSA calls. Simple substring matching. Separate tokens with comma. If this var is set but RTG_HSA_API_FILTER is unset, implies "all" HSA APIs are enabled with the exclusion of this list.                                                             |
| RTG_HSA_HOST_DISPATCH  | false                 | Trace when kernel dispatch is enqueued on the host.                                                                                                                                                                                                                         |
| RTG_HIP_API_FILTER     | all                   | Trace specific HIP calls. Special case 'all', otherwise simple substring matching. Separate tokens with comma. Examples: RTG_HIP_API_FILTER=hipMalloc,hipEventQuery # would match hipMalloc, hipMallocHost, etc.                                                            |
| RTG_HIP_API_FILTER_OUT | unset                 | Do not trace specific HIP calls. Simple substring matching. Separate tokens with comma. If this var is set but RTG_HIP_API_FILTER is unset, implies "all" HIP APIs are enabled with the exclusion of this list. Examples: RTG_HIP_API_FILTER_OUT=hipSetDevice,hipGetDevice  |
| RTG_HIP_API_ARGS       | false                 | If true, capture HIP API name and function arguments, otherwise just the name.                                                                                                                                                                                              |
| HCC_PROFILE            | 0                     | Set to non-zero to enable. "HCC_PROFILE=2" like in the days of the hcc runtime. Legacy HCC profiling, for use with rpt tool.  Generates output just like HCC_PROFILE=2 used to.                                                                                             |
| RTG_PROFILE            | true                  | Enable profiling of device kernels and barriers.                                                                                                                                                                                                                            |
| RTG_PROFILE_COPY       | true                  | Enable profiling of device async copy operations (noticeable overhead) initiated by  hsa_amd_memory_async_copy hsa_amd_memory_async_copy_rect                                                                                                                               |
| RTG_VERBOSE            | 0                     | 1 – Print env var settings 2 – Print list of HIP APIs that are going to be traced.                                                                                                                                                                                          |

## Recommended Profiling for First Time
The more you choose to profile, the higher the overhead imposed by the profiling.  That said, a good place to start would be:

HSA_TOOLS_LIB=/path/to/rtg_tracer.so RTG_HIP_API_FILTER=all

This will trace the HIP API and the HSA AQL packets (GPU activity) and is useful for identifying gaps where the GPU(s) are idle.

## Generating chrome://tracing JSON using rocm-timeline-generator.py
Once you have produced one or more text files (rtg_trace.txt.<PID>), you then use the rocm-timeline-generator.py script to produce a JSON file for input to the chrome://tracing application.

`./rocm-timeline-generator.py rtg_trace.txt.1123 rtg_trace.txt.1124`

```bash
rocm-timeline-generator.py

arguments:
    -k              replace HIP kernel launch function (hipLaunchKernel et al), with actual kernel names
    -o filename     output JSON to given filename
    -p              group traces by process ID instead of domain, e.g., HIP, HSA, GPU
    -v              verbose console output (extra debugging)
    -w              print workgroup sizes, sorted --- must have used RTG_HSA_HOST_DISPATCH with rtg_tracer
```
  
## Examples

### HCC_PROFILE=2 and rpt
Remember the summary of top-10 kernels we used to get from the hcc runtime and the rpt tool?  You can still do that.

rtg_trace.txt.23107 was created using HCC_PROFILE=2 HSA_TOOLS_LIB=../rtg_tracer/rtg_tracer.so ./nccl-example

```bash
../rpt rtg_trace.txt.23107
ROI_START: GPU0         0.000000:      +0.00 kernel  #0.1.0        1: __amd_rocclr_fillBuffer.kd
ROI_STOP : GPU0        69.178381:      +0.00 barrier #0.0.1      168:
ROI_TIME=   0.069 secs

Resource=GPU0 Showing 10/10 records    1.20% busy
      Total(%)    Time(us)    Calls  Avg(us)  Min(us)  Max(us)  Name
        67.29%    46555.7        1  46555.7  46555.7  46555.7  gap 10000us-100000us
        25.28%    17488.3       32    546.5    126.1    798.0  gap 100us-1000us
         2.78%     1922.5       57     33.7     25.1     49.6  gap 20us-50us
         2.12%     1465.7        1   1465.7   1465.7   1465.7  gap 1000us-10000us
         0.90%      622.2       10     62.2     50.6     80.8  gap 50us-100us
         0.45%      313.0        1    313.0    313.0    313.0  __amd_rocclr_copyBuffer.kd
         0.45%      308.3       66      4.7      3.4      7.0  __amd_rocclr_fillBuffer.kd
         0.35%      241.1       60      4.0      3.0      6.7  gap <10us
         0.30%      206.9       99      2.1      1.8      2.9
         0.07%       51.2        4     12.8     10.4     17.9  gap 10us-20us

Resource=DATA Showing 1/1 records    0.01% busy
      Total(%)    Time(us)    Calls  Avg(us)  Min(us)  Max(us)  Name
         0.01%        7.4        1      7.4      7.4      7.4  HostToDevice_16384_bytes
```
