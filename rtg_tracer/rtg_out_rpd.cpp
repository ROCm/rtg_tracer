/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#include <atomic>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <unistd.h>
#include <cxxabi.h>

#include <sqlite3.h>
#include <fmt/format.h>

#include "hsa_rsrc_factory.h"
#include "Table.h"
#include "ApiIdList.h"

#include "rtg_out_rpd.h"

typedef uint64_t timestamp_t;

typedef sqlite_int64 tid_t;

static inline tid_t get_tid_value() {
    tid_t val;
    std::ostringstream oss;
    oss << std::this_thread::get_id();
    std::istringstream iss(oss.str());
    iss >> val;
    return val;
}

static inline tid_t tid() {
    thread_local tid_t tid_ = get_tid_value();
    return tid_;
}

// C++ symbol demangle
static inline const char* cxx_demangle(const char* symbol) {
  size_t funcnamesize;
  int status;
  const char* ret = (symbol != NULL) ? abi::__cxa_demangle(symbol, NULL, &funcnamesize, &status) : symbol;
  return (ret != NULL) ? ret : symbol;
}

const sqlite_int64 EMPTY_STRING_ID = 1;

static int activeCount = 0;
static std::mutex activeMutex;

void RtgOutRpd::rpdstart()
{
    std::unique_lock<std::mutex> lock(activeMutex);
    if (activeCount == 0) {
        //fprintf(stderr, "rpd_tracer: START\n");
        s_apiTable->resumeRoctx(util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC));
        //start_tracing();
    }
    ++activeCount;
}

void RtgOutRpd::rpdstop()
{
    std::unique_lock<std::mutex> lock(activeMutex);
    if (activeCount == 1) {
        //fprintf(stderr, "rpd_tracer: STOP\n");
        //stop_tracing();
        s_apiTable->suspendRoctx(util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC));
    }
    --activeCount;
}

void RtgOutRpd::open(const string& filename)
{
    pid = getpid();

    s_metadataTable = new MetadataTable(filename.c_str());
    s_stringTable = new StringTable(filename.c_str());
    s_kernelApiTable = new KernelApiTable(filename.c_str());
    s_copyApiTable = new CopyApiTable(filename.c_str());
    s_opTable = new OpTable(filename.c_str());
    s_apiTable = new ApiTable(filename.c_str());

    // Offset primary keys so they do not collide between sessions
    sqlite3_int64 offset = s_metadataTable->sessionId() * (sqlite3_int64(1) << 32);
    s_metadataTable->setIdOffset(offset);
    s_stringTable->setIdOffset(offset);
    s_kernelApiTable->setIdOffset(offset);
    s_copyApiTable->setIdOffset(offset);
    s_opTable->setIdOffset(offset);
    s_apiTable->setIdOffset(offset);

    // Pick some apis to ignore
    s_apiList = new ApiIdList();
    s_apiList->setInvertMode(true);  // Omit the specified api
    s_apiList->add("hipGetDevice");
    s_apiList->add("hipSetDevice");
    s_apiList->add("hipGetLastError");
    s_apiList->add("__hipPushCallConfiguration");
    s_apiList->add("__hipPopCallConfiguration");
    s_apiList->add("hipCtxSetCurrent");
    s_apiList->add("hipEventRecord");
    s_apiList->add("hipEventQuery");
    s_apiList->add("hipGetDeviceProperties");
    s_apiList->add("hipPeekAtLastError");
    s_apiList->add("hipModuleGetFunction");
    s_apiList->add("hipEventCreateWithFlags");
}

void RtgOutRpd::hsa_api(const string& func, const string& args, lu tick, lu ticks, int localStatus)
{
    fprintf(stderr, "RtgOutRpd::hsa_api NOT IMPLEMENTED\n");
    exit(EXIT_FAILURE);
}

void RtgOutRpd::hsa_api(const string& func, const string& args, lu tick, lu ticks, uint64_t localStatus)
{
    fprintf(stderr, "RtgOutRpd::hsa_api NOT IMPLEMENTED\n");
    exit(EXIT_FAILURE);
}

void RtgOutRpd::hsa_api(const string& func, const string& args, lu tick, lu ticks)
{
    fprintf(stderr, "RtgOutRpd::hsa_api NOT IMPLEMENTED\n");
    exit(EXIT_FAILURE);
}

void RtgOutRpd::hsa_host_dispatch_kernel(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, const string& name, const hsa_kernel_dispatch_packet_t *packet)
{
    fprintf(stderr, "RtgOutRpd::hsa_host_dispatch_kernel NOT IMPLEMENTED\n");
    exit(EXIT_FAILURE);
}

void RtgOutRpd::hsa_host_dispatch_barrier(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, lu dep[5], const hsa_barrier_and_packet_t *packet)
{
    fprintf(stderr, "RtgOutRpd::hsa_host_dispatch_barrier NOT IMPLEMENTED\n");
    exit(EXIT_FAILURE);
}

static std::atomic<int> counter;

void RtgOutRpd::hsa_dispatch_kernel(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, const string& name, uint64_t correlation_id, bool demangle)
{
    OpTable::row row;
    row.gpuId = agent.handle; // TODO
    row.queueId = queue->id; // TODO
    row.sequenceId = counter++;
    //row.completionSignal = "";	//strcpy
    strncpy(row.completionSignal, "", 18);
    row.start = start;
    row.end = stop;
    //row.description_id = EMPTY_STRING_ID;
    row.description_id = s_stringTable->getOrCreate(cxx_demangle(name.c_str()));
    row.opType_id = s_stringTable->getOrCreate("KernelExecution");
    row.api_id = correlation_id;
    s_opTable->insert(row);

    //fprintf(stderr, "RtgOutRpd::hsa_dispatch_kernel NOT IMPLEMENTED\n");
    //exit(EXIT_FAILURE);
}

void RtgOutRpd::hsa_dispatch_barrier(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, lu dep[5])
{
    OpTable::row row;
    row.gpuId = agent.handle; // TODO
    row.queueId = queue->id; // TODO
    row.sequenceId = counter++;
    //row.completionSignal = "";	//strcpy
    strncpy(row.completionSignal, "", 18);
    row.start = start;
    row.end = stop;
    //row.description_id = EMPTY_STRING_ID;
    row.description_id = s_stringTable->getOrCreate("Barrier");
    row.opType_id = s_stringTable->getOrCreate("Barrier");
    // barriers do not current associate with HIP API calls, so set this to something that should never match a correlation_id
    row.api_id = std::numeric_limits<sqlite_int64>::max();
    //row.api_id = row.sequenceId;
    s_opTable->insert(row);

    //fprintf(stderr, "RtgOutRpd::hsa_dispatch_barrier NOT IMPLEMENTED\n");
    //exit(EXIT_FAILURE);
}

void RtgOutRpd::hsa_dispatch_copy(hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu dep[5])
{
    //fprintf(stderr, "RtgOutRpd::hsa_dispatch_copy NOT IMPLEMENTED\n");
    //exit(EXIT_FAILURE);
}

void RtgOutRpd::hip_api(uint32_t cid, struct hip_api_data_s *data, int status, lu tick, lu ticks, const char *kernname, bool args, bool demangle)
{
    ApiTable::row row;
    row.pid = pid;
    row.tid = tid();
    row.start = tick;
    row.end = tick+ticks;
    row.apiName_id = s_stringTable->getOrCreate(hip_api_name(cid));
    row.args_id = EMPTY_STRING_ID;
    row.phase = 0;
    row.api_id = data->correlation_id;
    // ApiTable expects two inserts, one for each phase
    s_apiTable->insert(row);

    char buff[4096];
    switch (cid) {
        case HIP_API_ID_hipMalloc:
            std::snprintf(buff, 4096, "size=0x%x",
                (uint32_t)(data->args.hipMalloc.size));
            row.args_id = s_stringTable->getOrCreate(std::string(buff)); 
            break;
        case HIP_API_ID_hipFree:
            std::snprintf(buff, 4096, "ptr=%p",
                data->args.hipFree.ptr);
            row.args_id = s_stringTable->getOrCreate(std::string(buff)); 
            break;

        case HIP_API_ID_hipLaunchCooperativeKernelMultiDevice:
            {
                const hipLaunchParams &params = data->args.hipLaunchCooperativeKernelMultiDevice.launchParamsList__val;
                std::string kernelName = cxx_demangle(hipKernelNameRefByPtr(params.func, params.stream));
                //std::snprintf(buff, 4096, "stream=%p | kernel=%s",
                //    params.stream,
                //    kernelName.c_str());
                //row.args_id = s_stringTable->getOrCreate(std::string(buff));

                KernelApiTable::row krow;
                krow.api_id = row.api_id;
                krow.stream = fmt::format("{}", (void*)params.stream);
                krow.gridX = params.gridDim.x;
                krow.gridY = params.gridDim.y;
                krow.gridZ = params.gridDim.z;
                krow.workgroupX = params.blockDim.x;
                krow.workgroupY = params.blockDim.y;
                krow.workgroupZ = params.blockDim.z;
                krow.groupSegmentSize = params.sharedMem;
                krow.privateSegmentSize = 0;
                krow.kernelName_id = s_stringTable->getOrCreate(kernelName);

                s_kernelApiTable->insert(krow);

                // Associate kernel name with op
                s_opTable->associateDescription(row.api_id, krow.kernelName_id);
            }
            break;

        case HIP_API_ID_hipExtLaunchMultiKernelMultiDevice:
            {
                const hipLaunchParams &params = data->args.hipExtLaunchMultiKernelMultiDevice.launchParamsList__val;
                std::string kernelName = cxx_demangle(hipKernelNameRefByPtr(params.func, params.stream));
                //std::snprintf(buff, 4096, "stream=%p | kernel=%s",
                //    params.stream,
                //    kernelName.c_str());
                //row.args_id = s_stringTable->getOrCreate(std::string(buff));

                KernelApiTable::row krow;
                krow.api_id = row.api_id;
                krow.stream = fmt::format("{}", (void*)params.stream);
                krow.gridX = params.gridDim.x;
                krow.gridY = params.gridDim.y;
                krow.gridZ = params.gridDim.z;
                krow.workgroupX = params.blockDim.x;
                krow.workgroupY = params.blockDim.y;
                krow.workgroupZ = params.blockDim.z;
                krow.groupSegmentSize = params.sharedMem;
                krow.privateSegmentSize = 0;
                krow.kernelName_id = s_stringTable->getOrCreate(kernelName);

                s_kernelApiTable->insert(krow);

                // Associate kernel name with op
                s_opTable->associateDescription(row.api_id, krow.kernelName_id);
            }
            break;

        case HIP_API_ID_hipLaunchKernel:
            {
                auto &params = data->args.hipLaunchKernel;
                std::string kernelName = cxx_demangle(hipKernelNameRefByPtr(params.function_address, params.stream));
                //std::snprintf(buff, 4096, "stream=%p | kernel=%s",
                //    params.stream,
                //    kernelName.c_str());
                //row.args_id = s_stringTable->getOrCreate(std::string(buff));

                KernelApiTable::row krow;
                krow.api_id = row.api_id;
                krow.stream = fmt::format("{}", (void*)params.stream);
                krow.gridX = params.numBlocks.x;
                krow.gridY = params.numBlocks.y;
                krow.gridZ = params.numBlocks.z;
                krow.workgroupX = params.dimBlocks.x;
                krow.workgroupY = params.dimBlocks.y;
                krow.workgroupZ = params.dimBlocks.z;
                krow.groupSegmentSize = params.sharedMemBytes;
                krow.privateSegmentSize = 0;
                krow.kernelName_id = s_stringTable->getOrCreate(kernelName);

                s_kernelApiTable->insert(krow);

                // Associate kernel name with op
                s_opTable->associateDescription(row.api_id, krow.kernelName_id);
            }
            break;

        case HIP_API_ID_hipExtLaunchKernel:
            {
                auto &params = data->args.hipExtLaunchKernel;
                std::string kernelName = cxx_demangle(hipKernelNameRefByPtr(params.function_address, params.stream));
                //std::snprintf(buff, 4096, "stream=%p | kernel=%s",
                //    params.stream,
                //    kernelName.c_str());
                //row.args_id = s_stringTable->getOrCreate(std::string(buff));

                KernelApiTable::row krow;
                krow.api_id = row.api_id;
                krow.stream = fmt::format("{}", (void*)params.stream);
                krow.gridX = params.numBlocks.x;
                krow.gridY = params.numBlocks.y;
                krow.gridZ = params.numBlocks.z;
                krow.workgroupX = params.dimBlocks.x;
                krow.workgroupY = params.dimBlocks.y;
                krow.workgroupZ = params.dimBlocks.z;
                krow.groupSegmentSize = params.sharedMemBytes;
                krow.privateSegmentSize = 0;
                krow.kernelName_id = s_stringTable->getOrCreate(kernelName);

                s_kernelApiTable->insert(krow);

                // Associate kernel name with op
                s_opTable->associateDescription(row.api_id, krow.kernelName_id);
            }
            break;

        case HIP_API_ID_hipLaunchCooperativeKernel:
            {
                auto &params = data->args.hipLaunchCooperativeKernel;
                std::string kernelName = cxx_demangle(hipKernelNameRefByPtr(params.f, params.stream));
                //std::snprintf(buff, 4096, "stream=%p | kernel=%s",
                //    params.stream,
                //    kernelName.c_str());
                //row.args_id = s_stringTable->getOrCreate(std::string(buff));

                KernelApiTable::row krow;
                krow.api_id = row.api_id;
                krow.stream = fmt::format("{}", (void*)params.stream);
                krow.gridX = params.gridDim.x;
                krow.gridY = params.gridDim.y;
                krow.gridZ = params.gridDim.z;
                krow.workgroupX = params.blockDimX.x;
                krow.workgroupY = params.blockDimX.y;
                krow.workgroupZ = params.blockDimX.z;
                krow.groupSegmentSize = params.sharedMemBytes;
                krow.privateSegmentSize = 0;
                krow.kernelName_id = s_stringTable->getOrCreate(kernelName);

                s_kernelApiTable->insert(krow);

                // Associate kernel name with op
                s_opTable->associateDescription(row.api_id, krow.kernelName_id);
            }
            break;

        case HIP_API_ID_hipHccModuleLaunchKernel:
            {
                auto &params = data->args.hipHccModuleLaunchKernel;
                std::string kernelName(cxx_demangle(hipKernelNameRef(params.f)));
                //std::snprintf(buff, 4096, "stream=%p | kernel=%s",
                //    params.stream,
                //    kernelName.c_str());
                //row.args_id = s_stringTable->getOrCreate(std::string(buff));

                KernelApiTable::row krow;
                krow.api_id = row.api_id;
                krow.stream = fmt::format("{}", (void*)params.hStream);
                krow.gridX = params.globalWorkSizeX;
                krow.gridY = params.globalWorkSizeY;
                krow.gridZ = params.globalWorkSizeZ;
                krow.workgroupX = params.blockDimX;
                krow.workgroupY = params.blockDimY;
                krow.workgroupZ = params.blockDimZ;
                krow.groupSegmentSize = params.sharedMemBytes;
                krow.privateSegmentSize = 0;
                krow.kernelName_id = s_stringTable->getOrCreate(kernelName);

                s_kernelApiTable->insert(krow);

                // Associate kernel name with op
                s_opTable->associateDescription(row.api_id, krow.kernelName_id);
            }
            break;

        case HIP_API_ID_hipModuleLaunchKernel:
            {
                auto &params = data->args.hipModuleLaunchKernel;
                std::string kernelName(cxx_demangle(hipKernelNameRef(params.f)));
                //std::snprintf(buff, 4096, "stream=%p | kernel=%s",
                //    params.stream,
                //    kernelName.c_str());
                //row.args_id = s_stringTable->getOrCreate(std::string(buff));

                KernelApiTable::row krow;
                krow.api_id = row.api_id;
                krow.stream = fmt::format("{}", (void*)params.stream);
                krow.gridX = params.gridDimX;
                krow.gridY = params.gridDimY;
                krow.gridZ = params.gridDimZ;
                krow.workgroupX = params.blockDimX;
                krow.workgroupY = params.blockDimY;
                krow.workgroupZ = params.blockDimZ;
                krow.groupSegmentSize = params.sharedMemBytes;
                krow.privateSegmentSize = 0;
                krow.kernelName_id = s_stringTable->getOrCreate(kernelName);

                s_kernelApiTable->insert(krow);

                // Associate kernel name with op
                s_opTable->associateDescription(row.api_id, krow.kernelName_id);
            }
            break;

        case HIP_API_ID_hipExtModuleLaunchKernel:
            {
                auto &params = data->args.hipExtModuleLaunchKernel;
                std::string kernelName(cxx_demangle(hipKernelNameRef(params.f)));
                //std::snprintf(buff, 4096, "stream=%p | kernel=%s",
                //    params.stream,
                //    kernelName.c_str());
                //row.args_id = s_stringTable->getOrCreate(std::string(buff));

                KernelApiTable::row krow;
                krow.api_id = row.api_id;
                krow.stream = fmt::format("{}", (void*)params.hStream);
                krow.gridX = params.globalWorkSizeX;
                krow.gridY = params.globalWorkSizeY;
                krow.gridZ = params.globalWorkSizeZ;
                krow.workgroupX = params.localWorkSizeX;
                krow.workgroupY = params.localWorkSizeY;
                krow.workgroupZ = params.localWorkSizeZ;
                krow.groupSegmentSize = params.sharedMemBytes;
                krow.privateSegmentSize = 0;
                krow.kernelName_id = s_stringTable->getOrCreate(kernelName);

                s_kernelApiTable->insert(krow);

                // Associate kernel name with op
                s_opTable->associateDescription(row.api_id, krow.kernelName_id);
            }
            break;

        case HIP_API_ID_hipMemcpy:
            //std::snprintf(buff, 4096, "dst=%p | src=%p | size=0x%x | kind=%u",
            //    data->args.hipMemcpy.dst,
            //    data->args.hipMemcpy.src,
            //    (uint32_t)(data->args.hipMemcpy.sizeBytes),
            //    (uint32_t)(data->args.hipMemcpy.kind));
            //row.args_id = s_stringTable->getOrCreate(std::string(buff));
            {
                CopyApiTable::row crow;
                crow.api_id = row.api_id;
                crow.size = (uint32_t)(data->args.hipMemcpy.sizeBytes);
                crow.dst = fmt::format("{}", data->args.hipMemcpy.dst);
                crow.src = fmt::format("{}", data->args.hipMemcpy.src);
                crow.kind = (uint32_t)(data->args.hipMemcpy.kind);
                crow.sync = true;
                s_copyApiTable->insert(crow);
            }
            break;
        case HIP_API_ID_hipMemcpy2D:
            //std::snprintf(buff, 4096, "dst=%p | src=%p | width=0x%x | height=0x%x | kind=%u",
            //    data->args.hipMemcpy2D.dst,
            //    data->args.hipMemcpy2D.src,
            //    (uint32_t)(data->args.hipMemcpy2D.width),
            //    (uint32_t)(data->args.hipMemcpy2D.height),
            //    (uint32_t)(data->args.hipMemcpy2D.kind));
            //row.args_id = s_stringTable->getOrCreate(std::string(buff));
            {
                CopyApiTable::row crow;
                crow.api_id = row.api_id;
                crow.width = (uint32_t)(data->args.hipMemcpy2D.width);
                crow.height = (uint32_t)(data->args.hipMemcpy2D.height);
                crow.dst = fmt::format("{}", data->args.hipMemcpy2D.dst);
                crow.src = fmt::format("{}", data->args.hipMemcpy2D.src);
                crow.kind = (uint32_t)(data->args.hipMemcpy2D.kind);
                crow.sync = true;
                s_copyApiTable->insert(crow);
            }
            break;
        case HIP_API_ID_hipMemcpy2DAsync:
            //std::snprintf(buff, 4096, "dst=%p | src=%p | width=0x%x | height=0x%x | kind=%u",
            //    data->args.hipMemcpy2DAsync.dst,
            //    data->args.hipMemcpy2DAsync.src,
            //    (uint32_t)(data->args.hipMemcpy2DAsync.width),
            //    (uint32_t)(data->args.hipMemcpy2DAsync.height),
            //    (uint32_t)(data->args.hipMemcpy2DAsync.kind));
            //row.args_id = s_stringTable->getOrCreate(std::string(buff));
            {
                CopyApiTable::row crow;
                crow.api_id = row.api_id;
                crow.stream = fmt::format("{}", (void*)data->args.hipMemcpy2DAsync.stream);
                crow.width = (uint32_t)(data->args.hipMemcpy2DAsync.width);
                crow.height = (uint32_t)(data->args.hipMemcpy2DAsync.height);
                crow.dst = fmt::format("{}", data->args.hipMemcpy2DAsync.dst);
                crow.src = fmt::format("{}", data->args.hipMemcpy2DAsync.src);
                crow.kind = (uint32_t)(data->args.hipMemcpy2DAsync.kind);
                crow.sync = false;
                s_copyApiTable->insert(crow);
            }
            break;
        case HIP_API_ID_hipMemcpyAsync:
            //std::snprintf(buff, 4096, "dst=%p | src=%p | size=0x%x | kind=%u",
            //    data->args.hipMemcpyAsync.dst,
            //    data->args.hipMemcpyAsync.src,
            //    (uint32_t)(data->args.hipMemcpyAsync.sizeBytes),
            //    (uint32_t)(data->args.hipMemcpyAsync.kind));
            //row.args_id = s_stringTable->getOrCreate(std::string(buff));
            {
                CopyApiTable::row crow;
                crow.api_id = row.api_id;
                crow.stream = fmt::format("{}", (void*)data->args.hipMemcpyAsync.stream);
                crow.size = (uint32_t)(data->args.hipMemcpyAsync.sizeBytes);
                crow.dst = fmt::format("{}", data->args.hipMemcpyAsync.dst);
                crow.src = fmt::format("{}", data->args.hipMemcpyAsync.src);
                crow.kind = (uint32_t)(data->args.hipMemcpyAsync.kind);
                crow.sync = false;
                s_copyApiTable->insert(crow);
            }
            break;
        case HIP_API_ID_hipMemcpyDtoD:
            //std::snprintf(buff, 4096, "dst=%p | src=%p | size=0x%x",
            //    data->args.hipMemcpyDtoD.dst,
            //    data->args.hipMemcpyDtoD.src,
            //    (uint32_t)(data->args.hipMemcpyDtoD.sizeBytes));
            //row.args_id = s_stringTable->getOrCreate(std::string(buff));
            {
                CopyApiTable::row crow;
                crow.api_id = row.api_id;
                crow.size = (uint32_t)(data->args.hipMemcpyDtoD.sizeBytes);
                crow.dst = fmt::format("{}", data->args.hipMemcpyDtoD.dst);
                crow.src = fmt::format("{}", data->args.hipMemcpyDtoD.src);
                crow.sync = true;
                s_copyApiTable->insert(crow);
            }

            break;
        case HIP_API_ID_hipMemcpyDtoDAsync:
            //std::snprintf(buff, 4096, "dst=%p | src=%p | size=0x%x",
            //    data->args.hipMemcpyDtoDAsync.dst,
            //    data->args.hipMemcpyDtoDAsync.src,
            //    (uint32_t)(data->args.hipMemcpyDtoDAsync.sizeBytes));
            //row.args_id = s_stringTable->getOrCreate(std::string(buff));
            {
                CopyApiTable::row crow;
                crow.api_id = row.api_id;
                crow.stream = fmt::format("{}", (void*)data->args.hipMemcpyDtoDAsync.stream);
                crow.size = (uint32_t)(data->args.hipMemcpyDtoDAsync.sizeBytes);
                crow.dst = fmt::format("{}", data->args.hipMemcpyDtoDAsync.dst);
                crow.src = fmt::format("{}", data->args.hipMemcpyDtoDAsync.src);
                crow.sync = false;
                s_copyApiTable->insert(crow);
            }

            break;
        case HIP_API_ID_hipMemcpyDtoH:
            //std::snprintf(buff, 4096, "dst=%p | src=%p | size=0x%x",
            //    data->args.hipMemcpyDtoH.dst,
            //    data->args.hipMemcpyDtoH.src,
            //    (uint32_t)(data->args.hipMemcpyDtoH.sizeBytes));
            //row.args_id = s_stringTable->getOrCreate(std::string(buff));
            {
                CopyApiTable::row crow;
                crow.api_id = row.api_id;
                crow.size = (uint32_t)(data->args.hipMemcpyDtoH.sizeBytes);
                crow.dst = fmt::format("{}", data->args.hipMemcpyDtoH.dst);
                crow.src = fmt::format("{}", data->args.hipMemcpyDtoH.src);
                crow.sync = true;
                s_copyApiTable->insert(crow);
            }
            break;
        case HIP_API_ID_hipMemcpyDtoHAsync:
            //std::snprintf(buff, 4096, "dst=%p | src=%p | size=0x%x",
            //    data->args.hipMemcpyDtoHAsync.dst,
            //    data->args.hipMemcpyDtoHAsync.src,
            //    (uint32_t)(data->args.hipMemcpyDtoHAsync.sizeBytes));
            //row.args_id = s_stringTable->getOrCreate(std::string(buff));
            {
                CopyApiTable::row crow;
                crow.api_id = row.api_id;
                crow.stream = fmt::format("{}", (void*)data->args.hipMemcpyDtoHAsync.stream);
                crow.size = (uint32_t)(data->args.hipMemcpyDtoHAsync.sizeBytes);
                crow.dst = fmt::format("{}", data->args.hipMemcpyDtoHAsync.dst);
                crow.src = fmt::format("{}", data->args.hipMemcpyDtoHAsync.src);
                crow.sync = false;
                s_copyApiTable->insert(crow);
            }
            break;
        case HIP_API_ID_hipMemcpyFromSymbol:
            //std::snprintf(buff, 4096, "dst=%p | symbol=%p | size=0x%x | kind=%u",
            //    data->args.hipMemcpyFromSymbol.dst,
            //    data->args.hipMemcpyFromSymbol.symbol,
            //    (uint32_t)(data->args.hipMemcpyFromSymbol.sizeBytes),
            //    (uint32_t)(data->args.hipMemcpyFromSymbol.kind));
            //row.args_id = s_stringTable->getOrCreate(std::string(buff));
            {
                CopyApiTable::row crow;
                crow.api_id = row.api_id;
                crow.size = (uint32_t)(data->args.hipMemcpyFromSymbol.sizeBytes);
                crow.dst = fmt::format("{}", data->args.hipMemcpyFromSymbol.dst);
                crow.src = fmt::format("{}", data->args.hipMemcpyFromSymbol.symbol);
                crow.kind = (uint32_t)(data->args.hipMemcpyFromSymbol.kind);
                crow.sync = true;
                s_copyApiTable->insert(crow);
            }
            break;
case HIP_API_ID_hipMemcpyFromSymbolAsync:
            //std::snprintf(buff, 4096, "dst=%p | symbol=%p | size=0x%x | kind=%u",
            //    data->args.hipMemcpyFromSymbolAsync.dst,
            //    data->args.hipMemcpyFromSymbolAsync.symbol,
            //    (uint32_t)(data->args.hipMemcpyFromSymbolAsync.sizeBytes),
            //    (uint32_t)(data->args.hipMemcpyFromSymbolAsync.kind));
            //row.args_id = s_stringTable->getOrCreate(std::string(buff));
            {
                CopyApiTable::row crow;
                crow.api_id = row.api_id;
                crow.stream = fmt::format("{}", (void*)data->args.hipMemcpyFromSymbolAsync.stream);
                crow.size = (uint32_t)(data->args.hipMemcpyFromSymbolAsync.sizeBytes);
                crow.dst = fmt::format("{}", data->args.hipMemcpyFromSymbolAsync.dst);
                crow.src = fmt::format("{}", data->args.hipMemcpyFromSymbolAsync.symbol);
                crow.kind = (uint32_t)(data->args.hipMemcpyFromSymbolAsync.kind);
                crow.sync = false;
                s_copyApiTable->insert(crow);
            }
            break;
        case HIP_API_ID_hipMemcpyHtoD:
            //std::snprintf(buff, 4096, "dst=%p | src=%p | size=0x%x",
            //    data->args.hipMemcpyHtoDAsync.dst,
            //    data->args.hipMemcpyHtoDAsync.src,
            //    (uint32_t)(data->args.hipMemcpyHtoDAsync.sizeBytes));
            //row.args_id = s_stringTable->getOrCreate(std::string(buff));
            {
                CopyApiTable::row crow;
                crow.api_id = row.api_id;
                crow.size = (uint32_t)(data->args.hipMemcpyHtoD.sizeBytes);
                crow.dst = fmt::format("{}", data->args.hipMemcpyHtoD.dst);
                crow.src = fmt::format("{}", data->args.hipMemcpyHtoD.src);
                crow.sync = true;
                s_copyApiTable->insert(crow);
            }
            break;
case HIP_API_ID_hipMemcpyHtoDAsync:
            //std::snprintf(buff, 4096, "dst=%p | src=%p | size=0x%x",
            //    data->args.hipMemcpyHtoDAsync.dst,
            //    data->args.hipMemcpyHtoDAsync.src,
            //    (uint32_t)(data->args.hipMemcpyHtoDAsync.sizeBytes));
            //row.args_id = s_stringTable->getOrCreate(std::string(buff));
            {
                CopyApiTable::row crow;
                crow.api_id = row.api_id;
                crow.stream = fmt::format("{}", (void*)data->args.hipMemcpyHtoDAsync.stream);
                crow.size = (uint32_t)(data->args.hipMemcpyHtoDAsync.sizeBytes);
                crow.dst = fmt::format("{}", data->args.hipMemcpyHtoDAsync.dst);
                crow.src = fmt::format("{}", data->args.hipMemcpyHtoDAsync.src);
                crow.sync = false;
                s_copyApiTable->insert(crow);
            }
            break;
        case HIP_API_ID_hipMemcpyPeer:
            //std::snprintf(buff, 4096, "dst=%p | device=%d | src=%p | device=%d | size=0x%x",
            //    data->args.hipMemcpyPeer.dst,
            //    data->args.hipMemcpyPeer.dstDeviceId,
            //    data->args.hipMemcpyPeer.src,
            //    data->args.hipMemcpyPeer.srcDeviceId,
            //    (uint32_t)(data->args.hipMemcpyPeer.sizeBytes));
            //row.args_id = s_stringTable->getOrCreate(std::string(buff));
            {
                CopyApiTable::row crow;
                crow.api_id = row.api_id;
                crow.size = (uint32_t)(data->args.hipMemcpyPeer.sizeBytes);
                crow.dst = fmt::format("{}", data->args.hipMemcpyPeer.dst);
                crow.src = fmt::format("{}", data->args.hipMemcpyPeer.src);
                crow.dstDevice = data->args.hipMemcpyPeer.dstDeviceId;
                crow.srcDevice = data->args.hipMemcpyPeer.srcDeviceId;
                crow.sync = true;
                s_copyApiTable->insert(crow);
            }
            break;
        case HIP_API_ID_hipMemcpyPeerAsync:
            //std::snprintf(buff, 4096, "dst=%p | device=%d | src=%p | device=%d | size=0x%x",
            //    data->args.hipMemcpyPeerAsync.dst,
            //    data->args.hipMemcpyPeerAsync.dstDeviceId,
            //    data->args.hipMemcpyPeerAsync.src,
            //    data->args.hipMemcpyPeerAsync.srcDevice,
            //    (uint32_t)(data->args.hipMemcpyPeerAsync.sizeBytes));
            //row.args_id = s_stringTable->getOrCreate(std::string(buff));
            {
                CopyApiTable::row crow;
                crow.api_id = row.api_id;
                crow.stream = fmt::format("{}", (void*)data->args.hipMemcpyPeerAsync.stream);
                crow.size = (uint32_t)(data->args.hipMemcpyPeerAsync.sizeBytes);
                crow.dst = fmt::format("{}", data->args.hipMemcpyPeerAsync.dst);
                crow.src = fmt::format("{}", data->args.hipMemcpyPeerAsync.src);
                crow.dstDevice = data->args.hipMemcpyPeerAsync.dstDeviceId;
                crow.srcDevice = data->args.hipMemcpyPeerAsync.srcDevice;
                crow.sync = false;
                s_copyApiTable->insert(crow);
            }
            break;
        case HIP_API_ID_hipMemcpyToSymbol:
            //std::snprintf(buff, 4096, "symbol=%p | src=%p | size=0x%x | kind=%u",
            //    data->args.hipMemcpyToSymbol.symbol,
            //    data->args.hipMemcpyToSymbol.src,
            //    (uint32_t)(data->args.hipMemcpyToSymbol.sizeBytes),
            //    (uint32_t)(data->args.hipMemcpyToSymbol.kind));
            //row.args_id = s_stringTable->getOrCreate(std::string(buff));
            {
                CopyApiTable::row crow;
                crow.api_id = row.api_id;
                crow.size = (uint32_t)(data->args.hipMemcpyToSymbol.sizeBytes);
                crow.dst = fmt::format("{}", data->args.hipMemcpyToSymbol.symbol);
                crow.src = fmt::format("{}", data->args.hipMemcpyToSymbol.src);
                crow.kind = (uint32_t)(data->args.hipMemcpyToSymbol.kind);
                crow.sync = true;
                s_copyApiTable->insert(crow);
            }
            break;
        case HIP_API_ID_hipMemcpyToSymbolAsync:
            //std::snprintf(buff, 4096, "symbol=%p | src=%p | size=0x%x | kind=%u",
            //    data->args.hipMemcpyToSymbolAsync.symbol,
            //    data->args.hipMemcpyToSymbolAsync.src,
            //    (uint32_t)(data->args.hipMemcpyToSymbolAsync.sizeBytes),
            //    (uint32_t)(data->args.hipMemcpyToSymbolAsync.kind));
            //row.args_id = s_stringTable->getOrCreate(std::string(buff));
            {
                CopyApiTable::row crow;
                crow.api_id = row.api_id;
                crow.stream = fmt::format("{}", (void*)data->args.hipMemcpyToSymbolAsync.stream);
                crow.size = (uint32_t)(data->args.hipMemcpyToSymbolAsync.sizeBytes);
                crow.dst = fmt::format("{}", data->args.hipMemcpyToSymbolAsync.symbol);
                crow.src = fmt::format("{}", data->args.hipMemcpyToSymbolAsync.src);
                crow.kind = (uint32_t)(data->args.hipMemcpyToSymbolAsync.kind);
                crow.sync = false;
                s_copyApiTable->insert(crow);
            }
            break;
        case HIP_API_ID_hipMemcpyWithStream:
            //std::snprintf(buff, 4096, "dst=%p | src=%p | size=0x%x | kind=%u", 
            //    data->args.hipMemcpyWithStream.dst,
            //    data->args.hipMemcpyWithStream.src,
            //    (uint32_t)(data->args.hipMemcpyWithStream.sizeBytes),
            //    (uint32_t)(data->args.hipMemcpyWithStream.kind));
            //row.args_id = s_stringTable->getOrCreate(std::string(buff)); 
            {
                CopyApiTable::row crow;
                crow.api_id = row.api_id;
                crow.stream = fmt::format("{}", (void*)data->args.hipMemcpyWithStream.stream);
                crow.size = (uint32_t)(data->args.hipMemcpyWithStream.sizeBytes);
                crow.dst = fmt::format("{}", data->args.hipMemcpyWithStream.dst);
                crow.src = fmt::format("{}", data->args.hipMemcpyWithStream.src);
                crow.kind = (uint32_t)(data->args.hipMemcpyWithStream.kind);
                crow.sync = false;
                s_copyApiTable->insert(crow);
            }
            break;
        default:
            break;
    }
    row.phase = 1;
    s_apiTable->insert(row);
}

void RtgOutRpd::roctx(uint64_t correlation_id, const string& message, lu tick, lu ticks)
{
    ApiTable::row row;
    row.pid = pid;
    row.tid = tid();
    row.start = tick;
    row.end = tick+ticks;
    row.apiName_id = s_stringTable->getOrCreate(std::string("UserMarker"));   // FIXME: can cache
    row.args_id = s_stringTable->getOrCreate(message.c_str());
    row.phase = 0;
    row.api_id = correlation_id;
    s_apiTable->insert(row);
    // ApiTable expects two inserts, one for each phase
    row.phase = 1;
    s_apiTable->insert(row);
}

void RtgOutRpd::roctx_mark(uint64_t correlation_id, const string& message, lu tick)
{
    roctx(correlation_id, message, tick, 1);
}

void RtgOutRpd::close()
{
    // Flush recorders
    const timestamp_t begin_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    s_metadataTable->finalize();
    s_stringTable->finalize();
    s_kernelApiTable->finalize();
    s_copyApiTable->finalize();
    s_opTable->finalize();
    s_apiTable->finalize();
    const timestamp_t end_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    printf("rpd_tracer: finalized in %f ms\n", 1.0 * (end_time - begin_time) / 1000000);
}

