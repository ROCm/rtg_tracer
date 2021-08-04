#include <atomic>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "rtg_out_rpd.h"

#include <sqlite3.h>

#include "hsa_rsrc_factory.h"
#include "Table.h"
#include "ApiIdList.h"

typedef uint64_t timestamp_t;

#include <unistd.h>
#include <sys/syscall.h>   /* For SYS_xxx definitions */

#include <cxxabi.h>

static inline uint32_t GetPid() { return syscall(__NR_getpid); }
static inline uint32_t GetTid() { return syscall(__NR_gettid); }

// C++ symbol demangle
static inline const char* cxx_demangle(const char* symbol) {
  size_t funcnamesize;
  int status;
  const char* ret = (symbol != NULL) ? abi::__cxa_demangle(symbol, NULL, &funcnamesize, &status) : symbol;
  return (ret != NULL) ? ret : symbol;
}

const sqlite_int64 EMPTY_STRING_ID = 1;

void RtgOutRpd::open(string filename)
{
    s_metadataTable = new MetadataTable(filename.c_str());
    s_stringTable = new StringTable(filename.c_str());
    s_opTable = new OpTable(filename.c_str());
    s_apiTable = new ApiTable(filename.c_str());

    // Offset primary keys so they do not collide between sessions
    sqlite3_int64 offset = s_metadataTable->sessionId() * (sqlite3_int64(1) << 32);
    s_metadataTable->setIdOffset(offset);
    s_stringTable->setIdOffset(offset);
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

void RtgOutRpd::hsa_api(int pid, string tid, string func, string args, lu tick, lu ticks, int localStatus)
{
    fprintf(stderr, "RtgOutRpd::hsa_api NOT IMPLEMENTED\n");
    exit(EXIT_FAILURE);
}

void RtgOutRpd::hsa_api(int pid, string tid, string func, string args, lu tick, lu ticks, uint64_t localStatus)
{
    fprintf(stderr, "RtgOutRpd::hsa_api NOT IMPLEMENTED\n");
    exit(EXIT_FAILURE);
}

void RtgOutRpd::hsa_api(int pid, string tid, string func, string args, lu tick, lu ticks)
{
    fprintf(stderr, "RtgOutRpd::hsa_api NOT IMPLEMENTED\n");
    exit(EXIT_FAILURE);
}

void RtgOutRpd::hsa_host_dispatch_kernel(int pid, string tid, hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, string name, const hsa_kernel_dispatch_packet_t *packet)
{
    fprintf(stderr, "RtgOutRpd::hsa_host_dispatch_kernel NOT IMPLEMENTED\n");
    exit(EXIT_FAILURE);
}

void RtgOutRpd::hsa_host_dispatch_barrier(int pid, string tid, hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, lu dep[5], const hsa_barrier_and_packet_t *packet)
{
    fprintf(stderr, "RtgOutRpd::hsa_host_dispatch_barrier NOT IMPLEMENTED\n");
    exit(EXIT_FAILURE);
}

static std::atomic<int> counter;

void RtgOutRpd::hsa_dispatch_kernel(int pid, string tid, hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, string name, uint64_t correlation_id)
{
    OpTable::row row;
    row.gpuId = agent.handle; // TODO
    row.queueId = queue->id; // TODO
    row.sequenceId = counter++;
    //row.completionSignal = "";	//strcpy
    strncpy(row.completionSignal, "", 18);
    row.start = start;
    row.end = start+stop;
    //row.description_id = EMPTY_STRING_ID;
    row.description_id = s_stringTable->getOrCreate(name.c_str());
    row.opType_id = s_stringTable->getOrCreate("KernelExecution");
    row.api_id = correlation_id;
    s_opTable->insert(row);

    //fprintf(stderr, "RtgOutRpd::hsa_dispatch_kernel NOT IMPLEMENTED\n");
    //exit(EXIT_FAILURE);
}

void RtgOutRpd::hsa_dispatch_barrier(int pid, string tid, hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, lu dep[5])
{
    OpTable::row row;
    row.gpuId = agent.handle; // TODO
    row.queueId = queue->id; // TODO
    row.sequenceId = counter++;
    //row.completionSignal = "";	//strcpy
    strncpy(row.completionSignal, "", 18);
    row.start = start;
    row.end = start+stop;
    //row.description_id = EMPTY_STRING_ID;
    row.description_id = s_stringTable->getOrCreate("Barrier");
    row.opType_id = s_stringTable->getOrCreate("Barrier");
    row.api_id = 0;
    //row.api_id = row.sequenceId;
    s_opTable->insert(row);

    //fprintf(stderr, "RtgOutRpd::hsa_dispatch_barrier NOT IMPLEMENTED\n");
    //exit(EXIT_FAILURE);
}

void RtgOutRpd::hsa_dispatch_copy(int pid, string tid, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu dep[5])
{
    //fprintf(stderr, "RtgOutRpd::hsa_dispatch_copy NOT IMPLEMENTED\n");
    //exit(EXIT_FAILURE);
}

void RtgOutRpd::hip_api(int pid, string tid, string func_andor_args, int status, lu tick, lu ticks, uint64_t correlation_id)
{
    string func;
    string args;
    auto pos = func_andor_args.find("(");
    bool has_args = (pos != string::npos);
    if (has_args) {
        func = func_andor_args.substr(0, pos);
        args = func_andor_args.substr(pos+1);
        args = args.substr(0, args.size()-1); // remove trailing ')'
    }
    else {
        func = func_andor_args;
    }

    ApiTable::row row;
    row.pid = GetPid();
    row.tid = GetTid();
    row.start = tick;
    row.end = tick+ticks;
    row.apiName_id = s_stringTable->getOrCreate(func.c_str());
    row.args_id = EMPTY_STRING_ID;
    row.phase = 0;
    if (has_args) {
        row.args_id = s_stringTable->getOrCreate(args.c_str());
    }
    row.api_id = correlation_id;

    s_apiTable->insert(row);
    //row.phase = 1;
    //s_apiTable->insert(row);
}

void RtgOutRpd::hip_api_kernel(int pid, string tid, string func_andor_args, string kernname, int status, lu tick, lu ticks, uint64_t correlation_id)
{
    hip_api(pid, tid, func_andor_args, status, tick, ticks, correlation_id);
}

void RtgOutRpd::roctx(int pid, string tid, uint64_t correlation_id, string message, lu tick, lu ticks)
{
    ApiTable::row row;
    row.pid = GetPid();
    row.tid = GetTid();
    row.start = tick;
    row.end = tick+ticks;
    row.apiName_id = s_stringTable->getOrCreate(std::string("UserMarker"));   // FIXME: can cache
    row.args_id = s_stringTable->getOrCreate(message.c_str());
    row.phase = 0;
    row.api_id = correlation_id;
    s_apiTable->insert(row);
    //row.phase = 1;
    //s_apiTable->insert(row);
}

void RtgOutRpd::roctx_mark(int pid, string tid, uint64_t correlation_id, string message, lu tick)
{
    roctx(pid, tid, correlation_id, message, tick, tick+1);
}

void RtgOutRpd::close()
{
    // Flush recorders
    const timestamp_t begin_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    s_stringTable->finalize();
    s_opTable->finalize();
    s_apiTable->finalize();
    const timestamp_t end_time = util::HsaTimer::clocktime_ns(util::HsaTimer::TIME_ID_CLOCK_MONOTONIC);
    printf("rpd_tracer: finalized in %f ms\n", 1.0 * (end_time - begin_time) / 1000000);
}

