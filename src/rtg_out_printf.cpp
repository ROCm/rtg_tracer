/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <sys/types.h>
#include <unistd.h>
#include <cxxabi.h>

#include <hip/hip_runtime_api.h>
#include <hip/amd_detail/hip_runtime_prof.h>
#include "missing_ostream_definitions.h"
#define HIP_PROF_HIP_API_STRING 1 // to enable hipApiString in hip_prof_str.h
#include <roctracer/hip_ostream_ops.h>
#include <hip/amd_detail/hip_prof_str.h>

#include "rtg_out_printf.h"

constexpr std::size_t BUF_SIZE = 4096;
constexpr std::size_t OUT_SIZE = 10240;

static inline std::string get_tid_string() {
    std::ostringstream tid_os;
    tid_os << std::this_thread::get_id();
    return tid_os.str();
}

static inline const char * tid() {
    thread_local std::string tid_ = get_tid_string();
    return tid_.c_str();
}

static std::string cpp_demangle(const std::string &symname) {
    std::string retval;
    size_t size = 0;
    int status = 0;
    char* result = abi::__cxa_demangle(symname.c_str(), NULL, &size, &status);
    if (result) {
        // caller of __cxa_demangle must free returned buffer
        retval = result;
        free(result);
        return retval;
    }
    else {
        // demangle failed?
        return symname;
    }
}

namespace {

struct TlsData {
    std::vector<const char *> out;
    FILE *stream;

    static std::mutex the_mutex;
    static std::unordered_map<std::thread::id, TlsData*> the_map;

    static TlsData* Get(FILE *stream) {
        thread_local TlsData data(stream);
        return &data;
    }

    TlsData(FILE *stream) : stream(stream) {
        out.reserve(OUT_SIZE);
        std::lock_guard<std::mutex> lock(the_mutex);
        the_map[std::this_thread::get_id()] = this;
    }

    ~TlsData() {
        flush();
    }

    void push(const char *line) {
        out.push_back(line);
        if (out.size() >= OUT_SIZE) {
            flush();
        }
    }

    void flush() {
        std::lock_guard<std::mutex> lock(the_mutex);
        for (size_t i=0; i<out.size(); ++i) {
            fprintf(stream, "%s", out[i]);
            delete [] out[i];
        }
        out.clear();
        out.reserve(OUT_SIZE);
    }

    static void flush_all(FILE *stream) {
        // not precisely thread-safe, but I don't want to use a recursive mutex here
        // another thread could be manipulating the_map, but probably not
        for (auto& keyval : the_map) {
            keyval.second->flush();
        }
    }
};

std::mutex TlsData::the_mutex;
std::unordered_map<std::thread::id, TlsData*> TlsData::the_map;

}

static void check(int ret)
{
    if (ret < 0 || size_t(ret) >= BUF_SIZE) {
        fprintf(stderr, "insufficient buffer size for RtgOutPrintf\n");
        exit(EXIT_FAILURE);
    }
}

void RtgOutPrintf::open(const string& filename)
{
    pid = getpid();
    stream = fopen(filename.c_str(), "w");

    hip_api_names.reserve(HIP_API_ID_NUMBER);
    for (int i=0; i<HIP_API_ID_NUMBER; ++i) {
        hip_api_names.push_back(hip_api_name(i));
    }
}

void RtgOutPrintf::hsa_api(const string& func, const string& args, lu tick, lu ticks, int localStatus)
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "HSA: pid:%d tid:%s %s %s ret=%d @%lu +%lu\n", pid, tid(), func.c_str(), args.c_str(), localStatus, tick, ticks));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::hsa_api(const string& func, const string& args, lu tick, lu ticks, uint64_t localStatus)
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "HSA: pid:%d tid:%s %s %s ret=%lu @%lu +%lu\n", pid, tid(), func.c_str(), args.c_str(), localStatus, tick, ticks));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::hsa_api(const string& func, const string& args, lu tick, lu ticks)
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "HSA: pid:%d tid:%s %s %s ret=void @%lu +%lu\n", pid, tid(), func.c_str(), args.c_str(), tick, ticks));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::hsa_host_dispatch_kernel(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, const string& name, const hsa_kernel_dispatch_packet_t *packet, bool demangle)
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "HSA: pid:%d tid:%s dispatch queue:%lu agent:%lu signal:%lu name:'%s' tick:%lu id:%lu workgroup:{%d,%d,%d} grid:{%d,%d,%d}\n",
            pid, tid(), queue->id, agent.handle, signal.handle, demangle ? cpp_demangle(name).c_str() : name.c_str(), tick, id,
            packet->workgroup_size_x, packet->workgroup_size_y, packet->workgroup_size_z,
            packet->grid_size_x, packet->grid_size_y, packet->grid_size_z));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::hsa_host_dispatch_barrier(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, lu dep[5], const hsa_barrier_and_packet_t *packet)
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "HSA: pid:%d tid:%s barrier queue:%lu agent:%lu signal:%lu dep1:%lu dep2:%lu dep3:%lu dep4:%lu dep5:%lu tick:%lu id:%lu\n",
            pid, tid(), queue->id, agent.handle, signal.handle, dep[0], dep[1], dep[2], dep[3], dep[4], tick, id));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::hsa_host_dispatch_vendor(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, lu dep, const hsa_amd_barrier_value_packet_t *packet)
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "HSA: pid:%d tid:%s vendor queue:%lu agent:%lu signal:%lu dep:%lu tick:%lu id:%lu\n",
            pid, tid(), queue->id, agent.handle, signal.handle, dep, tick, id));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::hsa_dispatch_kernel(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, const string& name, uint64_t correlation_id, bool demangle)
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "HSA: pid:%d tid:%s dispatch queue:%lu agent:%lu signal:%lu name:'%s' start:%lu stop:%lu id:%lu\n",
            pid, tid(), queue->id, agent.handle, signal.handle, demangle ? cpp_demangle(name).c_str() : name.c_str(), start, stop, id));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::hsa_dispatch_barrier(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, lu dep[5])
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "HSA: pid:%d tid:%s barrier queue:%lu agent:%lu signal:%lu start:%lu stop:%lu dep1:%lu dep2:%lu dep3:%lu dep4:%lu dep5:%lu id:%lu\n",
            pid, tid(), queue->id, agent.handle, signal.handle, start, stop, dep[0], dep[1], dep[2], dep[3], dep[4], id));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::hsa_dispatch_vendor(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, lu dep)
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "HSA: pid:%d tid:%s vendor queue:%lu agent:%lu signal:%lu start:%lu stop:%lu dep:%lu id:%lu\n",
            pid, tid(), queue->id, agent.handle, signal.handle, start, stop, dep, id));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::hsa_dispatch_copy(hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu dep[5])
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "HSA: pid:%d tid:%s copy agent:%lu signal:%lu start:%lu stop:%lu dep1:%lu dep2:%lu dep3:%lu dep4:%lu dep5:%lu\n",
            pid, tid(), agent.handle, signal.handle, start, stop, dep[0], dep[1], dep[2], dep[3], dep[4]));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::hip_api(uint32_t cid, struct hip_api_data_s *data, int status, lu tick, lu ticks, const char *kernname, bool args, bool demangle)
{
    char *buf = new char[BUF_SIZE];
    std::string msg;

    if (args) {
        // hipApiString returns strdup, need to free, but signature returns const
        const char* args = hipApiString((hip_api_id_t)cid, data);
        msg = args;
        free((char*)args);
    }
    else {
        msg = hip_api_names[cid];
    }

    if (NULL == kernname) {
        check(snprintf(buf, BUF_SIZE, "HIP: pid:%d tid:%s %s ret=%d @%lu +%lu\n", pid, tid(), msg.c_str(), status, tick, ticks));
    }
    else {
        check(snprintf(buf, BUF_SIZE, "HIP: pid:%d tid:%s %s [%s] ret=%d @%lu +%lu\n", pid, tid(), msg.c_str(), demangle ? cpp_demangle(kernname).c_str() : kernname, status, tick, ticks));
    }

    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::roctx(uint64_t correlation_id, const string& message, lu tick, lu ticks)
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "RTX: pid:%d tid:%s %s @%lu +%lu\n", pid, tid(), message.c_str(), tick, ticks));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::roctx_mark(uint64_t correlation_id, const string& message, lu tick)
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "RTX: pid:%d tid:%s %s @%lu\n", pid, tid(), message.c_str(), tick));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::close()
{
    TlsData::flush_all(stream);
    fclose(stream);
}

