#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "rtg_out_printf_lockless.h"

static inline std::string get_tid_string() {
    std::ostringstream tid_os;
    tid_os << std::this_thread::get_id();
    return tid_os.str();
}

static inline const char * tid() {
    thread_local std::string tid_ = get_tid_string();
    return tid_.c_str();
}

namespace {

struct TlsData {
    int pid;
    string tid_string;
    const char *tid;
    FILE *stream;

    static std::mutex the_class_mutex;
    static std::unordered_map<std::thread::id, TlsData*> the_map;

    static TlsData* Get(const std::string &filename) {
        thread_local TlsData data(filename);
        return &data;
    }

    TlsData(const std::string &filename) {
        pid = getpid();
        tid_string = get_tid_string();
        tid = tid_string.c_str();

        std::string filename_with_tid = filename + "." + tid_string;
        stream = fopen(filename_with_tid.c_str(), "w");

        std::lock_guard<std::mutex> lock(the_class_mutex);
        the_map[std::this_thread::get_id()] = this;
    }

    ~TlsData() {
        fclose(stream);
    }

    void flush() {
        fflush(stream);
    }

    void close() {
        fclose(stream);
    }

    static void flush_all() {
        // not precisely thread-safe, but I don't want to use a recursive mutex here
        // another thread could be manipulating the_map, but probably not
        for (auto& keyval : the_map) {
            keyval.second->flush();
        }
    }

    void hsa_api(const string& func, const string& args, lu tick, lu ticks, int localStatus);
    void hsa_api(const string& func, const string& args, lu tick, lu ticks, uint64_t localStatus);
    void hsa_api(const string& func, const string& args, lu tick, lu ticks);
    void hsa_host_dispatch_kernel (hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, const string& name, const hsa_kernel_dispatch_packet_t *packet);
    void hsa_host_dispatch_barrier(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, lu dep[5], const hsa_barrier_and_packet_t *packet);
    void hsa_dispatch_kernel (hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, const string& name, uint64_t correlation_id);
    void hsa_dispatch_barrier(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, lu dep[5]);
    void hsa_dispatch_copy   (hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu dep[5]);
    void hip_api(const string& func_andor_args, int status, lu tick, lu ticks, uint64_t correlation_id);
    void hip_api_kernel(const string& func_andor_args, const string& kernname, int status, lu tick, lu ticks, uint64_t correlation_id);
    void roctx(uint64_t correlation_id, const string& message, lu tick, lu ticks);
    void roctx_mark(uint64_t correlation_id, const string& message, lu tick);
};

std::mutex TlsData::the_class_mutex;
std::unordered_map<std::thread::id, TlsData*> TlsData::the_map;

}

void TlsData::hsa_api(const string& func, const string& args, lu tick, lu ticks, int localStatus)
{
    fprintf(stream, "HSA: pid:%d tid:%s %s %s ret=%d @%lu +%lu\n", pid, tid, func.c_str(), args.c_str(), localStatus, tick, ticks);
    flush();
}

void TlsData::hsa_api(const string& func, const string& args, lu tick, lu ticks, uint64_t localStatus)
{
    fprintf(stream, "HSA: pid:%d tid:%s %s %s ret=%lu @%lu +%lu\n", pid, tid, func.c_str(), args.c_str(), localStatus, tick, ticks);
    flush();
}

void TlsData::hsa_api(const string& func, const string& args, lu tick, lu ticks)
{
    fprintf(stream, "HSA: pid:%d tid:%s %s %s ret=void @%lu +%lu\n", pid, tid, func.c_str(), args.c_str(), tick, ticks);
    flush();
}

void TlsData::hsa_host_dispatch_kernel(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, const string& name, const hsa_kernel_dispatch_packet_t *packet)
{
    fprintf(stream, "HSA: pid:%d tid:%s dispatch queue:%lu agent:%lu signal:%lu name:'%s' tick:%lu id:%lu workgroup:{%d,%d,%d} grid:{%d,%d,%d}\n",
            pid, tid, queue->id, agent.handle, signal.handle, name.c_str(), tick, id,
            packet->workgroup_size_x, packet->workgroup_size_y, packet->workgroup_size_z,
            packet->grid_size_x, packet->grid_size_y, packet->grid_size_z);
    flush();
}

void TlsData::hsa_host_dispatch_barrier(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, lu dep[5], const hsa_barrier_and_packet_t *packet)
{
    fprintf(stream, "HSA: pid:%d tid:%s barrier queue:%lu agent:%lu signal:%lu dep1:%lu dep2:%lu dep3:%lu dep4:%lu dep5:%lu tick:%lu id:%lu\n",
            pid, tid, queue->id, agent.handle, signal.handle, dep[0], dep[1], dep[2], dep[3], dep[4], tick, id);
    flush();
}

void TlsData::hsa_dispatch_kernel(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, const string& name, uint64_t correlation_id)
{
    fprintf(stream, "HSA: pid:%d tid:%s dispatch queue:%lu agent:%lu signal:%lu name:'%s' start:%lu stop:%lu id:%lu\n",
            pid, tid, queue->id, agent.handle, signal.handle, name.c_str(), start, stop, id);
    flush();
}

void TlsData::hsa_dispatch_barrier(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, lu dep[5])
{
    fprintf(stream, "HSA: pid:%d tid:%s barrier queue:%lu agent:%lu signal:%lu start:%lu stop:%lu dep1:%lu dep2:%lu dep3:%lu dep4:%lu dep5:%lu id:%lu\n",
            pid, tid, queue->id, agent.handle, signal.handle, start, stop, dep[0], dep[1], dep[2], dep[3], dep[4], id);
    flush();
}

void TlsData::hsa_dispatch_copy(hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu dep[5])
{
    fprintf(stream, "HSA: pid:%d tid:%s copy agent:%lu signal:%lu start:%lu stop:%lu dep1:%lu dep2:%lu dep3:%lu dep4:%lu dep5:%lu\n",
            pid, tid, agent.handle, signal.handle, start, stop, dep[0], dep[1], dep[2], dep[3], dep[4]);
    flush();
}

void TlsData::hip_api(const string& func_andor_args, int status, lu tick, lu ticks, uint64_t correlation_id)
{
    fprintf(stream, "HIP: pid:%d tid:%s %s ret=%d @%lu +%lu\n", pid, tid, func_andor_args.c_str(), status, tick, ticks);
    flush();
}

void TlsData::hip_api_kernel(const string& func_andor_args, const string& kernname, int status, lu tick, lu ticks, uint64_t correlation_id)
{
    fprintf(stream, "HIP: pid:%d tid:%s %s [%s] ret=%d @%lu +%lu\n", pid, tid, func_andor_args.c_str(), kernname.c_str(), status, tick, ticks);
    flush();
}

void TlsData::roctx(uint64_t correlation_id, const string& message, lu tick, lu ticks)
{
    fprintf(stream, "RTX: pid:%d tid:%s %s @%lu +%lu\n", pid, tid, message.c_str(), tick, ticks);
    flush();
}

void TlsData::roctx_mark(uint64_t correlation_id, const string& message, lu tick)
{
    fprintf(stream, "RTX: pid:%d tid:%s %s @%lu\n", pid, tid, message.c_str(), tick);
    flush();
}

void RtgOutPrintfLockless::open(const string& filename)
{
    if (0 != mkdir(filename.c_str(), 0777)) {
        fprintf(stderr, "failed to create directory for %s\n", filename.c_str());
        exit(EXIT_FAILURE);
    }
    this->filename = filename + "/" + filename;
}

void RtgOutPrintfLockless::hsa_api(const string& func, const string& args, lu tick, lu ticks, int localStatus)
{
    TlsData::Get(filename)->hsa_api(func, args, tick, ticks, localStatus);
}

void RtgOutPrintfLockless::hsa_api(const string& func, const string& args, lu tick, lu ticks, uint64_t localStatus)
{
    TlsData::Get(filename)->hsa_api(func, args, tick, ticks, localStatus);
}

void RtgOutPrintfLockless::hsa_api(const string& func, const string& args, lu tick, lu ticks)
{
    TlsData::Get(filename)->hsa_api(func, args, tick, ticks);
}

void RtgOutPrintfLockless::hsa_host_dispatch_kernel(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, const string& name, const hsa_kernel_dispatch_packet_t *packet)
{
    TlsData::Get(filename)->hsa_host_dispatch_kernel(queue, agent, signal, tick, id, name, packet);
}

void RtgOutPrintfLockless::hsa_host_dispatch_barrier(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, lu dep[5], const hsa_barrier_and_packet_t *packet)
{
    TlsData::Get(filename)->hsa_host_dispatch_barrier(queue, agent, signal, tick, id, dep, packet);
}

void RtgOutPrintfLockless::hsa_dispatch_kernel(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, const string& name, uint64_t correlation_id)
{
    TlsData::Get(filename)->hsa_dispatch_kernel(queue, agent, signal, start, stop, id, name, correlation_id);
}

void RtgOutPrintfLockless::hsa_dispatch_barrier(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, lu dep[5])
{
    TlsData::Get(filename)->hsa_dispatch_barrier(queue, agent, signal, start, stop, id, dep);
}

void RtgOutPrintfLockless::hsa_dispatch_copy(hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu dep[5])
{
    TlsData::Get(filename)->hsa_dispatch_copy(agent, signal, start, stop, dep);
}

void RtgOutPrintfLockless::hip_api(const string& func_andor_args, int status, lu tick, lu ticks, uint64_t correlation_id)
{
    TlsData::Get(filename)->hip_api(func_andor_args, status, tick, ticks, correlation_id);
}

void RtgOutPrintfLockless::hip_api_kernel(const string& func_andor_args, const string& kernname, int status, lu tick, lu ticks, uint64_t correlation_id)
{
    TlsData::Get(filename)->hip_api_kernel(func_andor_args, kernname, status, tick, ticks, correlation_id);
}

void RtgOutPrintfLockless::roctx(uint64_t correlation_id, const string& message, lu tick, lu ticks)
{
    TlsData::Get(filename)->roctx(correlation_id, message, tick, ticks);
}

void RtgOutPrintfLockless::roctx_mark(uint64_t correlation_id, const string& message, lu tick)
{
    TlsData::Get(filename)->roctx_mark(correlation_id, message, tick);
}

void RtgOutPrintfLockless::close()
{
    TlsData::flush_all();
}

