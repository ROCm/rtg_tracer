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

#include "rtg_out_printf.h"

constexpr std::size_t BUF_SIZE = 4096;
//constexpr std::size_t OUT_SIZE = 10240;
constexpr std::size_t OUT_SIZE = 1;

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
    if (ret >= BUF_SIZE || ret < 0) {
        fprintf(stderr, "insufficient buffer size for RtgOutPrintf\n");
        exit(EXIT_FAILURE);
    }
}

void RtgOutPrintf::open(const string& filename)
{
    pid = getpid();
    if ("stderr" == filename) {
        stream = stderr;
    }
    else {
        stream = fopen(filename.c_str(), "w");
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

void RtgOutPrintf::hsa_host_dispatch_kernel(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, const string& name, const hsa_kernel_dispatch_packet_t *packet)
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "HSA: pid:%d tid:%s dispatch queue:%lu agent:%lu signal:%lu name:'%s' tick:%lu id:%lu workgroup:{%d,%d,%d} grid:{%d,%d,%d}\n",
            pid, tid(), queue->id, agent.handle, signal.handle, name.c_str(), tick, id,
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

void RtgOutPrintf::hsa_dispatch_kernel(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, const string& name, uint64_t correlation_id)
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "HSA: pid:%d tid:%s dispatch queue:%lu agent:%lu signal:%lu name:'%s' start:%lu stop:%lu id:%lu\n",
            pid, tid(), queue->id, agent.handle, signal.handle, name.c_str(), start, stop, id));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::hsa_dispatch_barrier(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, lu dep[5])
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "HSA: pid:%d tid:%s barrier queue:%lu agent:%lu signal:%lu start:%lu stop:%lu dep1:%lu dep2:%lu dep3:%lu dep4:%lu dep5:%lu id:%lu\n",
            pid, tid(), queue->id, agent.handle, signal.handle, start, stop, dep[0], dep[1], dep[2], dep[3], dep[4], id));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::hsa_dispatch_copy(hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu dep[5])
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "HSA: pid:%d tid:%s copy agent:%lu signal:%lu start:%lu stop:%lu dep1:%lu dep2:%lu dep3:%lu dep4:%lu dep5:%lu\n",
            pid, tid(), agent.handle, signal.handle, start, stop, dep[0], dep[1], dep[2], dep[3], dep[4]));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::hip_api(const string& func_andor_args, int status, lu tick, lu ticks, uint64_t correlation_id)
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "HIP: pid:%d tid:%s %s ret=%d @%lu +%lu\n", pid, tid(), func_andor_args.c_str(), status, tick, ticks));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::hip_api_kernel(const string& func_andor_args, const string& kernname, int status, lu tick, lu ticks, uint64_t correlation_id)
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "HIP: pid:%d tid:%s %s [%s] ret=%d @%lu +%lu\n", pid, tid(), func_andor_args.c_str(), kernname.c_str(), status, tick, ticks));
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
    if (stderr == stream) {
        // do nothing
    }
    else {
        TlsData::flush_all(stream);
        fclose(stream);
    }
}

