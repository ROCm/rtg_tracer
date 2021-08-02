#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "rtg_out_printf.h"

constexpr std::size_t BUF_SIZE = 4096;
constexpr std::size_t OUT_SIZE = 10240;

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
        {
            std::lock_guard<std::mutex> lock(the_mutex);
            the_map[std::this_thread::get_id()] = this;
        }
    }

    ~TlsData() {
        flush();
    }

    void push(const char *line) {
        out.push_back(line);
        if (out.size() >= OUT_SIZE) {
            {
                std::lock_guard<std::mutex> lock(the_mutex);
                for (auto& the_line : out) {
                    fprintf(stream, "%s", the_line);
                }
            }
            out.clear();
            out.reserve(OUT_SIZE);
        }
    }

    void flush() {
        std::lock_guard<std::mutex> lock(the_mutex);
        for (auto& the_line : out) {
            fprintf(stream, "%s", the_line);
        }
        fflush(stream);
    }

    static void flush_all(FILE *stream) {
        std::lock_guard<std::mutex> lock(the_mutex);
        for (auto& keyval : the_map) {
            for (auto& the_line : keyval.second->out) {
                fprintf(stream, "%s", the_line);
            }
        }
        fflush(stream);
    }
};

std::mutex TlsData::the_mutex;
std::unordered_map<std::thread::id, TlsData*> TlsData::the_map;

static void check(int ret)
{
    if (ret >= BUF_SIZE || ret < 0) {
        fprintf(stderr, "insufficient buffer size for RtgOutPrintf\n");
        exit(EXIT_FAILURE);
    }
}

void RtgOutPrintf::open(string filename)
{
    stream = fopen(filename.c_str(), "w");
}

void RtgOutPrintf::hsa_api(int pid, string tid, string func, string args, lu tick, lu ticks, int localStatus)
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "HSA: pid:%d tid:%s %s %s ret=%d @%lu +%lu\n", pid, tid.c_str(), func.c_str(), args.c_str(), localStatus, tick, ticks));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::hsa_api(int pid, string tid, string func, string args, lu tick, lu ticks, uint64_t localStatus)
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "HSA: pid:%d tid:%s %s %s ret=%lu @%lu +%lu\n", pid, tid.c_str(), func.c_str(), args.c_str(), localStatus, tick, ticks));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::hsa_api(int pid, string tid, string func, string args, lu tick, lu ticks)
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "HSA: pid:%d tid:%s %s %s ret=void @%lu +%lu\n", pid, tid.c_str(), func.c_str(), args.c_str(), tick, ticks));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::hsa_host_dispatch_kernel(int pid, string tid, hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, string name, const hsa_kernel_dispatch_packet_t *packet)
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "HSA: pid:%d tid:%s dispatch queue:%lu agent:%lu signal:%lu name:'%s' tick:%lu id:%lu workgroup:{%d,%d,%d} grid:{%d,%d,%d}\n",
            pid, tid.c_str(), queue->id, agent.handle, signal.handle, name.c_str(), tick, id,
            packet->workgroup_size_x, packet->workgroup_size_y, packet->workgroup_size_z,
            packet->grid_size_x, packet->grid_size_y, packet->grid_size_z));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::hsa_host_dispatch_barrier(int pid, string tid, hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, lu dep[5], const hsa_barrier_and_packet_t *packet)
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "HSA: pid:%d tid:%s barrier queue:%lu agent:%lu signal:%lu dep1:%lu dep2:%lu dep3:%lu dep4:%lu dep5:%lu tick:%lu id:%lu\n",
            pid, tid.c_str(), queue->id, agent.handle, signal.handle, dep[0], dep[1], dep[2], dep[3], dep[4], tick, id));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::hsa_dispatch_kernel(int pid, string tid, hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, string name)
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "HSA: pid:%d tid:%s dispatch queue:%lu agent:%lu signal:%lu name:'%s' start:%lu stop:%lu id:%lu\n",
            pid, tid.c_str(), queue->id, agent.handle, signal.handle, name.c_str(), start, stop, id));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::hsa_dispatch_barrier(int pid, string tid, hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, lu dep[5])
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "HSA: pid:%d tid:%s barrier queue:%lu agent:%lu signal:%lu start:%lu stop:%lu dep1:%lu dep2:%lu dep3:%lu dep4:%lu dep5:%lu id:%lu\n",
            pid, tid.c_str(), queue->id, agent.handle, signal.handle, start, stop, dep[0], dep[1], dep[2], dep[3], dep[4], id));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::hsa_dispatch_copy(int pid, string tid, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu dep[5])
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "HSA: pid:%d tid:%s copy agent:%lu signal:%lu start:%lu stop:%lu dep1:%lu dep2:%lu dep3:%lu dep4:%lu dep5:%lu\n",
            pid, tid.c_str(), agent.handle, signal.handle, start, stop, dep[0], dep[1], dep[2], dep[3], dep[4]));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::hip_api(int pid, string tid, string func_andor_args, int status, lu tick, lu ticks)
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "HIP: pid:%d tid:%s %s ret=%d @%lu +%lu\n", pid, tid.c_str(), func_andor_args.c_str(), status, tick, ticks));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::hip_api_kernel(int pid, string tid, string func_andor_args, string kernname, int status, lu tick, lu ticks)
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "HIP: pid:%d tid:%s %s [%s] ret=%d @%lu +%lu\n", pid, tid.c_str(), func_andor_args.c_str(), kernname.c_str(), status, tick, ticks));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::roctx(int pid, string tid, string message, lu tick, lu ticks)
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "RTX: pid:%d tid:%s %s @%lu +%lu\n", pid, tid.c_str(), message.c_str(), tick, ticks));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::roctx_mark(int pid, string tid, string message, lu tick)
{
    char *buf = new char[BUF_SIZE];
    check(snprintf(buf, BUF_SIZE, "RTX: pid:%d tid:%s %s @%lu\n", pid, tid.c_str(), message.c_str(), tick));
    TlsData::Get(stream)->push(buf);
}

void RtgOutPrintf::close()
{
    TlsData::flush_all(stream);
    fclose(stream);
}

