#include "rtg_out_printf.h"

void RtgOutPrintf::open(string filename)
{
    stream = fopen(filename.c_str(), "w");
}

void RtgOutPrintf::hsa_api(int pid, string tid, string func, string args, lu tick, lu ticks, int localStatus)
{
    fprintf(stream, "HSA: pid:%d tid:%s %s %s ret=%d @%lu +%lu\n", pid, tid.c_str(), func.c_str(), args.c_str(), localStatus, tick, ticks);
    fflush(stream);
}

void RtgOutPrintf::hsa_api(int pid, string tid, string func, string args, lu tick, lu ticks, uint64_t localStatus)
{
    fprintf(stream, "HSA: pid:%d tid:%s %s %s ret=%lu @%lu +%lu\n", pid, tid.c_str(), func.c_str(), args.c_str(), localStatus, tick, ticks);
    fflush(stream);
}

void RtgOutPrintf::hsa_api(int pid, string tid, string func, string args, lu tick, lu ticks)
{
    fprintf(stream, "HSA: pid:%d tid:%s %s %s ret=void @%lu +%lu\n", pid, tid.c_str(), func.c_str(), args.c_str(), tick, ticks);
    fflush(stream);
}

void RtgOutPrintf::hsa_host_dispatch_kernel(int pid, string tid, hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, string name, const hsa_kernel_dispatch_packet_t *packet)
{
    fprintf(stream, "HSA: pid:%d tid:%s dispatch queue:%lu agent:%lu signal:%lu name:'%s' tick:%lu id:%lu workgroup:{%d,%d,%d} grid:{%d,%d,%d}\n",
            pid, tid.c_str(), queue->id, agent.handle, signal.handle, name.c_str(), tick, id,
            packet->workgroup_size_x, packet->workgroup_size_y, packet->workgroup_size_z,
            packet->grid_size_x, packet->grid_size_y, packet->grid_size_z);
    fflush(stream);
}

void RtgOutPrintf::hsa_host_dispatch_barrier(int pid, string tid, hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, lu dep[5], const hsa_barrier_and_packet_t *packet)
{
    fprintf(stream, "HSA: pid:%d tid:%s barrier queue:%lu agent:%lu signal:%lu dep1:%lu dep2:%lu dep3:%lu dep4:%lu dep5:%lu tick:%lu id:%lu\n",
            pid, tid.c_str(), queue->id, agent.handle, signal.handle, dep[0], dep[1], dep[2], dep[3], dep[4], tick, id);
    fflush(stream);
}

void RtgOutPrintf::hsa_dispatch_kernel(int pid, string tid, hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, string name)
{
    fprintf(stream, "HSA: pid:%d tid:%s dispatch queue:%lu agent:%lu signal:%lu name:'%s' start:%lu stop:%lu id:%lu\n",
            pid, tid.c_str(), queue->id, agent.handle, signal.handle, name.c_str(), start, stop, id);
    fflush(stream);
}

void RtgOutPrintf::hsa_dispatch_barrier(int pid, string tid, hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, lu dep[5])
{
    fprintf(stream, "HSA: pid:%d tid:%s barrier queue:%lu agent:%lu signal:%lu start:%lu stop:%lu dep1:%lu dep2:%lu dep3:%lu dep4:%lu dep5:%lu id:%lu\n",
            pid, tid.c_str(), queue->id, agent.handle, signal.handle, start, stop, dep[0], dep[1], dep[2], dep[3], dep[4], id);
    fflush(stream);
}

void RtgOutPrintf::hsa_dispatch_copy(int pid, string tid, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu dep[5])
{
    fprintf(stream, "HSA: pid:%d tid:%s copy agent:%lu signal:%lu start:%lu stop:%lu dep1:%lu dep2:%lu dep3:%lu dep4:%lu dep5:%lu\n",
            pid, tid.c_str(), agent.handle, signal.handle, start, stop, dep[0], dep[1], dep[2], dep[3], dep[4]);
    fflush(stream);
}

void RtgOutPrintf::hip_api(int pid, string tid, string func_andor_args, int status, lu tick, lu ticks)
{
    fprintf(stream, "HIP: pid:%d tid:%s %s ret=%d @%lu +%lu\n", pid, tid.c_str(), func_andor_args.c_str(), status, tick, ticks);
    fflush(stream);
}

void RtgOutPrintf::hip_api_kernel(int pid, string tid, string func_andor_args, string kernname, int status, lu tick, lu ticks)
{
    fprintf(stream, "HIP: pid:%d tid:%s %s [%s] ret=%d @%lu +%lu\n", pid, tid.c_str(), func_andor_args.c_str(), kernname.c_str(), status, tick, ticks);
    fflush(stream);
}

void RtgOutPrintf::roctx(int pid, string tid, string message, lu tick, lu ticks)
{
    fprintf(stream, "RTX: pid:%d tid:%s %s @%lu +%lu\n", pid, tid.c_str(), message.c_str(), tick, ticks);
    fflush(stream);
}

void RtgOutPrintf::roctx_mark(int pid, string tid, string message, lu tick)
{
    fprintf(stream, "RTX: pid:%d tid:%s %s @%lu\n", pid, tid.c_str(), message.c_str(), tick);
    fflush(stream);
}

void RtgOutPrintf::close()
{
    fclose(stream);
}

