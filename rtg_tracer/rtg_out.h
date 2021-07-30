#pragma once

#include <string>

#include <hsa/hsa.h>

using std::string;

typedef long unsigned lu;

class RtgOut {

public:

virtual void open(string filename) = 0;

virtual void hsa_api(int pid, string tid, string func, string args, lu tick, lu ticks, int localStatus) = 0;
virtual void hsa_api(int pid, string tid, string func, string args, lu tick, lu ticks, uint64_t localStatus) = 0;
virtual void hsa_api(int pid, string tid, string func, string args, lu tick, lu ticks) = 0;

virtual void hsa_host_dispatch_kernel (int pid, string tid, hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, string name, const hsa_kernel_dispatch_packet_t *packet) = 0;
virtual void hsa_host_dispatch_barrier(int pid, string tid, hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, lu dep[5], const hsa_barrier_and_packet_t *packet) = 0;

virtual void hsa_dispatch_kernel (int pid, string tid, hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, string name) = 0;
virtual void hsa_dispatch_barrier(int pid, string tid, hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, lu dep[5]) = 0;
virtual void hsa_dispatch_copy   (int pid, string tid, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu dep[5]) = 0;

virtual void hip_api(int pid, string tid, string func_andor_args, int status, lu tick, lu ticks) = 0;
virtual void hip_api_kernel(int pid, string tid, string func_andor_args, string kernname, int status, lu tick, lu ticks) = 0;

virtual void roctx(int pid, string tid, string message, lu tick, lu ticks) = 0;
virtual void roctx_mark(int pid, string tid, string message, lu tick) = 0;

virtual void close() = 0;

};
