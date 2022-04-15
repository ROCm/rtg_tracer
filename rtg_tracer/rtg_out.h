/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once

#include <string>

#include <hsa/hsa.h>

using std::string;

typedef long unsigned lu;

struct hip_api_data_s; // hip_api_data_t is a typedef

class RtgOut {

public:

virtual void open(const string& filename) = 0;

virtual void hsa_api(const string& func, const string& args, lu tick, lu ticks, int localStatus) = 0;
virtual void hsa_api(const string& func, const string& args, lu tick, lu ticks, uint64_t localStatus) = 0;
virtual void hsa_api(const string& func, const string& args, lu tick, lu ticks) = 0;

virtual void hsa_host_dispatch_kernel (hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, const string& name, const hsa_kernel_dispatch_packet_t *packet) = 0;
virtual void hsa_host_dispatch_barrier(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, lu dep[5], const hsa_barrier_and_packet_t *packet) = 0;

virtual void hsa_dispatch_kernel (hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, const string& name, uint64_t correlation_id) = 0;
virtual void hsa_dispatch_barrier(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, lu dep[5]) = 0;
virtual void hsa_dispatch_copy   (hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu dep[5]) = 0;

virtual void hip_api(uint32_t cid, struct hip_api_data_s *data, int status, lu tick, lu ticks, const std::string &kernname, bool args) = 0;

virtual void roctx(uint64_t correlation_id, const string& message, lu tick, lu ticks) = 0;
virtual void roctx_mark(uint64_t correlation_id, const string& message, lu tick) = 0;

virtual void close() = 0;

};
