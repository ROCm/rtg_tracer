/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once

#include "rtg_out.h"

class RtgOutPrintfLockless : public RtgOut {

public:

virtual void open(const string& filename) override;

virtual void hsa_api(const string& func, const string& args, lu tick, lu ticks, int localStatus) override;
virtual void hsa_api(const string& func, const string& args, lu tick, lu ticks, uint64_t localStatus) override;
virtual void hsa_api(const string& func, const string& args, lu tick, lu ticks) override;

virtual void hsa_host_dispatch_kernel(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, const string& name, const hsa_kernel_dispatch_packet_t *packet) override;
virtual void hsa_host_dispatch_barrier(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, lu dep[5], const hsa_barrier_and_packet_t *packet) override;

virtual void hsa_dispatch_kernel(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, const string& name, uint64_t correlation_id) override;
virtual void hsa_dispatch_barrier(hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, lu dep[5]) override;
virtual void hsa_dispatch_copy(hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu dep[5]) override;

virtual void hip_api(uint32_t cid, struct hip_api_data_s *data, int status, lu tick, lu ticks, const std::string &kernname, bool args) override;

virtual void roctx(uint64_t correlation_id, const string& message, lu tick, lu ticks) override;
virtual void roctx_mark(uint64_t correlation_id, const string& message, lu tick) override;

virtual void close() override;

private:

string filename;
std::vector<std::string> hip_api_names;

};
