#pragma once

#include "rtg_out.h"

class MetadataTable;
class StringTable;
class OpTable;
class ApiTable;
class ApiIdList;

class RtgOutRpd : public RtgOut {

public:

virtual void open(string filename) override;

virtual void hsa_api(int pid, string tid, string func, string args, lu tick, lu ticks, int localStatus) override;
virtual void hsa_api(int pid, string tid, string func, string args, lu tick, lu ticks, uint64_t localStatus) override;
virtual void hsa_api(int pid, string tid, string func, string args, lu tick, lu ticks) override;

virtual void hsa_host_dispatch_kernel(int pid, string tid, hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, string name, const hsa_kernel_dispatch_packet_t *packet) override;
virtual void hsa_host_dispatch_barrier(int pid, string tid, hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, lu dep[5], const hsa_barrier_and_packet_t *packet) override;

virtual void hsa_dispatch_kernel(int pid, string tid, hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, string name) override;
virtual void hsa_dispatch_barrier(int pid, string tid, hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, lu dep[5]) override;
virtual void hsa_dispatch_copy(int pid, string tid, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu dep[5]) override;

virtual void hip_api(int pid, string tid, string func_andor_args, int status, lu tick, lu ticks, uint64_t correlation_id) override;
virtual void hip_api_kernel(int pid, string tid, string func_andor_args, string kernname, int status, lu tick, lu ticks, uint64_t correlation_id) override;

virtual void roctx(int pid, string tid, uint64_t correlation_id, string message, lu tick, lu ticks) override;
virtual void roctx_mark(int pid, string tid, uint64_t correlation_id, string message, lu tick) override;

virtual void close() override;

// Table Recorders
MetadataTable *s_metadataTable = NULL;
StringTable *s_stringTable = NULL;
OpTable *s_opTable = NULL;
ApiTable *s_apiTable = NULL;
// API list
ApiIdList *s_apiList = NULL;

};
