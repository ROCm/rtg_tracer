#pragma once

#include "rtg_out.h"

class MetadataTable;
class StringTable;
class OpTable;
class ApiTable;
class ApiIdList;

class RtgOutRpd : public RtgOut {

public:

virtual void open(string filename);

virtual void hsa_api(int pid, string tid, string func, string args, lu tick, lu ticks, int localStatus);
virtual void hsa_api(int pid, string tid, string func, string args, lu tick, lu ticks, uint64_t localStatus);
virtual void hsa_api(int pid, string tid, string func, string args, lu tick, lu ticks);

virtual void hsa_host_dispatch_kernel(int pid, string tid, hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, string name, const hsa_kernel_dispatch_packet_t *packet);
virtual void hsa_host_dispatch_barrier(int pid, string tid, hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu tick, lu id, lu dep[5], const hsa_barrier_and_packet_t *packet);

virtual void hsa_dispatch_kernel(int pid, string tid, hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, string name);
virtual void hsa_dispatch_barrier(int pid, string tid, hsa_queue_t *queue, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu id, lu dep[5]);
virtual void hsa_dispatch_copy(int pid, string tid, hsa_agent_t agent, hsa_signal_t signal, lu start, lu stop, lu dep[5]);

virtual void hip_api(int pid, string tid, string func_andor_args, int status, lu tick, lu ticks);
virtual void hip_api_kernel(int pid, string tid, string func_andor_args, string kernname, int status, lu tick, lu ticks);

virtual void roctx(int pid, string tid, string message, lu tick, lu ticks);
virtual void roctx_mark(int pid, string tid, string message, lu tick);

virtual void close();

// Table Recorders
MetadataTable *s_metadataTable = NULL;
StringTable *s_stringTable = NULL;
OpTable *s_opTable = NULL;
ApiTable *s_apiTable = NULL;
// API list
ApiIdList *s_apiList = NULL;

};
