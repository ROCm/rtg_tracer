/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <list>
#include <mutex>
#include <sstream>
#include <string>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <cxxabi.h>

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <hsa/amd_hsa_kernel_code.h>
#include <hsa/amd_hsa_signal.h>

#include <hip/hip_runtime_api.h>
#include <hip/amd_detail/hip_runtime_prof.h>
#include <hip/amd_detail/hip_prof_str.h>

#include "missing_ostream_definitions.h"

#include <roctracer/roctracer.h>
#include <roctracer/roctracer_roctx.h>

#include "ctpl_stl.h"

// User options, set using env vars.
#include "flags.h"

// These declarations were getting quite lengthy; moved to separate file to aid readability.
#include "ToStringDefinitions.h"

#include "rtg_out_printf.h"
#include "rtg_out_printf_lockless.h"
#include "rtg_out_rpd.h"

#define RTG_DISABLE_LOGGING 0
#define RTG_ENABLE_HSA_AMD_MEMORY_LOCK_TO_POOL 0
#define RTG_ENABLE_HSA_AMD_RUNTIME_QUEUE_CREATE_REGISTER 0

namespace RTG {

//////////////////////////////////////////////////////////////////////////////
// typedefs //////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef hsa_ven_amd_aqlprofile_pfn_t pfn_t;
typedef hsa_ven_amd_aqlprofile_event_t event_t;
typedef hsa_ven_amd_aqlprofile_parameter_t parameter_t;
typedef hsa_ven_amd_aqlprofile_profile_t profile_t;
typedef hsa_ext_amd_aql_pm4_packet_t packet_t;
typedef uint32_t packet_word_t;
typedef uint64_t timestamp_t;
typedef std::atomic<unsigned int> counter_t;
//typedef std::recursive_mutex mutex_t;
typedef std::mutex mutex_t;
typedef std::unordered_map<uint64_t, const char*> symbols_map_t;

//////////////////////////////////////////////////////////////////////////////
// structs and classes ///////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

struct roctx_data_t {
    std::string message;
    uint64_t tick;
    uint64_t correlation_id;

    roctx_data_t(const char *m, uint64_t t, uint64_t c) : message(m), tick(t), correlation_id(c) {}
};

// for tracking agents and streams with 0-based index.
class AgentInfo {
public:
    hsa_agent_t get_agent() { return agent; }
    int get_agent_index() { return index; }
    int get_queue_index(hsa_queue_t *queue);
    hsa_queue_t* get_signal_queue(); // this is the agent's signal_queue, not one of the the per-queue signal queues

    struct Op {
        std::string name;
        uint64_t id; // correlation_id
    };

    void insert_op(const Op &op);
    uint64_t find_op(const std::string &name);

    static AgentInfo* Get(hsa_agent_t agent);
    static AgentInfo* Get(int index);
    static void get_agent_queue_indexes(hsa_agent_t agent, hsa_queue_t *queue, int &agent_index, int &queue_index);

    static void Init(const std::vector<hsa_agent_t> &agents);

private:
    explicit AgentInfo(hsa_agent_t agent, int index);

    hsa_agent_t agent;
    int index;
    hsa_queue_t *signal_queue;
    std::unordered_map<uint64_t, int> queue_index_map;
    mutex_t queue_index_mutex;
    mutex_t op_mutex;
    std::list<Op> op_list;
    static std::unordered_map<uint64_t, AgentInfo*> s_agent_info_map;
    static std::vector<AgentInfo*> s_agent_info;
};

//////////////////////////////////////////////////////////////////////////////
// global static variables ///////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

static hsa_signal_t gs_NullSignal; // handle is always 0

static HsaApiTable* gs_OrigHsaTable;     // The HSA Runtime's original table
static CoreApiTable gs_OrigCoreApiTable; // The HSA Runtime's versions of HSA core API functions
static AmdExtTable gs_OrigExtApiTable;   // The HSA Runtime's versions of HSA ext API functions
static hsa_ven_amd_loader_1_01_pfn_t gs_OrigLoaderExtTable; // The HSA Runtime's versions of HSA loader ext API functions
static std::unordered_map<std::string,bool> gs_hsa_enabled_map; // Lookup which HSA functions we want to trace

// Executables loading tracking, for looking up kernel names.
static mutex_t gs_kernel_name_mutex_; // protects gs_symbols_map_
static symbols_map_t* gs_symbols_map_; // maps HSA executable address to kernel name, protected by gs_kernel_name_mutex_

static std::vector<hsa_agent_t> gs_gpu_agents; // all gpu agents, in order reported by HSA, possibly filtered by visible
static std::vector<hsa_agent_t> gs_cpu_agents; // all cpu agents, in order reported by HSA

static ctpl::thread_pool gs_pool(1);        // thread pool for signal waits
static ctpl::thread_pool gs_signal_pool(1); // thread pool for signal destroy

static counter_t gs_host_count_dispatches{0}; // counts kernel disaptches
static counter_t gs_host_count_barriers{0};   // counts barrier dispaches
static counter_t gs_host_count_copies{0};     // counts async copies
static counter_t gs_host_count_signals{0};    // counts signals we create
static counter_t gs_cb_count_dispatches{0};   // counts kernel disaptches that have completed
static counter_t gs_cb_count_barriers{0};     // counts barrier dispaches that have completed
static counter_t gs_cb_count_copies{0};       // counts async copies that have completed
static counter_t gs_cb_count_signals{0};      // counts signals we destroy
static counter_t gs_did{0};                   // global dispach id

static std::atomic<uint64_t> gs_correlation_id_counter{0}; // global API counter, for correlation ID, for HIP and roctx calls

static bool loaded{false};
static RtgOut* gs_out{NULL};  // output interface
static FILE *gs_stream{NULL}; // output only used for HCC_PROFILE mode

// Need to allocate hip_api_data_t, but cannot use new operator due to incomplete default constructors.
static thread_local std::vector<char[sizeof(hip_api_data_t)]> gstl_hip_api_data(HIP_API_ID_NUMBER);
static thread_local std::vector<uint64_t> gstl_hip_api_tick(HIP_API_ID_NUMBER);

static thread_local std::vector<roctx_data_t> gstl_roctx_stack; // for roctx range push pop
static thread_local std::unordered_map<int,roctx_data_t> gstl_roctx_range; // for roctx range start stop

//////////////////////////////////////////////////////////////////////////////
// global static function declarations ///////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// Ensure proper shutdown order. Single funtion to perform all resource cleanup.
static void finalize();
static void finalize_once();

// Support for HIP API callbacks.
static void* hip_activity_callback(uint32_t cid, activity_record_t* record, const void* data, void* arg);
static void* hip_api_callback(uint32_t domain, uint32_t cid, const void* data, void* arg);

// Support for ROCTX callbacks.
static void* roctx_callback(uint32_t domain, uint32_t cid, const void* data, void* arg);

// Support for HSA function table hooks.
static void InitCoreApiTable(CoreApiTable* table);
static void InitAmdExtTable(AmdExtTable* table);
static void InitEnabledTable(std::string what_to_trace, std::string what_not_to_trace);
static void InitEnabledTableCore(bool value);
static void InitEnabledTableExtApi(bool value);

// HSA executable tracking, for looking up kernel names.
static const char* GetKernelNameRef(uint64_t addr);
static hsa_status_t hsa_executable_freeze_interceptor(hsa_executable_t executable, const char *options);
static hsa_status_t hsa_executable_symbols_cb(hsa_executable_t exec, hsa_executable_symbol_t symbol, void *data);

// HSA iterator callbcaks
static hsa_status_t hsa_iterate_agent_cb(hsa_agent_t agent, void* data);
static void hsa_amd_queue_intercept_cb(const void* in_packets, uint64_t count, uint64_t user_que_idx, void* data, hsa_amd_queue_intercept_packet_writer writer);

// HSA helpers
static hsa_signal_t CreateSignal();
static void DestroySignal(hsa_signal_t signal);

///////////////////////////////////
/// class AgentInfo implementation
///////////////////////////////////
std::unordered_map<uint64_t, AgentInfo*> AgentInfo::s_agent_info_map;
std::vector<AgentInfo*> AgentInfo::s_agent_info;

AgentInfo::AgentInfo(hsa_agent_t agent, int index)
    : agent(agent)
    , index{index}
    , signal_queue{nullptr}
    , queue_index_map{}
    , queue_index_mutex{}
{
}

int AgentInfo::get_queue_index(hsa_queue_t *queue) {
    std::lock_guard<mutex_t> lock(queue_index_mutex);
    if (queue_index_map.count(queue->id) == 0) {
        queue_index_map[queue->id] = queue_index_map.size() + 1;
    }
    return queue_index_map[queue->id];
}

hsa_queue_t* AgentInfo::get_signal_queue() {
    if (signal_queue == nullptr) {
        hsa_status_t status;
        // create a regular queue; this is for our fake signaling queue
        status = gs_OrigCoreApiTable.hsa_queue_create_fn(agent, 2048, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr,
                std::numeric_limits<unsigned int>::max(), std::numeric_limits<unsigned int>::max(), &signal_queue);
        if (status != HSA_STATUS_SUCCESS) {
            const char *msg;
            hsa_status_string(status, &msg);
            fprintf(stderr, "RTG Tracer: could not create agent signal queue: %s\n", msg);
            exit(EXIT_FAILURE);
        }
        // make sure profiling is enabled for the newly created queue
        status = gs_OrigExtApiTable.hsa_amd_profiling_set_profiler_enabled_fn(signal_queue, true);
        if (status != HSA_STATUS_SUCCESS) {
            const char *msg;
            hsa_status_string(status, &msg);
            fprintf(stderr, "RTG Tracer: could not create agent signal queue with profile: %s\n", msg);
            exit(EXIT_FAILURE);
        }
    }

    return signal_queue;
}

void AgentInfo::insert_op(const Op &op)
{
    std::lock_guard<mutex_t> lock(op_mutex);
    op_list.emplace_back(op);
}

uint64_t AgentInfo::find_op(const std::string &name)
{
    constexpr uint64_t not_found = std::numeric_limits<uint64_t>::max();
    uint64_t id = not_found;

    {
        std::lock_guard<mutex_t> lock(op_mutex);
        auto it = op_list.begin();
        for (; it != op_list.end(); ++it) {
            if (it->name == name) {
                id = it->id;
                op_list.erase(it);
                break;
            }
        }
    }

    if (id == not_found) {
        // HIP runtime (rocclr) can launch kernels without a corresponding HIP API call
        if (name.find("__amd_rocclr") == std::string::npos) {
            fprintf(stderr, "RTG Tracer: correlation id error for '%s'\n", name.c_str());
            exit(EXIT_FAILURE);
        }
    }

    return id;
}

AgentInfo* AgentInfo::Get(hsa_agent_t agent) {
    return s_agent_info_map[agent.handle];
}

AgentInfo* AgentInfo::Get(int index) {
    return s_agent_info[index];
}

void AgentInfo::get_agent_queue_indexes(hsa_agent_t agent, hsa_queue_t *queue, int &agent_index, int &queue_index)
{
    auto *info = AgentInfo::Get(agent);
    agent_index = info->get_agent_index();
    queue_index = info->get_queue_index(queue);
}

void AgentInfo::Init(const std::vector<hsa_agent_t> &agents)
{
    int index = 0;
    for (auto agent : agents) {
        auto info = new RTG::AgentInfo(agent, index++);
        s_agent_info.emplace_back(info);
        s_agent_info_map.emplace(agent.handle, info);
    }
}

struct InterceptCallbackData
{
    InterceptCallbackData(hsa_queue_t *queue, hsa_agent_t agent, hsa_queue_t *signal_queue)
        : queue(queue), agent(agent), signal_queue(signal_queue), agent_index(0), queue_index(0), seq_index(0)
    {
        AgentInfo::get_agent_queue_indexes(agent, queue, agent_index, queue_index);
    }
    hsa_queue_t *queue;
    hsa_agent_t agent;
    hsa_queue_t *signal_queue;
    int agent_index;
    int queue_index;
    std::atomic<int> seq_index;
};

static inline int pid() {
    static int pid_ = getpid();
    return pid_;
}

static inline std::string pidstr() {
    std::ostringstream pid_os;
    pid_os << pid();
    return pid_os.str();
}

static inline std::string tidstr() {
    std::ostringstream tid_os;
    tid_os << std::this_thread::get_id();
    return tid_os.str();
}

uint64_t inline tick() {
    struct timespec tp;
    ::clock_gettime(CLOCK_MONOTONIC, &tp);
    return (uint64_t)tp.tv_sec * (1000ULL * 1000ULL * 1000ULL) + (uint64_t)tp.tv_nsec;
}

#if RTG_DISABLE_LOGGING

#define LOG_HCC

// HSA APIs
#define TRACE(...)
#define LOG_STATUS(status) ({ hsa_status_t localStatus = status; localStatus; })
#define LOG_SIGNAL(status) ({ hsa_signal_value_t localStatus = status; localStatus; })
#define LOG_UINT64(status) ({ uint64_t localStatus = status; localStatus; })
#define LOG_UINT32(status) ({ uint32_t localStatus = status; localStatus; })
#define LOG_VOID(status) ({ status; })

// HSA async dispatches
#define LOG_DISPATCH_HOST
#define LOG_DISPATCH
#define LOG_BARRIER_HOST
#define LOG_BARRIER
#define LOG_COPY

#define LOG_HIP

#define LOG_ROCTX
#define LOG_ROCTX_MARK

#else // RTG_DISABLE_LOGGING

// copied from hcc runtime's HCC_PROFILE=2
#define LOG_HCC(start, end, type, tag, msg, agent_id_, queue_id_, seq_num_) \
{\
    std::stringstream sstream;\
    sstream << "profile: " << std::setw(7) << type << ";\t" \
                         << std::setw(40) << tag\
                         << ";\t" << std::fixed << std::setw(6) << std::setprecision(1) << (end-start)/1000.0 << " us;";\
    sstream << "\t" << start << ";\t" << end << ";";\
    sstream << "\t" << "#" << agent_id_ << "." << queue_id_ << "." << seq_num_ << ";"; \
    sstream <<  msg << "\n";\
    fprintf(gs_stream, "%s", sstream.str().c_str());\
}

#define TRACE(...) \
    std::string func = __func__; \
    std::string args = "(" + ToString(__VA_ARGS__) + ")"; \
    uint64_t tick_ = tick();

#define LOG_STATUS(status)                                                               \
    ({                                                                                   \
        hsa_status_t localStatus = status; /*local copy so status only evaluated once*/  \
        gs_out->hsa_api(func, args, tick_, tick() - tick_, localStatus);                 \
        localStatus;                                                                     \
    })

#define LOG_SIGNAL(status)                                                                     \
    ({                                                                                         \
        hsa_signal_value_t localStatus = status; /*local copy so status only evaluated once*/  \
        gs_out->hsa_api(func, args, tick_, tick() - tick_, static_cast<int>(localStatus));     \
        localStatus;                                                                           \
    })

#define LOG_UINT64(status)                                                           \
    ({                                                                               \
        uint64_t localStatus = status; /*local copy so status only evaluated once*/  \
        gs_out->hsa_api(func, args, tick_, tick() - tick_, localStatus);             \
        localStatus;                                                                 \
    })

#define LOG_UINT32(status)                                                                      \
    ({                                                                                          \
        uint32_t localStatus = status; /*local copy so status only evaluated once*/             \
        gs_out->hsa_api(func, args, tick_, tick() - tick_, static_cast<uint64_t>(localStatus)); \
        localStatus;                                                                            \
    })

#define LOG_VOID(status)                                    \
    ({                                                      \
        status;                                             \
        gs_out->hsa_api(func, args, tick_, tick() - tick_); \
    })

#define LOG_DISPATCH_HOST gs_out->hsa_host_dispatch_kernel
#define LOG_BARRIER_HOST  gs_out->hsa_host_dispatch_barrier
#define LOG_DISPATCH      gs_out->hsa_dispatch_kernel
#define LOG_BARRIER       gs_out->hsa_dispatch_barrier
#define LOG_COPY          gs_out->hsa_dispatch_copy

#define LOG_HIP           gs_out->hip_api

#define LOG_ROCTX         gs_out->roctx
#define LOG_ROCTX_MARK    gs_out->roctx_mark

#endif // RTG_DISABLE_LOGGING

static const uint16_t kInvalidHeader = (HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE) |
    (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static const uint16_t kBarrierHeader = (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
    (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static const hsa_barrier_and_packet_t kBarrierPacket = {kInvalidHeader, 0, 0, {0,0,0,0,0}, 0, {0}};

static uint64_t submit_to_signal_queue(hsa_queue_t *signal_queue, hsa_signal_t new_signal, hsa_signal_t original_signal)
{
    // Submit a new packet just for decrementing the original signal. This is done in a separate queue. Signal packet depends on real packet.
    const uint32_t queueSize = signal_queue->size;
    const uint32_t queueMask = queueSize - 1;
    uint64_t index = gs_OrigCoreApiTable.hsa_queue_add_write_index_screlease_fn(signal_queue, 1);
    uint64_t read = gs_OrigCoreApiTable.hsa_queue_load_read_index_relaxed_fn(signal_queue);
    while ((index - gs_OrigCoreApiTable.hsa_queue_load_read_index_scacquire_fn(signal_queue)) >= queueMask) {
        sched_yield();
    }
    hsa_barrier_and_packet_t *barrier = &((hsa_barrier_and_packet_t*)(signal_queue->base_address))[index & queueMask];
    *barrier = kBarrierPacket;
    barrier->completion_signal = original_signal;
    if (new_signal.handle) {
        barrier->dep_signal[0] = new_signal;
    }
    barrier->header = kBarrierHeader;
    gs_OrigCoreApiTable.hsa_signal_store_relaxed_fn(signal_queue->doorbell_signal, index);
    //fprintf(stderr, "RTG Tracer: wrote new BARRIER AND at queue %lu index %lu (prev read was %lu)\n", signal_queue->id, index, read);
    return index;
}

enum CopyDirection {
    H2H = 0,
    H2D = 1,
    D2H = 2,
    D2D = 3,
};

struct SignalCallbackData
{
    SignalCallbackData(std::string name, uint64_t cid, InterceptCallbackData *data, hsa_signal_t deleter_signal, hsa_signal_t signal, hsa_signal_t orig_signal, bool owns_orig_signal, const hsa_kernel_dispatch_packet_t *packet)
        : name(name), data(data), queue(data->queue), agent(data->agent), deleter_signal(deleter_signal), signal(signal), orig_signal(orig_signal), owns_orig_signal(owns_orig_signal), bytes(0), direction(0),
            is_copy(false), is_barrier(false), dep{0,0,0,0,0}, id_(gs_did++), seq_num_(data->seq_index++), cid(cid)
    {
        if (RTG_HSA_HOST_DISPATCH) {
            LOG_DISPATCH_HOST(queue, agent, signal, tick(), id_, name, packet);
        }
    }

    SignalCallbackData(hsa_queue_t* queue, hsa_agent_t agent, hsa_signal_t deleter_signal, hsa_signal_t signal, hsa_signal_t orig_signal, bool owns_orig_signal, uint32_t num_dep_signals, const hsa_signal_t* dep_signals, size_t bytes, int direction, int seq)
        : name(), queue(queue), agent(agent), deleter_signal(deleter_signal), signal(signal), orig_signal(orig_signal), owns_orig_signal(owns_orig_signal), bytes(bytes), direction(direction),
            is_copy(true),
            is_barrier(false),
            dep{
                  (num_dep_signals>0 ? dep_signals[0].handle : 0),
                  (num_dep_signals>1 ? dep_signals[1].handle : 0),
                  (num_dep_signals>2 ? dep_signals[2].handle : 0),
                  (num_dep_signals>3 ? dep_signals[3].handle : 0),
                  (num_dep_signals>4 ? dep_signals[4].handle : 0)
            },
            id_(0),
            seq_num_(seq),
            cid(0)

    {}

    SignalCallbackData(InterceptCallbackData *data, hsa_signal_t deleter_signal, hsa_signal_t signal, hsa_signal_t orig_signal, bool owns_orig_signal, const hsa_barrier_and_packet_t* packet)
        : name(), data(data), queue(data->queue), agent(data->agent), deleter_signal(deleter_signal), signal(signal), orig_signal(orig_signal), owns_orig_signal(owns_orig_signal), bytes(0), direction(0),
            is_copy(false),
            is_barrier(true),
            dep{
                  (packet->dep_signal[0].handle),
                  (packet->dep_signal[1].handle),
                  (packet->dep_signal[2].handle),
                  (packet->dep_signal[3].handle),
                  (packet->dep_signal[4].handle)
            },
            id_(gs_did++),
            seq_num_(data->seq_index++),
            cid(0)
    {
        if (RTG_HSA_HOST_DISPATCH) {
            LOG_BARRIER_HOST(queue, agent, signal, tick(), id_, dep, packet);
        }
    }

    bool compute_profile() {
        hsa_status_t status;
        if (is_copy) {
            hsa_amd_profiling_async_copy_time_t copy_time{};
            status = gs_OrigExtApiTable.hsa_amd_profiling_get_async_copy_time_fn(
                    signal, &copy_time);
            if (status != HSA_STATUS_SUCCESS) {
                const char *msg;
                gs_OrigCoreApiTable.hsa_status_string_fn(status, &msg);
                fprintf(stderr, "RTG Tracer: signal callback copy time failed: %s\n", msg);
                return false;
            }
            start = copy_time.start;
            stop = copy_time.end;
        }
        else {
            hsa_amd_profiling_dispatch_time_t dispatch_time{};
            status = gs_OrigExtApiTable.hsa_amd_profiling_get_dispatch_time_fn(
                    agent, signal, &dispatch_time);
            if (status != HSA_STATUS_SUCCESS) {
                const char *msg;
                gs_OrigCoreApiTable.hsa_status_string_fn(status, &msg);
                fprintf(stderr, "RTG Tracer: signal callback dispatch time failed: %s\n", msg);
                return false;
            }
            start = dispatch_time.start;
            stop = dispatch_time.end;
        }
        return true;
    }

    std::string name;
    InterceptCallbackData *data;
    hsa_queue_t* queue;
    hsa_agent_t agent;
    hsa_signal_t signal;
    hsa_signal_t orig_signal;
    hsa_signal_t deleter_signal;
    bool owns_orig_signal;
    size_t bytes;
    int direction;
    bool is_copy;
    bool is_barrier;
    long unsigned dep[5];
    long unsigned start;
    long unsigned stop;
    long unsigned id_;
    int seq_num_;
    uint64_t cid;
};

static void SignalDestroyer(int id, hsa_signal_t deleter, hsa_signal_t ours, hsa_signal_t theirs, bool owns_orig_signal)
{
    //fprintf(stderr, "RTG Tracer: SignalDestroyer id=%d deleter=%lu ours=%lu theirs=%lu\n", id, deleter.handle, ours.handle, theirs.handle);
    gs_OrigCoreApiTable.hsa_signal_wait_relaxed_fn(deleter, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
    DestroySignal(ours);
    DestroySignal(deleter);
    if (owns_orig_signal) {
        DestroySignal(theirs);
    }
}

const char * getDirectionString(int dir)
{
    switch(dir) {
        case H2H: return "HostToHost";
        case H2D: return "HostToDevice";
        case D2H: return "DeviceToHost";
        case D2D: return "DeviceToDevice";
    }
    return "UnknownCopy";
}

std::string getCopyString(size_t sizeBytes, uint64_t start, uint64_t end)
{
    double bw = (double)(sizeBytes)/(end-start) * (1000.0/1024.0) * (1000.0/1024.0);
    std::ostringstream ss;
    ss << "\t" << sizeBytes << " bytes;\t" << sizeBytes/1024.0/1024 << " MB;\t" << bw << " GB/s;";
    return ss.str();
}

static void SignalWaiter(int id, SignalCallbackData *data)
{
    //fprintf(stderr, "RTG Tracer: SignalWaiter id=%d queue=%lu agent=%lu signal=%lu orig_signal=%lu\n", id, data->queue->id, data->agent.handle, data->signal.handle, data->orig_signal.handle);
    gs_OrigCoreApiTable.hsa_signal_wait_relaxed_fn(data->signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

    bool okay = data->compute_profile();
    if (okay) {
        if (HCC_PROFILE) {
            if (data->is_barrier) {
                LOG_HCC(data->start, data->stop, "barrier", "", "", data->data->agent_index, data->data->queue_index, data->seq_num_);
                ++gs_cb_count_barriers;
            }
            else if (data->is_copy) {
                int agent_id = 0;
                int queue_id = 0;
                AgentInfo::get_agent_queue_indexes(data->agent, data->queue, agent_id, queue_id);
                const char *tag_ = getDirectionString(data->direction);
                std::string msgstr = getCopyString(data->bytes, data->start, data->stop);
                LOG_HCC(data->start, data->stop, "copy", tag_, msgstr.c_str(), agent_id, queue_id, data->seq_num_);
                ++gs_cb_count_copies;
            }
            else {
                LOG_HCC(data->start, data->stop, "kernel", data->name, "", data->data->agent_index, data->data->queue_index, data->seq_num_);
                ++gs_cb_count_dispatches;
            }
        }
        else {
            if (data->is_barrier) {
                LOG_BARRIER(data->queue, data->agent, data->signal, data->start, data->stop, data->id_, data->dep);
                ++gs_cb_count_barriers;
            }
            else if (data->is_copy) {
                LOG_COPY(data->agent, data->signal, data->start, data->stop, data->dep);
                ++gs_cb_count_copies;
            }
            else {
                LOG_DISPATCH(data->queue, data->agent, data->signal, data->start, data->stop, data->id_, data->name, data->cid);
                ++gs_cb_count_dispatches;
            }
        }
    }
    else {
        fprintf(stderr, "SOMETHING IS NOT OKAY\n");
    }

    // we created the signal, we must free, but we can't until we know it is no longer needed
    // so wait on the associated original signal
    gs_signal_pool.push(SignalDestroyer, data->deleter_signal, data->signal, data->orig_signal, data->owns_orig_signal);
    delete data;
}

bool InitHsaTable(HsaApiTable* pTable)
{
    if (pTable == nullptr) {
        fprintf(stderr, "RTG Tracer: HSA Runtime provided a nullptr API Table");
        return false;
    }

    gs_OrigHsaTable = pTable;

    // This saves the original pointers
    memcpy(static_cast<void*>(&gs_OrigCoreApiTable),
           static_cast<const void*>(pTable->core_),
           sizeof(CoreApiTable));

    // This saves the original pointers
    memcpy(static_cast<void*>(&gs_OrigExtApiTable),
           static_cast<const void*>(pTable->amd_ext_),
           sizeof(AmdExtTable));

    hsa_status_t status = HSA_STATUS_SUCCESS;
    status = gs_OrigCoreApiTable.hsa_system_get_major_extension_table_fn(
            HSA_EXTENSION_AMD_LOADER,
            1,
            sizeof(hsa_ven_amd_loader_1_01_pfn_t),
            &gs_OrigLoaderExtTable);

    if (status != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "RTG Tracer: Cannot get loader extension function table");
        return false;
    }

    InitCoreApiTable(pTable->core_);
    InitAmdExtTable(pTable->amd_ext_);

    if (RTG_PROFILE_COPY) {
        status = hsa_amd_profiling_async_copy_enable(true);
        if (status != HSA_STATUS_SUCCESS) {
            fprintf(stderr, "RTG Tracer: hsa_amd_profiling_async_copy_enable failed\n");
            return false;
        }
    }

    return true;
}

void RestoreHsaTable(HsaApiTable* pTable)
{
    // This restores the original pointers
    memcpy(static_cast<void*>(pTable->core_),
           static_cast<const void*>(&gs_OrigCoreApiTable),
           sizeof(CoreApiTable));

    // This restores the original pointers
    memcpy(static_cast<void*>(pTable->amd_ext_),
           static_cast<const void*>(&gs_OrigExtApiTable),
           sizeof(AmdExtTable));
}

hsa_status_t hsa_init() {
    TRACE();
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_init_fn());
}

hsa_status_t hsa_shut_down() {
    TRACE();
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_shut_down_fn());
}

hsa_status_t hsa_system_get_info(hsa_system_info_t attribute, void* value) {
    TRACE(attribute, value);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_system_get_info_fn(attribute, value));
}

hsa_status_t hsa_extension_get_name(uint16_t extension, const char** name) {
    TRACE(extension, name);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_extension_get_name_fn(extension, name));
}

hsa_status_t hsa_system_extension_supported(uint16_t extension, uint16_t version_major, uint16_t version_minor, bool* result) {
    TRACE(extension, version_major, version_minor, result);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_system_extension_supported_fn(extension, version_major, version_minor, result));
}

hsa_status_t hsa_system_major_extension_supported(uint16_t extension, uint16_t version_major, uint16_t* version_minor, bool* result) {
    TRACE(extension, version_major, version_minor, result);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_system_major_extension_supported_fn(extension, version_major, version_minor, result));
}

hsa_status_t hsa_system_get_extension_table(uint16_t extension, uint16_t version_major, uint16_t version_minor, void* table) {
    TRACE(extension, version_major, version_minor, table);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_system_get_extension_table_fn( extension, version_major, version_minor, table));
}

hsa_status_t hsa_system_get_major_extension_table(uint16_t extension, uint16_t version_major, size_t table_length, void* table) {
    TRACE(extension, version_major, table_length, table);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_system_get_major_extension_table_fn(extension, version_major, table_length, table));
}

hsa_status_t hsa_iterate_agents(hsa_status_t (*callback)(hsa_agent_t agent, void* data), void* data) {
    TRACE(callback, data);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_iterate_agents_fn(callback, data));
}

hsa_status_t hsa_agent_get_info(hsa_agent_t agent, hsa_agent_info_t attribute, void* value) {
    TRACE(agent, attribute, value);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_agent_get_info_fn(agent, attribute, value));
}

hsa_status_t hsa_agent_get_exception_policies(hsa_agent_t agent, hsa_profile_t profile, uint16_t* mask) {
    TRACE(agent, profile, mask);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_agent_get_exception_policies_fn(agent, profile, mask));
}

hsa_status_t hsa_cache_get_info(hsa_cache_t cache, hsa_cache_info_t attribute, void* value) {
    TRACE(cache, attribute, value);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_cache_get_info_fn(cache, attribute, value));
}

hsa_status_t hsa_agent_iterate_caches(hsa_agent_t agent, hsa_status_t (*callback)(hsa_cache_t cache, void* data), void* value) {
    TRACE(agent, callback);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_agent_iterate_caches_fn(agent, callback, value));
}

hsa_status_t hsa_agent_extension_supported(uint16_t extension, hsa_agent_t agent, uint16_t version_major, uint16_t version_minor, bool* result) {
    TRACE(extension, agent, version_major, version_minor, result);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_agent_extension_supported_fn(extension, agent, version_major, version_minor, result));
}

hsa_status_t hsa_agent_major_extension_supported(uint16_t extension, hsa_agent_t agent, uint16_t version_major, uint16_t* version_minor, bool* result) {
    TRACE(extension, agent, version_major, version_minor, result);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_agent_major_extension_supported_fn(extension, agent, version_major, version_minor, result));
}

hsa_status_t hsa_queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type, void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data), void* data, uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t** queue) {
    TRACE(agent, size, type, callback, data, private_segment_size, group_segment_size, queue);
    hsa_status_t status;
    if (RTG_PROFILE) {
        hsa_queue_t *signal_queue;
        // create a regular queue; this is for our fake signaling queue
        status = gs_OrigCoreApiTable.hsa_queue_create_fn(agent, size, type, nullptr, nullptr, private_segment_size, group_segment_size, &signal_queue);
        if (status != HSA_STATUS_SUCCESS) {
            return status;
        }
        // make sure profiling is enabled for the newly created queue
        status = gs_OrigExtApiTable.hsa_amd_profiling_set_profiler_enabled_fn(signal_queue, true);
        if (status != HSA_STATUS_SUCCESS) {
            return status;
        }
        // call special ext api to create an interceptible queue
        status = gs_OrigExtApiTable.hsa_amd_queue_intercept_create_fn(agent, size, type, callback, data, private_segment_size, group_segment_size, queue);
        if (status != HSA_STATUS_SUCCESS) {
            return status;
        }
        // make sure profiling is enabled for the newly created queue
        status = gs_OrigExtApiTable.hsa_amd_profiling_set_profiler_enabled_fn(*queue, true);
        if (status != HSA_STATUS_SUCCESS) {
            return status;
        }
        // set our intercept callback
        // we leak the InterceptCallbackData instance
        InterceptCallbackData *data = new InterceptCallbackData(*queue, agent, signal_queue);
        status = gs_OrigExtApiTable.hsa_amd_queue_intercept_register_fn(
                *queue, hsa_amd_queue_intercept_cb, data);
    }
    else {
        status = gs_OrigCoreApiTable.hsa_queue_create_fn(agent, size, type, callback, data, private_segment_size, group_segment_size, queue);
    }
    if (gs_hsa_enabled_map["hsa_queue_create"]) {
        return LOG_STATUS(status);
    }
    return status;
}

hsa_status_t hsa_soft_queue_create(hsa_region_t region, uint32_t size, hsa_queue_type32_t type, uint32_t features, hsa_signal_t completion_signal, hsa_queue_t** queue) {
    TRACE(region, size, type, features, completion_signal, queue);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_soft_queue_create_fn(region, size, type, features, completion_signal, queue));
}

hsa_status_t hsa_queue_destroy(hsa_queue_t* queue) {
    TRACE(queue);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_queue_destroy_fn(queue));
}

hsa_status_t hsa_queue_inactivate(hsa_queue_t* queue) {
    TRACE(queue);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_queue_inactivate_fn(queue));
}

uint64_t hsa_queue_load_read_index_scacquire(const hsa_queue_t* queue) {
    TRACE(queue);
    return LOG_UINT64(gs_OrigCoreApiTable.hsa_queue_load_read_index_scacquire_fn(queue));
}

uint64_t hsa_queue_load_read_index_relaxed(const hsa_queue_t* queue) {
    TRACE(queue);
    return LOG_UINT64(gs_OrigCoreApiTable.hsa_queue_load_read_index_relaxed_fn(queue));
}

uint64_t hsa_queue_load_write_index_scacquire(const hsa_queue_t* queue) {
    TRACE(queue);
    return LOG_UINT64(gs_OrigCoreApiTable.hsa_queue_load_write_index_scacquire_fn(queue));
}

uint64_t hsa_queue_load_write_index_relaxed(const hsa_queue_t* queue) {
    TRACE(queue);
    return LOG_UINT64(gs_OrigCoreApiTable.hsa_queue_load_write_index_relaxed_fn(queue));
}

void hsa_queue_store_write_index_relaxed(const hsa_queue_t* queue, uint64_t value) {
    TRACE(queue, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_queue_store_write_index_relaxed_fn(queue, value));
}

void hsa_queue_store_write_index_screlease(const hsa_queue_t* queue, uint64_t value) {
    TRACE(queue, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_queue_store_write_index_screlease_fn(queue, value));
}

uint64_t hsa_queue_cas_write_index_scacq_screl(const hsa_queue_t* queue, uint64_t expected, uint64_t value) {
    TRACE(queue, expected, value);
    return LOG_UINT64(gs_OrigCoreApiTable.hsa_queue_cas_write_index_scacq_screl_fn(queue, expected, value));
}

uint64_t hsa_queue_cas_write_index_scacquire(const hsa_queue_t* queue, uint64_t expected, uint64_t value) {
    TRACE(queue, expected, value);
    return LOG_UINT64(gs_OrigCoreApiTable.hsa_queue_cas_write_index_scacquire_fn(queue, expected, value));
}

uint64_t hsa_queue_cas_write_index_relaxed(const hsa_queue_t* queue, uint64_t expected, uint64_t value) {
    TRACE(queue, expected, value);
    return LOG_UINT64(gs_OrigCoreApiTable.hsa_queue_cas_write_index_relaxed_fn(queue, expected, value));
}

uint64_t hsa_queue_cas_write_index_screlease(const hsa_queue_t* queue, uint64_t expected, uint64_t value) {
    TRACE(queue, expected, value);
    return LOG_UINT64(gs_OrigCoreApiTable.hsa_queue_cas_write_index_screlease_fn(queue, expected, value));
}

uint64_t hsa_queue_add_write_index_scacq_screl(const hsa_queue_t* queue, uint64_t value) {
    TRACE(queue, value);
    return LOG_UINT64(gs_OrigCoreApiTable.hsa_queue_add_write_index_scacq_screl_fn(queue, value));
}

uint64_t hsa_queue_add_write_index_scacquire(const hsa_queue_t* queue, uint64_t value) {
    TRACE(queue, value);
    return LOG_UINT64(gs_OrigCoreApiTable.hsa_queue_add_write_index_scacquire_fn(queue, value));
}

uint64_t hsa_queue_add_write_index_relaxed(const hsa_queue_t* queue, uint64_t value) {
    TRACE(queue, value);
    return LOG_UINT64(gs_OrigCoreApiTable.hsa_queue_add_write_index_relaxed_fn(queue, value));
}

uint64_t hsa_queue_add_write_index_screlease(const hsa_queue_t* queue, uint64_t value) {
    TRACE(queue, value);
    return LOG_UINT64(gs_OrigCoreApiTable.hsa_queue_add_write_index_screlease_fn(queue, value));
}

void hsa_queue_store_read_index_relaxed(const hsa_queue_t* queue, uint64_t value) {
    TRACE(queue, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_queue_store_read_index_relaxed_fn(queue, value));
}

void hsa_queue_store_read_index_screlease(const hsa_queue_t* queue, uint64_t value) {
    TRACE(queue, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_queue_store_read_index_screlease_fn(queue, value));
}

hsa_status_t hsa_agent_iterate_regions(hsa_agent_t agent, hsa_status_t (*callback)(hsa_region_t region, void* data), void* data) {
    TRACE(agent, callback, data);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_agent_iterate_regions_fn(agent, callback, data));
}

hsa_status_t hsa_region_get_info(hsa_region_t region, hsa_region_info_t attribute, void* value) {
    TRACE(region, attribute, value);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_region_get_info_fn(region, attribute, value));
}

hsa_status_t hsa_memory_register(void* address, size_t size) {
    TRACE(address, size);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_memory_register_fn(address, size));
}

hsa_status_t hsa_memory_deregister(void* address, size_t size) {
    TRACE(address, size);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_memory_deregister_fn(address, size));
}

hsa_status_t hsa_memory_allocate(hsa_region_t region, size_t size, void** ptr) {
    TRACE(region, size, ptr);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_memory_allocate_fn(region, size, ptr));
}

hsa_status_t hsa_memory_free(void* ptr) {
    TRACE(ptr);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_memory_free_fn(ptr));
}

hsa_status_t hsa_memory_copy(void* dst, const void* src, size_t size) {
    TRACE(dst, src, size);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_memory_copy_fn(dst, src, size));
}

hsa_status_t hsa_memory_assign_agent(void* ptr, hsa_agent_t agent, hsa_access_permission_t access) {
    TRACE(ptr, agent, access);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_memory_assign_agent_fn(ptr, agent, access));
}

hsa_status_t hsa_signal_create(hsa_signal_value_t initial_value, uint32_t num_consumers, const hsa_agent_t* consumers, hsa_signal_t* signal) {
    TRACE(initial_value, num_consumers, consumers, signal);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_signal_create_fn(initial_value, num_consumers, consumers, signal));
}

hsa_status_t hsa_signal_destroy(hsa_signal_t signal) {
    //fprintf(stderr, "RTG Tracer: hsa_signal_destroy signal=%lu\n", signal.handle);
    TRACE(signal);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_signal_destroy_fn(signal));
}

hsa_signal_value_t hsa_signal_load_relaxed(hsa_signal_t signal) {
    TRACE(signal);
    return LOG_SIGNAL(gs_OrigCoreApiTable.hsa_signal_load_relaxed_fn(signal));
}

hsa_signal_value_t hsa_signal_load_scacquire(hsa_signal_t signal) {
    TRACE(signal);
    return LOG_SIGNAL(gs_OrigCoreApiTable.hsa_signal_load_scacquire_fn(signal));
}

void hsa_signal_store_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_signal_store_relaxed_fn(signal, value));
}

void hsa_signal_store_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_signal_store_screlease_fn(signal, value));
}

void hsa_signal_silent_store_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_signal_silent_store_relaxed_fn(signal, value));
}

void hsa_signal_silent_store_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_signal_silent_store_screlease_fn(signal, value));
}

hsa_signal_value_t hsa_signal_wait_relaxed(hsa_signal_t signal, hsa_signal_condition_t condition, hsa_signal_value_t compare_value, uint64_t timeout_hint, hsa_wait_state_t wait_expectancy_hint) {
    TRACE(signal, condition, compare_value, timeout_hint, wait_expectancy_hint);
    return LOG_SIGNAL(gs_OrigCoreApiTable.hsa_signal_wait_relaxed_fn(signal, condition, compare_value, timeout_hint, wait_expectancy_hint));
}

hsa_signal_value_t hsa_signal_wait_scacquire(hsa_signal_t signal, hsa_signal_condition_t condition, hsa_signal_value_t compare_value, uint64_t timeout_hint, hsa_wait_state_t wait_expectancy_hint) {
    TRACE(signal, condition, compare_value, timeout_hint, wait_expectancy_hint);
    return LOG_SIGNAL(gs_OrigCoreApiTable.hsa_signal_wait_scacquire_fn(signal, condition, compare_value, timeout_hint, wait_expectancy_hint));
}

hsa_status_t hsa_signal_group_create(uint32_t num_signals, const hsa_signal_t* signals, uint32_t num_consumers, const hsa_agent_t* consumers, hsa_signal_group_t* signal_group) {
    TRACE(num_signals, signals, num_consumers, consumers, signal_group);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_signal_group_create_fn(num_signals, signals, num_consumers, consumers, signal_group));
}

hsa_status_t hsa_signal_group_destroy(hsa_signal_group_t signal_group) {
    TRACE(signal_group);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_signal_group_destroy_fn(signal_group));
}

hsa_status_t hsa_signal_group_wait_any_relaxed(hsa_signal_group_t signal_group, const hsa_signal_condition_t* conditions, const hsa_signal_value_t* compare_values, hsa_wait_state_t wait_state_hint, hsa_signal_t* signal, hsa_signal_value_t* value) {
    TRACE(signal_group, conditions, compare_values, wait_state_hint, signal, value);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_signal_group_wait_any_relaxed_fn(signal_group, conditions, compare_values, wait_state_hint, signal, value));
}

hsa_status_t hsa_signal_group_wait_any_scacquire(hsa_signal_group_t signal_group, const hsa_signal_condition_t* conditions, const hsa_signal_value_t* compare_values, hsa_wait_state_t wait_state_hint, hsa_signal_t* signal, hsa_signal_value_t* value) {
    TRACE(signal_group, conditions, compare_values, wait_state_hint, signal, value);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_signal_group_wait_any_scacquire_fn( signal_group, conditions, compare_values, wait_state_hint, signal, value));
}

void hsa_signal_and_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_signal_and_relaxed_fn(signal, value));
}

void hsa_signal_and_scacquire(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_signal_and_scacquire_fn(signal, value));
}

void hsa_signal_and_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_signal_and_screlease_fn(signal, value));
}

void hsa_signal_and_scacq_screl(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_signal_and_scacq_screl_fn(signal, value));
}

void hsa_signal_or_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_signal_or_relaxed_fn(signal, value));
}

void hsa_signal_or_scacquire(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_signal_or_scacquire_fn(signal, value));
}

void hsa_signal_or_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_signal_or_screlease_fn(signal, value));
}

void hsa_signal_or_scacq_screl(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_signal_or_scacq_screl_fn(signal, value));
}

void hsa_signal_xor_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_signal_xor_relaxed_fn(signal, value));
}

void hsa_signal_xor_scacquire(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_signal_xor_scacquire_fn(signal, value));
}

void hsa_signal_xor_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_signal_xor_screlease_fn(signal, value));
}

void hsa_signal_xor_scacq_screl(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_signal_xor_scacq_screl_fn(signal, value));
}

void hsa_signal_add_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_signal_add_relaxed_fn(signal, value));
}

void hsa_signal_add_scacquire(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_signal_add_scacquire_fn(signal, value));
}

void hsa_signal_add_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_signal_add_screlease_fn(signal, value));
}

void hsa_signal_add_scacq_screl(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_signal_add_scacq_screl_fn(signal, value));
}

void hsa_signal_subtract_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_signal_subtract_relaxed_fn(signal, value));
}

void hsa_signal_subtract_scacquire(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_signal_subtract_scacquire_fn(signal, value));
}

void hsa_signal_subtract_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_signal_subtract_screlease_fn(signal, value));
}

void hsa_signal_subtract_scacq_screl(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    LOG_VOID(gs_OrigCoreApiTable.hsa_signal_subtract_scacq_screl_fn(signal, value));
}

hsa_signal_value_t hsa_signal_exchange_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    return LOG_SIGNAL(gs_OrigCoreApiTable.hsa_signal_exchange_relaxed_fn(signal, value));
}

hsa_signal_value_t hsa_signal_exchange_scacquire(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    return LOG_SIGNAL(gs_OrigCoreApiTable.hsa_signal_exchange_scacquire_fn(signal, value));
}

hsa_signal_value_t hsa_signal_exchange_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    return LOG_SIGNAL(gs_OrigCoreApiTable.hsa_signal_exchange_screlease_fn(signal, value));
}

hsa_signal_value_t hsa_signal_exchange_scacq_screl(hsa_signal_t signal, hsa_signal_value_t value) {
    TRACE(signal, value);
    return LOG_SIGNAL(gs_OrigCoreApiTable.hsa_signal_exchange_scacq_screl_fn(signal, value));
}

hsa_signal_value_t hsa_signal_cas_relaxed(hsa_signal_t signal, hsa_signal_value_t expected, hsa_signal_value_t value) {
    TRACE(signal, expected, value);
    return LOG_SIGNAL(gs_OrigCoreApiTable.hsa_signal_cas_relaxed_fn(signal, expected, value));
}

hsa_signal_value_t hsa_signal_cas_scacquire(hsa_signal_t signal, hsa_signal_value_t expected, hsa_signal_value_t value) {
    TRACE(signal, expected, value);
    return LOG_SIGNAL(gs_OrigCoreApiTable.hsa_signal_cas_scacquire_fn(signal, expected, value));
}

hsa_signal_value_t hsa_signal_cas_screlease(hsa_signal_t signal, hsa_signal_value_t expected, hsa_signal_value_t value) {
    TRACE(signal, expected, value);
    return LOG_SIGNAL(gs_OrigCoreApiTable.hsa_signal_cas_screlease_fn(signal, expected, value));
}

hsa_signal_value_t hsa_signal_cas_scacq_screl(hsa_signal_t signal, hsa_signal_value_t expected, hsa_signal_value_t value) {
    TRACE(signal, expected, value);
    return LOG_SIGNAL(gs_OrigCoreApiTable.hsa_signal_cas_scacq_screl_fn(signal, expected, value));
}

//===--- Instruction Set Architecture -------------------------------------===//

hsa_status_t hsa_isa_from_name(const char *name, hsa_isa_t *isa) {
    TRACE(name, isa);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_isa_from_name_fn(name, isa));
}

hsa_status_t hsa_agent_iterate_isas(hsa_agent_t agent, hsa_status_t (*callback)(hsa_isa_t isa, void *data), void *data) {
    TRACE(agent, callback, data);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_agent_iterate_isas_fn(agent, callback, data));
}

/* deprecated */
hsa_status_t hsa_isa_get_info(hsa_isa_t isa, hsa_isa_info_t attribute, uint32_t index, void *value) {
    TRACE(isa, attribute, index, value);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_isa_get_info_fn(isa, attribute, index, value));
}

hsa_status_t hsa_isa_get_info_alt(hsa_isa_t isa, hsa_isa_info_t attribute, void *value) {
    TRACE(isa, attribute, value);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_isa_get_info_alt_fn(isa, attribute, value));
}

hsa_status_t hsa_isa_get_exception_policies(hsa_isa_t isa, hsa_profile_t profile, uint16_t *mask) {
    TRACE(isa, profile, mask);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_isa_get_exception_policies_fn(isa, profile, mask));
}

hsa_status_t hsa_isa_get_round_method(hsa_isa_t isa, hsa_fp_type_t fp_type, hsa_flush_mode_t flush_mode, hsa_round_method_t *round_method) {
    TRACE(isa, fp_type, flush_mode, round_method);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_isa_get_round_method_fn(isa, fp_type, flush_mode, round_method));
}

hsa_status_t hsa_wavefront_get_info(hsa_wavefront_t wavefront, hsa_wavefront_info_t attribute, void *value) {
    TRACE(wavefront, attribute, value);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_wavefront_get_info_fn(wavefront, attribute, value));
}

hsa_status_t hsa_isa_iterate_wavefronts(hsa_isa_t isa, hsa_status_t (*callback)(hsa_wavefront_t wavefront, void *data), void *data) {
    TRACE(isa, callback, data);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_isa_iterate_wavefronts_fn(isa, callback, data));
}

/* deprecated */
hsa_status_t hsa_isa_compatible(hsa_isa_t code_object_isa, hsa_isa_t agent_isa, bool *result) {
    TRACE(code_object_isa, agent_isa, result);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_isa_compatible_fn(code_object_isa, agent_isa, result));
}

//===--- Code Objects (deprecated) ----------------------------------------===//

/* deprecated */
hsa_status_t hsa_code_object_serialize(hsa_code_object_t code_object, hsa_status_t (*alloc_callback)(size_t size, hsa_callback_data_t data, void **address), hsa_callback_data_t callback_data, const char *options, void **serialized_code_object, size_t *serialized_code_object_size) {
    TRACE(code_object, alloc_callback, callback_data, options, serialized_code_object, serialized_code_object_size);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_code_object_serialize_fn(code_object, alloc_callback, callback_data, options, serialized_code_object, serialized_code_object_size));
}

/* deprecated */
hsa_status_t hsa_code_object_deserialize(void *serialized_code_object, size_t serialized_code_object_size, const char *options, hsa_code_object_t *code_object) {
    TRACE(serialized_code_object, serialized_code_object_size, options, code_object);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_code_object_deserialize_fn(serialized_code_object, serialized_code_object_size, options, code_object));
}

/* deprecated */
hsa_status_t hsa_code_object_destroy(hsa_code_object_t code_object) {
    TRACE(code_object);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_code_object_destroy_fn(code_object));
}

/* deprecated */
hsa_status_t hsa_code_object_get_info(hsa_code_object_t code_object, hsa_code_object_info_t attribute, void *value) {
    TRACE(code_object, attribute, value);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_code_object_get_info_fn(code_object, attribute, value));
}

/* deprecated */
hsa_status_t hsa_code_object_get_symbol(hsa_code_object_t code_object, const char *symbol_name, hsa_code_symbol_t *symbol) {
    TRACE(code_object, symbol_name, symbol);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_code_object_get_symbol_fn(code_object, symbol_name, symbol));
}

/* deprecated */
hsa_status_t hsa_code_object_get_symbol_from_name(hsa_code_object_t code_object, const char *module_name, const char *symbol_name, hsa_code_symbol_t *symbol) {
    TRACE(code_object, module_name, symbol_name, symbol);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_code_object_get_symbol_from_name_fn(code_object, module_name, symbol_name, symbol));
}

/* deprecated */
hsa_status_t hsa_code_symbol_get_info(hsa_code_symbol_t code_symbol, hsa_code_symbol_info_t attribute, void *value) {
    TRACE(code_symbol, attribute, value);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_code_symbol_get_info_fn(code_symbol, attribute, value));
}

/* deprecated */
hsa_status_t hsa_code_object_iterate_symbols(hsa_code_object_t code_object, hsa_status_t (*callback)(hsa_code_object_t code_object, hsa_code_symbol_t symbol, void *data), void *data) {
    TRACE(code_object, callback, data);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_code_object_iterate_symbols_fn(code_object, callback, data));
}

//===--- Executable -------------------------------------------------------===//

hsa_status_t hsa_code_object_reader_create_from_file(hsa_file_t file, hsa_code_object_reader_t *code_object_reader) {
    TRACE(file, code_object_reader);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_code_object_reader_create_from_file_fn(file, code_object_reader));
}

hsa_status_t hsa_code_object_reader_create_from_memory(const void *code_object, size_t size, hsa_code_object_reader_t *code_object_reader) {
    TRACE(code_object, size, code_object_reader);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_code_object_reader_create_from_memory_fn(code_object, size, code_object_reader));
}

hsa_status_t hsa_code_object_reader_destroy(hsa_code_object_reader_t code_object_reader) {
    TRACE(code_object_reader);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_code_object_reader_destroy_fn(code_object_reader));
}

/* deprecated */
hsa_status_t hsa_executable_create(hsa_profile_t profile, hsa_executable_state_t executable_state, const char *options, hsa_executable_t *executable) {
    TRACE(profile, executable_state, options, executable);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_executable_create_fn(profile, executable_state, options, executable));
}

hsa_status_t hsa_executable_create_alt(hsa_profile_t profile, hsa_default_float_rounding_mode_t default_float_rounding_mode, const char *options, hsa_executable_t *executable) {
    TRACE(profile, default_float_rounding_mode, options, executable);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_executable_create_alt_fn(profile, default_float_rounding_mode, options, executable));
}

hsa_status_t hsa_executable_destroy(hsa_executable_t executable) {
    TRACE(executable);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_executable_destroy_fn(executable));
}

/* deprecated */
hsa_status_t hsa_executable_load_code_object(hsa_executable_t executable, hsa_agent_t agent, hsa_code_object_t code_object, const char *options) {
    TRACE(executable, agent, code_object, options);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_executable_load_code_object_fn(executable, agent, code_object, options));
}

hsa_status_t hsa_executable_load_program_code_object(hsa_executable_t executable, hsa_code_object_reader_t code_object_reader, const char *options, hsa_loaded_code_object_t *loaded_code_object) {
    TRACE(executable, code_object_reader, options, loaded_code_object);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_executable_load_program_code_object_fn(executable, code_object_reader, options, loaded_code_object));
}

hsa_status_t hsa_executable_load_agent_code_object(hsa_executable_t executable, hsa_agent_t agent, hsa_code_object_reader_t code_object_reader, const char *options, hsa_loaded_code_object_t *loaded_code_object) {
    TRACE(executable, agent, code_object_reader, options, loaded_code_object);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_executable_load_agent_code_object_fn(executable, agent, code_object_reader, options, loaded_code_object));
}

hsa_status_t hsa_executable_freeze(hsa_executable_t executable, const char *options) {
    // this function is always intercepted, but we might not want to trace it; TRACE only sets some vars and is safe to always call
    TRACE(executable, options);
    hsa_status_t status;
    if (RTG_PROFILE) {
        status = hsa_executable_freeze_interceptor(executable, options);
    }
    else {
        status = gs_OrigCoreApiTable.hsa_executable_freeze_fn(executable, options);
    }
    if (gs_hsa_enabled_map["hsa_executable_freeze"]) {
        return LOG_STATUS(status);
    }
    return status;
}

hsa_status_t hsa_executable_get_info(hsa_executable_t executable, hsa_executable_info_t attribute, void *value) {
    TRACE(executable, attribute, value);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_executable_get_info_fn(executable, attribute, value));
}

hsa_status_t hsa_executable_global_variable_define(hsa_executable_t executable, const char *variable_name, void *address) {
    TRACE(executable, variable_name, address);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_executable_global_variable_define_fn(executable, variable_name, address));
}

hsa_status_t hsa_executable_agent_global_variable_define(hsa_executable_t executable, hsa_agent_t agent, const char *variable_name, void *address) {
    TRACE(executable, agent, variable_name, address);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_executable_agent_global_variable_define_fn(executable, agent, variable_name, address));
}

hsa_status_t hsa_executable_readonly_variable_define(hsa_executable_t executable, hsa_agent_t agent, const char *variable_name, void *address) {
    TRACE(executable, agent, variable_name, address);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_executable_readonly_variable_define_fn(executable, agent, variable_name, address));
}

hsa_status_t hsa_executable_validate(hsa_executable_t executable, uint32_t *result) {
    TRACE(executable, result);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_executable_validate_fn(executable, result));
}

hsa_status_t hsa_executable_validate_alt(hsa_executable_t executable, const char *options, uint32_t *result) {
    TRACE(executable, options, result);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_executable_validate_alt_fn(executable, options, result));
}

/* deprecated */
hsa_status_t hsa_executable_get_symbol(hsa_executable_t executable, const char *module_name, const char *symbol_name, hsa_agent_t agent, int32_t call_convention, hsa_executable_symbol_t *symbol) {
    TRACE(executable, module_name, symbol_name, agent, call_convention, symbol);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_executable_get_symbol_fn(executable, module_name, symbol_name, agent, call_convention, symbol));
}

hsa_status_t hsa_executable_get_symbol_by_name(hsa_executable_t executable, const char *symbol_name, const hsa_agent_t *agent, hsa_executable_symbol_t *symbol) {
    TRACE(executable, symbol_name, agent, symbol);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_executable_get_symbol_by_name_fn(executable, symbol_name, agent, symbol));
}

hsa_status_t hsa_executable_symbol_get_info(hsa_executable_symbol_t executable_symbol, hsa_executable_symbol_info_t attribute, void *value) {
    TRACE(executable_symbol, attribute, value);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_executable_symbol_get_info_fn(executable_symbol, attribute, value));
}

/* deprecated */
hsa_status_t hsa_executable_iterate_symbols(hsa_executable_t executable, hsa_status_t (*callback)(hsa_executable_t executable, hsa_executable_symbol_t symbol, void *data), void *data) {
    TRACE(executable, callback, data);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_executable_iterate_symbols_fn(executable, callback, data));
}

hsa_status_t hsa_executable_iterate_agent_symbols(hsa_executable_t executable, hsa_agent_t agent, hsa_status_t (*callback)(hsa_executable_t exec, hsa_agent_t agent, hsa_executable_symbol_t symbol, void *data), void *data) {
    TRACE(executable, agent, callback, data);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_executable_iterate_agent_symbols_fn(executable, agent, callback, data));
}

hsa_status_t hsa_executable_iterate_program_symbols(hsa_executable_t executable, hsa_status_t (*callback)(hsa_executable_t exec, hsa_executable_symbol_t symbol, void *data), void *data) {
    TRACE(executable, callback, data);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_executable_iterate_program_symbols_fn(executable, callback, data));
}

//===--- Runtime Notifications --------------------------------------------===//

hsa_status_t hsa_status_string(hsa_status_t status, const char **status_string) {
    TRACE(status, status_string);
    return LOG_STATUS(gs_OrigCoreApiTable.hsa_status_string_fn(status, status_string));
}

#define FN_NAME(name) name ## _fn
#define MINE_OR_THEIRS(name) table->FN_NAME(name) = gs_hsa_enabled_map[ #name ] ? RTG::name : gs_OrigCoreApiTable.FN_NAME(name)

static void InitCoreApiTable(CoreApiTable* table) {
    // Initialize function pointers for Hsa Core Runtime Api's
    MINE_OR_THEIRS(hsa_init);
    //table->hsa_shut_down_fn = RTG::hsa_shut_down;
    MINE_OR_THEIRS(hsa_system_get_info);
    MINE_OR_THEIRS(hsa_system_extension_supported);
    MINE_OR_THEIRS(hsa_system_get_extension_table);
    MINE_OR_THEIRS(hsa_iterate_agents);
    MINE_OR_THEIRS(hsa_agent_get_info);
    MINE_OR_THEIRS(hsa_agent_get_exception_policies);
    MINE_OR_THEIRS(hsa_agent_extension_supported);
    table->FN_NAME(hsa_queue_create) = RTG::hsa_queue_create; // always mine, needed for profiling
    MINE_OR_THEIRS(hsa_soft_queue_create);
    MINE_OR_THEIRS(hsa_queue_destroy);
    MINE_OR_THEIRS(hsa_queue_inactivate);
    MINE_OR_THEIRS(hsa_queue_load_read_index_scacquire);
    MINE_OR_THEIRS(hsa_queue_load_read_index_relaxed);
    MINE_OR_THEIRS(hsa_queue_load_write_index_scacquire);
    MINE_OR_THEIRS(hsa_queue_load_write_index_relaxed);
    MINE_OR_THEIRS(hsa_queue_store_write_index_relaxed);
    MINE_OR_THEIRS(hsa_queue_store_write_index_screlease);
    MINE_OR_THEIRS(hsa_queue_cas_write_index_scacq_screl);
    MINE_OR_THEIRS(hsa_queue_cas_write_index_scacquire);
    MINE_OR_THEIRS(hsa_queue_cas_write_index_relaxed);
    MINE_OR_THEIRS(hsa_queue_cas_write_index_screlease);
    MINE_OR_THEIRS(hsa_queue_add_write_index_scacq_screl);
    MINE_OR_THEIRS(hsa_queue_add_write_index_scacquire);
    MINE_OR_THEIRS(hsa_queue_add_write_index_relaxed);
    MINE_OR_THEIRS(hsa_queue_add_write_index_screlease);
    MINE_OR_THEIRS(hsa_queue_store_read_index_relaxed);
    MINE_OR_THEIRS(hsa_queue_store_read_index_screlease);
    MINE_OR_THEIRS(hsa_agent_iterate_regions);
    MINE_OR_THEIRS(hsa_region_get_info);
    MINE_OR_THEIRS(hsa_memory_register);
    MINE_OR_THEIRS(hsa_memory_deregister);
    MINE_OR_THEIRS(hsa_memory_allocate);
    MINE_OR_THEIRS(hsa_memory_free);
    MINE_OR_THEIRS(hsa_memory_copy);
    MINE_OR_THEIRS(hsa_memory_assign_agent);
    MINE_OR_THEIRS(hsa_signal_create);
    MINE_OR_THEIRS(hsa_signal_destroy);
    MINE_OR_THEIRS(hsa_signal_load_relaxed);
    MINE_OR_THEIRS(hsa_signal_load_scacquire);
    MINE_OR_THEIRS(hsa_signal_store_relaxed);
    MINE_OR_THEIRS(hsa_signal_store_screlease);
    MINE_OR_THEIRS(hsa_signal_wait_relaxed);
    MINE_OR_THEIRS(hsa_signal_wait_scacquire);
    MINE_OR_THEIRS(hsa_signal_and_relaxed);
    MINE_OR_THEIRS(hsa_signal_and_scacquire);
    MINE_OR_THEIRS(hsa_signal_and_screlease);
    MINE_OR_THEIRS(hsa_signal_and_scacq_screl);
    MINE_OR_THEIRS(hsa_signal_or_relaxed);
    MINE_OR_THEIRS(hsa_signal_or_scacquire);
    MINE_OR_THEIRS(hsa_signal_or_screlease);
    MINE_OR_THEIRS(hsa_signal_or_scacq_screl);
    MINE_OR_THEIRS(hsa_signal_xor_relaxed);
    MINE_OR_THEIRS(hsa_signal_xor_scacquire);
    MINE_OR_THEIRS(hsa_signal_xor_screlease);
    MINE_OR_THEIRS(hsa_signal_xor_scacq_screl);
    MINE_OR_THEIRS(hsa_signal_exchange_relaxed);
    MINE_OR_THEIRS(hsa_signal_exchange_scacquire);
    MINE_OR_THEIRS(hsa_signal_exchange_screlease);
    MINE_OR_THEIRS(hsa_signal_exchange_scacq_screl);
    MINE_OR_THEIRS(hsa_signal_add_relaxed);
    MINE_OR_THEIRS(hsa_signal_add_scacquire);
    MINE_OR_THEIRS(hsa_signal_add_screlease);
    MINE_OR_THEIRS(hsa_signal_add_scacq_screl);
    MINE_OR_THEIRS(hsa_signal_subtract_relaxed);
    MINE_OR_THEIRS(hsa_signal_subtract_scacquire);
    MINE_OR_THEIRS(hsa_signal_subtract_screlease);
    MINE_OR_THEIRS(hsa_signal_subtract_scacq_screl);
    MINE_OR_THEIRS(hsa_signal_cas_relaxed);
    MINE_OR_THEIRS(hsa_signal_cas_scacquire);
    MINE_OR_THEIRS(hsa_signal_cas_screlease);
    MINE_OR_THEIRS(hsa_signal_cas_scacq_screl);

    //===--- Instruction Set Architecture -----------------------------------===//

    MINE_OR_THEIRS(hsa_isa_from_name);
    // Deprecated since v1.1.
    MINE_OR_THEIRS(hsa_isa_get_info);
    // Deprecated since v1.1.
    MINE_OR_THEIRS(hsa_isa_compatible);

    //===--- Code Objects (deprecated) --------------------------------------===//

    // Deprecated since v1.1.
    MINE_OR_THEIRS(hsa_code_object_serialize);
    // Deprecated since v1.1.
    MINE_OR_THEIRS(hsa_code_object_deserialize);
    // Deprecated since v1.1.
    MINE_OR_THEIRS(hsa_code_object_destroy);
    // Deprecated since v1.1.
    MINE_OR_THEIRS(hsa_code_object_get_info);
    // Deprecated since v1.1.
    MINE_OR_THEIRS(hsa_code_object_get_symbol);
    // Deprecated since v1.1.
    MINE_OR_THEIRS(hsa_code_symbol_get_info);
    // Deprecated since v1.1.
    MINE_OR_THEIRS(hsa_code_object_iterate_symbols);

    //===--- Executable -----------------------------------------------------===//

    // Deprecated since v1.1.
    MINE_OR_THEIRS(hsa_executable_create);
    MINE_OR_THEIRS(hsa_executable_destroy);
    // Deprecated since v1.1.
    MINE_OR_THEIRS(hsa_executable_load_code_object);
    table->FN_NAME(hsa_executable_freeze) = RTG::hsa_executable_freeze; // always mine, needed for profiling
    MINE_OR_THEIRS(hsa_executable_get_info);
    MINE_OR_THEIRS(hsa_executable_global_variable_define);
    MINE_OR_THEIRS(hsa_executable_agent_global_variable_define);
    MINE_OR_THEIRS(hsa_executable_readonly_variable_define);
    MINE_OR_THEIRS(hsa_executable_validate);
    // Deprecated since v1.1.
    MINE_OR_THEIRS(hsa_executable_get_symbol);
    MINE_OR_THEIRS(hsa_executable_symbol_get_info);
    // Deprecated since v1.1.
    MINE_OR_THEIRS(hsa_executable_iterate_symbols);

    //===--- Runtime Notifications ------------------------------------------===//

    MINE_OR_THEIRS(hsa_status_string);

    // Start HSA v1.1 additions
    MINE_OR_THEIRS(hsa_extension_get_name);
    MINE_OR_THEIRS(hsa_system_major_extension_supported);
    MINE_OR_THEIRS(hsa_system_get_major_extension_table);
    MINE_OR_THEIRS(hsa_agent_major_extension_supported);
    MINE_OR_THEIRS(hsa_cache_get_info);
    MINE_OR_THEIRS(hsa_agent_iterate_caches);
    // Silent store optimization is present in all signal ops when no agents are sleeping.
    MINE_OR_THEIRS(hsa_signal_silent_store_relaxed);
    MINE_OR_THEIRS(hsa_signal_silent_store_screlease);
    MINE_OR_THEIRS(hsa_signal_group_create);
    MINE_OR_THEIRS(hsa_signal_group_destroy);
    MINE_OR_THEIRS(hsa_signal_group_wait_any_scacquire);
    MINE_OR_THEIRS(hsa_signal_group_wait_any_relaxed);

    //===--- Instruction Set Architecture - HSA v1.1 additions --------------===//

    MINE_OR_THEIRS(hsa_agent_iterate_isas);
    MINE_OR_THEIRS(hsa_isa_get_info_alt);
    MINE_OR_THEIRS(hsa_isa_get_exception_policies);
    MINE_OR_THEIRS(hsa_isa_get_round_method);
    MINE_OR_THEIRS(hsa_wavefront_get_info);
    MINE_OR_THEIRS(hsa_isa_iterate_wavefronts);

    //===--- Code Objects (deprecated) - HSA v1.1 additions -----------------===//

    // Deprecated since v1.1.
    MINE_OR_THEIRS(hsa_code_object_get_symbol_from_name);

    //===--- Executable - HSA v1.1 additions --------------------------------===//

    MINE_OR_THEIRS(hsa_code_object_reader_create_from_file);
    MINE_OR_THEIRS(hsa_code_object_reader_create_from_memory);
    MINE_OR_THEIRS(hsa_code_object_reader_destroy);
    MINE_OR_THEIRS(hsa_executable_create_alt);
    MINE_OR_THEIRS(hsa_executable_load_program_code_object);
    MINE_OR_THEIRS(hsa_executable_load_agent_code_object);
    MINE_OR_THEIRS(hsa_executable_validate_alt);
    MINE_OR_THEIRS(hsa_executable_get_symbol_by_name);
    MINE_OR_THEIRS(hsa_executable_iterate_agent_symbols);
    MINE_OR_THEIRS(hsa_executable_iterate_program_symbols);
}

#undef FN_NAME
#undef MINE_OR_THEIRS

/*
 * Following set of functions are bundled as AMD Extension Apis
 */

// Pass through stub functions
hsa_status_t hsa_amd_coherency_get_type(hsa_agent_t agent, hsa_amd_coherency_type_t* type) {
    TRACE(agent, type);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_coherency_get_type_fn(agent, type));
}

// Pass through stub functions
hsa_status_t hsa_amd_coherency_set_type(hsa_agent_t agent, hsa_amd_coherency_type_t type) {
    TRACE(agent, type);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_coherency_set_type_fn(agent, type));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_profiling_set_profiler_enabled(hsa_queue_t* queue, int enable) {
    TRACE(queue, enable);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_profiling_set_profiler_enabled_fn(queue, enable));
}

hsa_status_t hsa_amd_profiling_async_copy_enable(bool enable) {
    TRACE(enable);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_profiling_async_copy_enable_fn(enable));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_profiling_get_dispatch_time(hsa_agent_t agent, hsa_signal_t signal, hsa_amd_profiling_dispatch_time_t* time) {
    TRACE(agent, signal, time);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_profiling_get_dispatch_time_fn(agent, signal, time));
}

hsa_status_t hsa_amd_profiling_get_async_copy_time(hsa_signal_t hsa_signal, hsa_amd_profiling_async_copy_time_t* time) {
    TRACE(hsa_signal, time);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_profiling_get_async_copy_time_fn(hsa_signal, time));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_profiling_convert_tick_to_system_domain(hsa_agent_t agent, uint64_t agent_tick, uint64_t* system_tick) {
    TRACE(agent, agent_tick, system_tick);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_profiling_convert_tick_to_system_domain_fn(agent, agent_tick, system_tick));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_signal_async_handler(hsa_signal_t signal, hsa_signal_condition_t cond, hsa_signal_value_t value, hsa_amd_signal_handler handler, void* arg) {
    TRACE(signal, cond, value, handler, arg);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_signal_async_handler_fn(signal, cond, value, handler, arg));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_async_function(void (*callback)(void* arg), void* arg) {
    TRACE(callback, arg);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_async_function_fn(callback, arg));
}

// Mirrors Amd Extension Apis
uint32_t hsa_amd_signal_wait_any(uint32_t signal_count, hsa_signal_t* signals, hsa_signal_condition_t* conds, hsa_signal_value_t* values, uint64_t timeout_hint, hsa_wait_state_t wait_hint, hsa_signal_value_t* satisfying_value) {
    TRACE(signal_count, signals, conds, values, timeout_hint, wait_hint, satisfying_value);
    return LOG_UINT32(gs_OrigExtApiTable.hsa_amd_signal_wait_any_fn(signal_count, signals, conds, values, timeout_hint, wait_hint, satisfying_value));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_queue_cu_set_mask(const hsa_queue_t* queue, uint32_t num_cu_mask_count, const uint32_t* cu_mask) {
    TRACE(queue, num_cu_mask_count, cu_mask);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_queue_cu_set_mask_fn(queue, num_cu_mask_count, cu_mask));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_memory_pool_get_info(hsa_amd_memory_pool_t memory_pool, hsa_amd_memory_pool_info_t attribute, void* value) {
    TRACE(memory_pool, attribute, value);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_memory_pool_get_info_fn(memory_pool, attribute, value));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_agent_iterate_memory_pools(hsa_agent_t agent, hsa_status_t (*callback)(hsa_amd_memory_pool_t memory_pool, void* data), void* data) {
    TRACE(agent, callback, data);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_agent_iterate_memory_pools_fn(agent, callback, data));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t memory_pool, size_t size, uint32_t flags, void** ptr) {
    TRACE(memory_pool, size, flags, ptr);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_memory_pool_allocate_fn(memory_pool, size, flags, ptr));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_memory_pool_free(void* ptr) {
    TRACE(ptr);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_memory_pool_free_fn(ptr));
}

inline hsa_device_type_t agent_type(hsa_agent_t x)
{
    hsa_device_type_t r{};
    gs_OrigCoreApiTable.hsa_agent_get_info_fn(x, HSA_AGENT_INFO_DEVICE, &r);
    return r;
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_memory_async_copy(void* dst, hsa_agent_t dst_agent, const void* src, hsa_agent_t src_agent, size_t size, uint32_t num_dep_signals, const hsa_signal_t* dep_signals, hsa_signal_t completion_signal) {
    ++RTG::gs_host_count_copies;
    if (RTG_PROFILE_COPY) {
        if (completion_signal.handle == 0) {
            fprintf(stderr, "RTG Tracer: hsa_amd_memory_async_copy no signal\n");
        }
        else {
            hsa_agent_t agent_to_use;
            int direction;
            auto src_type = agent_type(src_agent);
            auto dst_type = agent_type(dst_agent);
            if (src_type == HSA_DEVICE_TYPE_CPU && dst_type == HSA_DEVICE_TYPE_CPU) {
                fprintf(stderr, "RTG Tracer: hsa_amd_memory_async_copy used with two cpu agents\n");
                exit(EXIT_FAILURE);
            }
            else if (src_type == HSA_DEVICE_TYPE_CPU && dst_type != HSA_DEVICE_TYPE_CPU) {
                direction = H2D;
                agent_to_use = dst_agent;
            }
            else if (src_type != HSA_DEVICE_TYPE_CPU && dst_type == HSA_DEVICE_TYPE_CPU) {
                direction = D2H;
                agent_to_use = src_agent;
            }
            else {
                direction = D2D;
                agent_to_use = src_agent;
            }
            hsa_signal_t new_signal = CreateSignal();
            hsa_signal_t deleter_signal = CreateSignal();
            hsa_queue_t *queue = AgentInfo::Get(agent_to_use)->get_signal_queue();
            uint64_t index = submit_to_signal_queue(queue, new_signal, completion_signal);
            submit_to_signal_queue(queue, gs_NullSignal, deleter_signal);
            gs_pool.push(SignalWaiter, new SignalCallbackData(queue, agent_to_use, deleter_signal, new_signal, completion_signal, false, num_dep_signals, dep_signals, size, direction, index));
            TRACE(dst, dst_agent, src, src_agent, size, num_dep_signals, dep_signals, completion_signal);
            return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_memory_async_copy_fn(dst, dst_agent, src, src_agent, size, num_dep_signals, dep_signals, new_signal));
        }
    }
    // print as usual
    TRACE(dst, dst_agent, src, src_agent, size, num_dep_signals, dep_signals, completion_signal);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_memory_async_copy_fn(dst, dst_agent, src, src_agent, size, num_dep_signals, dep_signals, completion_signal));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_memory_async_copy_rect(const hsa_pitched_ptr_t* dst, const hsa_dim3_t* dst_offset, const hsa_pitched_ptr_t* src, const hsa_dim3_t* src_offset, const hsa_dim3_t* range, hsa_agent_t copy_agent, hsa_amd_copy_direction_t dir, uint32_t num_dep_signals, const hsa_signal_t* dep_signals, hsa_signal_t completion_signal) {
    ++RTG::gs_host_count_copies;
    if (RTG_PROFILE_COPY) {
        if (completion_signal.handle == 0) {
            fprintf(stderr, "RTG Tracer: hsa_amd_memory_async_copy_rect no signal\n");
        }
        else {
            if (agent_type(copy_agent) != HSA_DEVICE_TYPE_CPU) {
                hsa_signal_t new_signal = CreateSignal();
                hsa_signal_t deleter_signal = CreateSignal();
                hsa_queue_t *queue = AgentInfo::Get(copy_agent)->get_signal_queue();
                uint64_t index = submit_to_signal_queue(queue, new_signal, completion_signal);
                submit_to_signal_queue(queue, gs_NullSignal, deleter_signal);
                gs_pool.push(SignalWaiter, new SignalCallbackData(queue, copy_agent, deleter_signal, new_signal, completion_signal, false, num_dep_signals, dep_signals, range->x*range->y, dir, index));
                TRACE(dst, dst_offset, src, src_offset, range, copy_agent, dir, num_dep_signals, dep_signals, completion_signal);
                return gs_OrigExtApiTable.hsa_amd_memory_async_copy_rect_fn(dst, dst_offset, src, src_offset, range, copy_agent, dir, num_dep_signals, dep_signals, new_signal);
            }
        }
    }
    // print as usual
    TRACE(dst, dst_offset, src, src_offset, range, copy_agent, dir, num_dep_signals, dep_signals, completion_signal);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_memory_async_copy_rect_fn(dst, dst_offset, src, src_offset, range, copy_agent, dir, num_dep_signals, dep_signals, completion_signal));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_agent_memory_pool_get_info(hsa_agent_t agent, hsa_amd_memory_pool_t memory_pool, hsa_amd_agent_memory_pool_info_t attribute, void* value) {
    TRACE(agent, memory_pool, attribute, value);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_agent_memory_pool_get_info_fn(agent, memory_pool, attribute, value));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_agents_allow_access(uint32_t num_agents, const hsa_agent_t* agents, const uint32_t* flags, const void* ptr) {
    TRACE(num_agents, agents, flags, ptr);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_agents_allow_access_fn(num_agents, agents, flags, ptr));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_memory_pool_can_migrate(hsa_amd_memory_pool_t src_memory_pool, hsa_amd_memory_pool_t dst_memory_pool, bool* result) {
    TRACE(src_memory_pool, dst_memory_pool, result);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_memory_pool_can_migrate_fn(src_memory_pool, dst_memory_pool, result));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_memory_migrate(const void* ptr, hsa_amd_memory_pool_t memory_pool, uint32_t flags) {
    TRACE(ptr, memory_pool, flags);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_memory_migrate_fn(ptr, memory_pool, flags));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_memory_lock(void* host_ptr, size_t size, hsa_agent_t* agents, int num_agent, void** agent_ptr) {
    TRACE(host_ptr, size, agent_ptr, num_agent, agent_ptr);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_memory_lock_fn(host_ptr, size, agents, num_agent, agent_ptr));
}

#if RTG_ENABLE_HSA_AMD_MEMORY_LOCK_TO_POOL
// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_memory_lock_to_pool(void* host_ptr, size_t size, hsa_agent_t* agents, int num_agent, hsa_amd_memory_pool_t pool, uint32_t flags, void** agent_ptr) {
    TRACE(host_ptr, size, agent_ptr, num_agent, pool, flags, agent_ptr);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_memory_lock_to_pool_fn(host_ptr, size, agents, num_agent, pool, flags, agent_ptr));
}
#endif

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_register_deallocation_callback(void* ptr, hsa_amd_deallocation_callback_t callback, void* user_data) {
    TRACE(ptr, callback, user_data);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_register_deallocation_callback_fn(ptr, callback, user_data));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_deregister_deallocation_callback(void* ptr, hsa_amd_deallocation_callback_t callback) {
    TRACE(ptr, callback);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_deregister_deallocation_callback_fn(ptr, callback));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_signal_value_pointer(hsa_signal_t signal, volatile hsa_signal_value_t** value_ptr) {
    TRACE(signal, value_ptr);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_signal_value_pointer_fn(signal, value_ptr));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_svm_attributes_set(void* ptr, size_t size, hsa_amd_svm_attribute_pair_t* attribute_list, size_t attribute_count) {
    TRACE(ptr, size, attribute_list, attribute_count);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_svm_attributes_set_fn(ptr, size, attribute_list, attribute_count));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_svm_attributes_get(void* ptr, size_t size, hsa_amd_svm_attribute_pair_t* attribute_list, size_t attribute_count) {
    TRACE(ptr, size, attribute_list, attribute_count);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_svm_attributes_get_fn(ptr, size, attribute_list, attribute_count));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_svm_prefetch_async(void* ptr, size_t size, hsa_agent_t agent, uint32_t num_dep_signals, const hsa_signal_t* dep_signals, hsa_signal_t completion_signal) {
    TRACE(ptr, size, agent, num_dep_signals, dep_signals, completion_signal);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_svm_prefetch_async_fn(ptr, size, agent, num_dep_signals, dep_signals, completion_signal));
}

#if (HIP_VERSION_MAJOR == 4 && HIP_VERSION_MINOR >= 4) || HIP_VERSION_MAJOR > 4
// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_queue_cu_get_mask(const hsa_queue_t* queue, uint32_t num_cu_mask_count, uint32_t* cu_mask) {
    TRACE(queue, num_cu_mask_count, cu_mask);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_queue_cu_get_mask_fn(queue, num_cu_mask_count, cu_mask));
}
#endif

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_memory_unlock(void* host_ptr) {
    TRACE(host_ptr);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_memory_unlock_fn(host_ptr));

}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_memory_fill(void* ptr, uint32_t value, size_t count) {
    TRACE(ptr, value, count);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_memory_fill_fn(ptr, value, count));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_interop_map_buffer(uint32_t num_agents, hsa_agent_t* agents, int interop_handle, uint32_t flags, size_t* size, void** ptr, size_t* metadata_size, const void** metadata) {
    TRACE(num_agents, agents, interop_handle, flags, size, ptr, metadata_size, metadata);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_interop_map_buffer_fn(num_agents, agents, interop_handle, flags, size, ptr, metadata_size, metadata));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_interop_unmap_buffer(void* ptr) {
    TRACE(ptr);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_interop_unmap_buffer_fn(ptr));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_image_create(hsa_agent_t agent, const hsa_ext_image_descriptor_t *image_descriptor, const hsa_amd_image_descriptor_t *image_layout, const void *image_data, hsa_access_permission_t access_permission, hsa_ext_image_t *image) {
    TRACE(agent, image_descriptor, image_layout, image_data, access_permission, image);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_image_create_fn(agent, image_descriptor, image_layout, image_data, access_permission, image));
}

// Mirrors Amd Extension Apis
#if (HIP_VERSION_MAJOR == 4 && HIP_VERSION_MINOR >= 4) || HIP_VERSION_MAJOR > 4
#define MAYBE_CONST const
#else
#define MAYBE_CONST
#endif
hsa_status_t hsa_amd_pointer_info(MAYBE_CONST void* ptr, hsa_amd_pointer_info_t* info, void* (*alloc)(size_t), uint32_t* num_agents_accessible, hsa_agent_t** accessible) {
    TRACE(ptr, info, alloc, accessible);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_pointer_info_fn(ptr, info, alloc, num_agents_accessible, accessible));
}
#undef MAYBE_CONST

// Mirrors Amd Extension Apis
#if (HIP_VERSION_MAJOR == 4 && HIP_VERSION_MINOR >= 4) || HIP_VERSION_MAJOR > 4
#define MAYBE_CONST const
#else
#define MAYBE_CONST
#endif
hsa_status_t hsa_amd_pointer_info_set_userdata(MAYBE_CONST void* ptr, void* userptr) {
    TRACE(ptr, userptr);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_pointer_info_set_userdata_fn(ptr, userptr));
}
#undef MAYBE_CONST

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_ipc_memory_create(void* ptr, size_t len, hsa_amd_ipc_memory_t* handle) {
    TRACE(ptr, len, handle);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_ipc_memory_create_fn(ptr, len, handle));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_ipc_memory_attach(const hsa_amd_ipc_memory_t* ipc, size_t len, uint32_t num_agents, const hsa_agent_t* mapping_agents, void** mapped_ptr) {
    TRACE(ipc, len, num_agents, mapping_agents, mapped_ptr);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_ipc_memory_attach_fn(ipc, len, num_agents, mapping_agents, mapped_ptr));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_ipc_memory_detach(void* mapped_ptr) {
    TRACE(mapped_ptr);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_ipc_memory_detach_fn(mapped_ptr));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_signal_create(hsa_signal_value_t initial_value, uint32_t num_consumers, const hsa_agent_t* consumers, uint64_t attributes, hsa_signal_t* signal) {
    TRACE(initial_value, num_consumers, consumers, attributes, signal);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_signal_create_fn(initial_value, num_consumers, consumers, attributes, signal));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_ipc_signal_create(hsa_signal_t signal, hsa_amd_ipc_signal_t* handle) {
    TRACE(signal, handle);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_ipc_signal_create_fn(signal, handle));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_ipc_signal_attach(const hsa_amd_ipc_signal_t* handle, hsa_signal_t* signal) {
    TRACE(handle, signal);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_ipc_signal_attach_fn(handle, signal));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_register_system_event_handler(hsa_amd_system_event_callback_t callback, void* data) {
    TRACE(callback, data);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_register_system_event_handler_fn(callback, data));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_queue_intercept_create(hsa_agent_t agent_handle, uint32_t size, hsa_queue_type32_t type, void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data), void* data, uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t** queue) {
    TRACE(agent_handle, size, type, callback, data, private_segment_size, group_segment_size, queue);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_queue_intercept_create_fn(agent_handle, size, type, callback, data, private_segment_size, group_segment_size, queue));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_queue_intercept_register(hsa_queue_t* queue, hsa_amd_queue_intercept_handler callback, void* user_data) {
    TRACE(queue, callback, user_data);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_queue_intercept_register_fn(queue, callback, user_data));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_queue_set_priority(hsa_queue_t* queue, hsa_amd_queue_priority_t priority) {
    TRACE(queue, priority);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_queue_set_priority_fn(queue, priority));
}

#define FN_NAME(name) name ## _fn
#define MINE_OR_THEIRS(name) table->FN_NAME(name) = gs_hsa_enabled_map[ #name ] ? RTG::name : gs_OrigExtApiTable.FN_NAME(name)

static void InitAmdExtTable(AmdExtTable* table) {
    // Initialize function pointers for Amd Extension Api's
    MINE_OR_THEIRS(hsa_amd_coherency_get_type);
    MINE_OR_THEIRS(hsa_amd_coherency_set_type);
    MINE_OR_THEIRS(hsa_amd_profiling_set_profiler_enabled);
    MINE_OR_THEIRS(hsa_amd_profiling_async_copy_enable);
    MINE_OR_THEIRS(hsa_amd_profiling_get_dispatch_time);
    MINE_OR_THEIRS(hsa_amd_profiling_get_async_copy_time);
    MINE_OR_THEIRS(hsa_amd_profiling_convert_tick_to_system_domain);
    MINE_OR_THEIRS(hsa_amd_signal_async_handler);
    MINE_OR_THEIRS(hsa_amd_async_function);
    MINE_OR_THEIRS(hsa_amd_signal_wait_any);
    MINE_OR_THEIRS(hsa_amd_queue_cu_set_mask);
    MINE_OR_THEIRS(hsa_amd_memory_pool_get_info);
    MINE_OR_THEIRS(hsa_amd_agent_iterate_memory_pools);
    MINE_OR_THEIRS(hsa_amd_memory_pool_allocate);
    MINE_OR_THEIRS(hsa_amd_memory_pool_free);
    MINE_OR_THEIRS(hsa_amd_memory_async_copy);
    MINE_OR_THEIRS(hsa_amd_agent_memory_pool_get_info);
    MINE_OR_THEIRS(hsa_amd_agents_allow_access);
    MINE_OR_THEIRS(hsa_amd_memory_pool_can_migrate);
    MINE_OR_THEIRS(hsa_amd_memory_migrate);
    MINE_OR_THEIRS(hsa_amd_memory_lock);
    MINE_OR_THEIRS(hsa_amd_memory_unlock);
    MINE_OR_THEIRS(hsa_amd_memory_fill);
    MINE_OR_THEIRS(hsa_amd_interop_map_buffer);
    MINE_OR_THEIRS(hsa_amd_interop_unmap_buffer);
    MINE_OR_THEIRS(hsa_amd_image_create);
    MINE_OR_THEIRS(hsa_amd_pointer_info);
    MINE_OR_THEIRS(hsa_amd_pointer_info_set_userdata);
    MINE_OR_THEIRS(hsa_amd_ipc_memory_create);
    MINE_OR_THEIRS(hsa_amd_ipc_memory_attach);
    MINE_OR_THEIRS(hsa_amd_ipc_memory_detach);
    MINE_OR_THEIRS(hsa_amd_signal_create);
    MINE_OR_THEIRS(hsa_amd_ipc_signal_create);
    MINE_OR_THEIRS(hsa_amd_ipc_signal_attach);
    MINE_OR_THEIRS(hsa_amd_register_system_event_handler);
    MINE_OR_THEIRS(hsa_amd_queue_intercept_create);
    MINE_OR_THEIRS(hsa_amd_queue_intercept_register);
    MINE_OR_THEIRS(hsa_amd_queue_set_priority);
    MINE_OR_THEIRS(hsa_amd_memory_async_copy_rect);
#if RTG_ENABLE_HSA_AMD_RUNTIME_QUEUE_CREATE_REGISTER
    MINE_OR_THEIRS(hsa_amd_runtime_queue_create_register);
#endif
#if RTG_ENABLE_HSA_AMD_MEMORY_LOCK_TO_POOL
    MINE_OR_THEIRS(hsa_amd_memory_lock_to_pool);
#endif
    MINE_OR_THEIRS(hsa_amd_register_deallocation_callback);
    MINE_OR_THEIRS(hsa_amd_deregister_deallocation_callback);
    MINE_OR_THEIRS(hsa_amd_signal_value_pointer);
    MINE_OR_THEIRS(hsa_amd_svm_attributes_set);
    MINE_OR_THEIRS(hsa_amd_svm_attributes_get);
    MINE_OR_THEIRS(hsa_amd_svm_prefetch_async);
#if (HIP_VERSION_MAJOR == 4 && HIP_VERSION_MINOR >= 4) || HIP_VERSION_MAJOR > 4
    MINE_OR_THEIRS(hsa_amd_queue_cu_get_mask);
#endif
}

static std::vector<std::string> split(const std::string& s, char delimiter)
{
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);
   while (std::getline(tokenStream, token, delimiter)) {
      tokens.push_back(token);
   }
   return tokens;
}

static void InitEnabledTable(std::string what_to_trace, std::string what_not_to_trace) {
    // init all keys to false initially
    InitEnabledTableCore(false);
    InitEnabledTableExtApi(false);

    // tokens given by user
    std::vector<std::string> tokens = split(what_to_trace, ',');
    for (auto s : tokens) {
        // special tokens "all", "core", and "ext"
        if ("all" == s || "*" == s) {
            InitEnabledTableCore(true);
            InitEnabledTableExtApi(true);
        }
        else if ("core" == s) {
            InitEnabledTableCore(true);
        }
        else if ("ext" == s) {
            InitEnabledTableExtApi(true);
        }
        else {
            // otherwise, go through entire map and look for matches
            for (auto &&kv : gs_hsa_enabled_map) {
                if (kv.first.find(s) != std::string::npos) {
                    kv.second = true;
                }
            }
        }
    }
    tokens = split(what_not_to_trace, ',');
    for (auto s : tokens) {
        for (auto &&kv : gs_hsa_enabled_map) {
            if (kv.first.find(s) != std::string::npos) {
                kv.second = false;
            }
        }
    }
}

static void InitEnabledTableCore(bool value) {
    // Initialize function pointers for Hsa Core Runtime Api's
    gs_hsa_enabled_map["hsa_init"] = value;
    //gs_hsa_enabled_map["hsa_shut_down"] = value;
    gs_hsa_enabled_map["hsa_system_get_info"] = value;
    gs_hsa_enabled_map["hsa_system_extension_supported"] = value;
    gs_hsa_enabled_map["hsa_system_get_extension_table"] = value;
    gs_hsa_enabled_map["hsa_iterate_agents"] = value;
    gs_hsa_enabled_map["hsa_agent_get_info"] = value;
    gs_hsa_enabled_map["hsa_agent_get_exception_policies"] = value;
    gs_hsa_enabled_map["hsa_agent_extension_supported"] = value;
    gs_hsa_enabled_map["hsa_queue_create"] = value;
    gs_hsa_enabled_map["hsa_soft_queue_create"] = value;
    gs_hsa_enabled_map["hsa_queue_destroy"] = value;
    gs_hsa_enabled_map["hsa_queue_inactivate"] = value;
    gs_hsa_enabled_map["hsa_queue_load_read_index_scacquire"] = value;
    gs_hsa_enabled_map["hsa_queue_load_read_index_relaxed"] = value;
    gs_hsa_enabled_map["hsa_queue_load_write_index_scacquire"] = value;
    gs_hsa_enabled_map["hsa_queue_load_write_index_relaxed"] = value;
    gs_hsa_enabled_map["hsa_queue_store_write_index_relaxed"] = value;
    gs_hsa_enabled_map["hsa_queue_store_write_index_screlease"] = value;
    gs_hsa_enabled_map["hsa_queue_cas_write_index_scacq_screl"] = value;
    gs_hsa_enabled_map["hsa_queue_cas_write_index_scacquire"] = value;
    gs_hsa_enabled_map["hsa_queue_cas_write_index_relaxed"] = value;
    gs_hsa_enabled_map["hsa_queue_cas_write_index_screlease"] = value;
    gs_hsa_enabled_map["hsa_queue_add_write_index_scacq_screl"] = value;
    gs_hsa_enabled_map["hsa_queue_add_write_index_scacquire"] = value;
    gs_hsa_enabled_map["hsa_queue_add_write_index_relaxed"] = value;
    gs_hsa_enabled_map["hsa_queue_add_write_index_screlease"] = value;
    gs_hsa_enabled_map["hsa_queue_store_read_index_relaxed"] = value;
    gs_hsa_enabled_map["hsa_queue_store_read_index_screlease"] = value;
    gs_hsa_enabled_map["hsa_agent_iterate_regions"] = value;
    gs_hsa_enabled_map["hsa_region_get_info"] = value;
    gs_hsa_enabled_map["hsa_memory_register"] = value;
    gs_hsa_enabled_map["hsa_memory_deregister"] = value;
    gs_hsa_enabled_map["hsa_memory_allocate"] = value;
    gs_hsa_enabled_map["hsa_memory_free"] = value;
    gs_hsa_enabled_map["hsa_memory_copy"] = value;
    gs_hsa_enabled_map["hsa_memory_assign_agent"] = value;
    gs_hsa_enabled_map["hsa_signal_create"] = value;
    gs_hsa_enabled_map["hsa_signal_destroy"] = value;
    gs_hsa_enabled_map["hsa_signal_load_relaxed"] = value;
    gs_hsa_enabled_map["hsa_signal_load_scacquire"] = value;
    gs_hsa_enabled_map["hsa_signal_store_relaxed"] = value;
    gs_hsa_enabled_map["hsa_signal_store_screlease"] = value;
    gs_hsa_enabled_map["hsa_signal_wait_relaxed"] = value;
    gs_hsa_enabled_map["hsa_signal_wait_scacquire"] = value;
    gs_hsa_enabled_map["hsa_signal_and_relaxed"] = value;
    gs_hsa_enabled_map["hsa_signal_and_scacquire"] = value;
    gs_hsa_enabled_map["hsa_signal_and_screlease"] = value;
    gs_hsa_enabled_map["hsa_signal_and_scacq_screl"] = value;
    gs_hsa_enabled_map["hsa_signal_or_relaxed"] = value;
    gs_hsa_enabled_map["hsa_signal_or_scacquire"] = value;
    gs_hsa_enabled_map["hsa_signal_or_screlease"] = value;
    gs_hsa_enabled_map["hsa_signal_or_scacq_screl"] = value;
    gs_hsa_enabled_map["hsa_signal_xor_relaxed"] = value;
    gs_hsa_enabled_map["hsa_signal_xor_scacquire"] = value;
    gs_hsa_enabled_map["hsa_signal_xor_screlease"] = value;
    gs_hsa_enabled_map["hsa_signal_xor_scacq_screl"] = value;
    gs_hsa_enabled_map["hsa_signal_exchange_relaxed"] = value;
    gs_hsa_enabled_map["hsa_signal_exchange_scacquire"] = value;
    gs_hsa_enabled_map["hsa_signal_exchange_screlease"] = value;
    gs_hsa_enabled_map["hsa_signal_exchange_scacq_screl"] = value;
    gs_hsa_enabled_map["hsa_signal_add_relaxed"] = value;
    gs_hsa_enabled_map["hsa_signal_add_scacquire"] = value;
    gs_hsa_enabled_map["hsa_signal_add_screlease"] = value;
    gs_hsa_enabled_map["hsa_signal_add_scacq_screl"] = value;
    gs_hsa_enabled_map["hsa_signal_subtract_relaxed"] = value;
    gs_hsa_enabled_map["hsa_signal_subtract_scacquire"] = value;
    gs_hsa_enabled_map["hsa_signal_subtract_screlease"] = value;
    gs_hsa_enabled_map["hsa_signal_subtract_scacq_screl"] = value;
    gs_hsa_enabled_map["hsa_signal_cas_relaxed"] = value;
    gs_hsa_enabled_map["hsa_signal_cas_scacquire"] = value;
    gs_hsa_enabled_map["hsa_signal_cas_screlease"] = value;
    gs_hsa_enabled_map["hsa_signal_cas_scacq_screl"] = value;

    //===--- Instruction Set Architecture -----------------------------------===//

    gs_hsa_enabled_map["hsa_isa_from_name"] = value;
    // Deprecated since v1.1.
    gs_hsa_enabled_map["hsa_isa_get_info"] = value;
    // Deprecated since v1.1.
    gs_hsa_enabled_map["hsa_isa_compatible"] = value;

    //===--- Code Objects (deprecated) --------------------------------------===//

    // Deprecated since v1.1.
    gs_hsa_enabled_map["hsa_code_object_serialize"] = value;
    // Deprecated since v1.1.
    gs_hsa_enabled_map["hsa_code_object_deserialize"] = value;
    // Deprecated since v1.1.
    gs_hsa_enabled_map["hsa_code_object_destroy"] = value;
    // Deprecated since v1.1.
    gs_hsa_enabled_map["hsa_code_object_get_info"] = value;
    // Deprecated since v1.1.
    gs_hsa_enabled_map["hsa_code_object_get_symbol"] = value;
    // Deprecated since v1.1.
    gs_hsa_enabled_map["hsa_code_symbol_get_info"] = value;
    // Deprecated since v1.1.
    gs_hsa_enabled_map["hsa_code_object_iterate_symbols"] = value;

    //===--- Executable -----------------------------------------------------===//

    // Deprecated since v1.1.
    gs_hsa_enabled_map["hsa_executable_create"] = value;
    gs_hsa_enabled_map["hsa_executable_destroy"] = value;
    // Deprecated since v1.1.
    gs_hsa_enabled_map["hsa_executable_load_code_object"] = value;
    gs_hsa_enabled_map["hsa_executable_freeze"] = value;
    gs_hsa_enabled_map["hsa_executable_get_info"] = value;
    gs_hsa_enabled_map["hsa_executable_global_variable_define"] = value;
    gs_hsa_enabled_map["hsa_executable_agent_global_variable_define"] = value;
    gs_hsa_enabled_map["hsa_executable_readonly_variable_define"] = value;
    gs_hsa_enabled_map["hsa_executable_validate"] = value;
    // Deprecated since v1.1.
    gs_hsa_enabled_map["hsa_executable_get_symbol"] = value;
    gs_hsa_enabled_map["hsa_executable_symbol_get_info"] = value;
    // Deprecated since v1.1.
    gs_hsa_enabled_map["hsa_executable_iterate_symbols"] = value;

    //===--- Runtime Notifications ------------------------------------------===//

    gs_hsa_enabled_map["hsa_status_string"] = value;

    // Start HSA v1.1 additions
    gs_hsa_enabled_map["hsa_extension_get_name"] = value;
    gs_hsa_enabled_map["hsa_system_major_extension_supported"] = value;
    gs_hsa_enabled_map["hsa_system_get_major_extension_table"] = value;
    gs_hsa_enabled_map["hsa_agent_major_extension_supported"] = value;
    gs_hsa_enabled_map["hsa_cache_get_info"] = value;
    gs_hsa_enabled_map["hsa_agent_iterate_caches"] = value;
    // Silent store optimization is present in all signal ops when no agents are sleeping.
    gs_hsa_enabled_map["hsa_signal_store_relaxed"] = value;
    gs_hsa_enabled_map["hsa_signal_store_screlease"] = value;
    gs_hsa_enabled_map["hsa_signal_group_create"] = value;
    gs_hsa_enabled_map["hsa_signal_group_destroy"] = value;
    gs_hsa_enabled_map["hsa_signal_group_wait_any_scacquire"] = value;
    gs_hsa_enabled_map["hsa_signal_group_wait_any_relaxed"] = value;

    //===--- Instruction Set Architecture - HSA v1.1 additions --------------===//

    gs_hsa_enabled_map["hsa_agent_iterate_isas"] = value;
    gs_hsa_enabled_map["hsa_isa_get_info_alt"] = value;
    gs_hsa_enabled_map["hsa_isa_get_exception_policies"] = value;
    gs_hsa_enabled_map["hsa_isa_get_round_method"] = value;
    gs_hsa_enabled_map["hsa_wavefront_get_info"] = value;
    gs_hsa_enabled_map["hsa_isa_iterate_wavefronts"] = value;

    //===--- Code Objects (deprecated) - HSA v1.1 additions -----------------===//

    // Deprecated since v1.1.
    gs_hsa_enabled_map["hsa_code_object_get_symbol_from_name"] = value;

    //===--- Executable - HSA v1.1 additions --------------------------------===//

    gs_hsa_enabled_map["hsa_code_object_reader_create_from_file"] = value;
    gs_hsa_enabled_map["hsa_code_object_reader_create_from_memory"] = value;
    gs_hsa_enabled_map["hsa_code_object_reader_destroy"] = value;
    gs_hsa_enabled_map["hsa_executable_create_alt"] = value;
    gs_hsa_enabled_map["hsa_executable_load_program_code_object"] = value;
    gs_hsa_enabled_map["hsa_executable_load_agent_code_object"] = value;
    gs_hsa_enabled_map["hsa_executable_validate_alt"] = value;
    gs_hsa_enabled_map["hsa_executable_get_symbol_by_name"] = value;
    gs_hsa_enabled_map["hsa_executable_iterate_agent_symbols"] = value;
    gs_hsa_enabled_map["hsa_executable_iterate_program_symbols"] = value;
}

static void InitEnabledTableExtApi(bool value) {
    // Initialize function pointers for Amd Extension Api's
    gs_hsa_enabled_map["hsa_amd_coherency_get_type"] = value;
    gs_hsa_enabled_map["hsa_amd_coherency_set_type"] = value;
    gs_hsa_enabled_map["hsa_amd_profiling_set_profiler_enabled"] = value;
    gs_hsa_enabled_map["hsa_amd_profiling_async_copy_enable"] = value;
    gs_hsa_enabled_map["hsa_amd_profiling_get_dispatch_time"] = value;
    gs_hsa_enabled_map["hsa_amd_profiling_get_async_copy_time"] = value;
    gs_hsa_enabled_map["hsa_amd_profiling_convert_tick_to_system_domain"] = value;
    gs_hsa_enabled_map["hsa_amd_signal_async_handler"] = value;
    gs_hsa_enabled_map["hsa_amd_async_function"] = value;
    gs_hsa_enabled_map["hsa_amd_signal_wait_any"] = value;
    gs_hsa_enabled_map["hsa_amd_queue_cu_set_mask"] = value;
    gs_hsa_enabled_map["hsa_amd_memory_pool_get_info"] = value;
    gs_hsa_enabled_map["hsa_amd_agent_iterate_memory_pools"] = value;
    gs_hsa_enabled_map["hsa_amd_memory_pool_allocate"] = value;
    gs_hsa_enabled_map["hsa_amd_memory_pool_free"] = value;
    gs_hsa_enabled_map["hsa_amd_memory_async_copy"] = value;
    gs_hsa_enabled_map["hsa_amd_agent_memory_pool_get_info"] = value;
    gs_hsa_enabled_map["hsa_amd_agents_allow_access"] = value;
    gs_hsa_enabled_map["hsa_amd_memory_pool_can_migrate"] = value;
    gs_hsa_enabled_map["hsa_amd_memory_migrate"] = value;
    gs_hsa_enabled_map["hsa_amd_memory_lock"] = value;
    gs_hsa_enabled_map["hsa_amd_memory_unlock"] = value;
    gs_hsa_enabled_map["hsa_amd_memory_fill"] = value;
    gs_hsa_enabled_map["hsa_amd_interop_map_buffer"] = value;
    gs_hsa_enabled_map["hsa_amd_interop_unmap_buffer"] = value;
    gs_hsa_enabled_map["hsa_amd_image_create"] = value;
    gs_hsa_enabled_map["hsa_amd_pointer_info"] = value;
    gs_hsa_enabled_map["hsa_amd_pointer_info_set_userdata"] = value;
    gs_hsa_enabled_map["hsa_amd_ipc_memory_create"] = value;
    gs_hsa_enabled_map["hsa_amd_ipc_memory_attach"] = value;
    gs_hsa_enabled_map["hsa_amd_ipc_memory_detach"] = value;
    gs_hsa_enabled_map["hsa_amd_signal_create"] = value;
    gs_hsa_enabled_map["hsa_amd_ipc_signal_create"] = value;
    gs_hsa_enabled_map["hsa_amd_ipc_signal_attach"] = value;
    gs_hsa_enabled_map["hsa_amd_register_system_event_handler"] = value;
    gs_hsa_enabled_map["hsa_amd_queue_intercept_create"] = value;
    gs_hsa_enabled_map["hsa_amd_queue_intercept_register"] = value;
    gs_hsa_enabled_map["hsa_amd_queue_set_priority"] = value;
    gs_hsa_enabled_map["hsa_amd_memory_async_copy_rect"] = value;
#if RTG_ENABLE_HSA_AMD_RUNTIME_QUEUE_CREATE_REGISTER
    gs_hsa_enabled_map["hsa_amd_runtime_queue_create_register"] = value;
#endif
#if RTG_ENABLE_HSA_AMD_MEMORY_LOCK_TO_POOL
    gs_hsa_enabled_map["hsa_amd_memory_lock_to_pool"] = value;
#endif
    gs_hsa_enabled_map["hsa_amd_register_deallocation_callback"] = value;
    gs_hsa_enabled_map["hsa_amd_deregister_deallocation_callback"] = value;
    gs_hsa_enabled_map["hsa_amd_signal_value_pointer"] = value;
    gs_hsa_enabled_map["hsa_amd_svm_attributes_set"] = value;
    gs_hsa_enabled_map["hsa_amd_svm_attributes_get"] = value;
    gs_hsa_enabled_map["hsa_amd_svm_prefetch_async"] = value;
#if (HIP_VERSION_MAJOR == 4 && HIP_VERSION_MINOR >= 4) || HIP_VERSION_MAJOR > 4
    gs_hsa_enabled_map["hsa_amd_queue_cu_get_mask"] = value;
#endif
}

#define RTG_HSA_CHECK_STATUS(msg, status) do { \
    if (status != HSA_STATUS_SUCCESS) {        \
        fprintf(stderr, "RTG Tracer: " msg);   \
        abort();                               \
    }                                          \
} while (false)

static hsa_status_t hsa_executable_symbols_cb(hsa_executable_t exec, hsa_executable_symbol_t symbol, void *data) {
    hsa_symbol_kind_t value = (hsa_symbol_kind_t)0;
    hsa_status_t status = gs_OrigCoreApiTable.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &value);
    RTG_HSA_CHECK_STATUS("Error in getting symbol info", status);
    if (value == HSA_SYMBOL_KIND_KERNEL) {
        uint64_t addr = 0;
        uint32_t len = 0;
        status = gs_OrigCoreApiTable.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &addr);
        RTG_HSA_CHECK_STATUS("Error in getting kernel object", status);
        status = gs_OrigCoreApiTable.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &len);
        RTG_HSA_CHECK_STATUS("Error in getting name len", status);
        char *name = new char[len + 1];
        status = gs_OrigCoreApiTable.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name);
        RTG_HSA_CHECK_STATUS("Error in getting kernel name", status);
        name[len] = 0;
        auto ret = gs_symbols_map_->emplace(addr, name);
        if (ret.second == false) {
            delete[] ret.first->second;
            ret.first->second = name;
        }
    }
    return HSA_STATUS_SUCCESS;
}

static hsa_status_t hsa_iterate_agent_cb(hsa_agent_t agent, void* data) {
    hsa_device_type_t dev_type = HSA_DEVICE_TYPE_CPU;

    hsa_status_t stat = gs_OrigCoreApiTable.hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_DEVICE, &dev_type);

    if (stat != HSA_STATUS_SUCCESS) {
        return stat;
    }

    if (dev_type == HSA_DEVICE_TYPE_CPU) {
        gs_cpu_agents.push_back(agent);
    } else if (dev_type == HSA_DEVICE_TYPE_GPU) {
        gs_gpu_agents.push_back(agent);
    }

    return stat;
}

static hsa_status_t hsa_executable_freeze_interceptor(hsa_executable_t executable, const char *options) {
    std::lock_guard<mutex_t> lck(gs_kernel_name_mutex_);
    if (gs_symbols_map_ == NULL) gs_symbols_map_ = new symbols_map_t;
    hsa_status_t status = gs_OrigCoreApiTable.hsa_executable_iterate_symbols_fn(executable, hsa_executable_symbols_cb, NULL);
    RTG_HSA_CHECK_STATUS("Error in iterating executable symbols", status);
    return gs_OrigCoreApiTable.hsa_executable_freeze_fn(executable, options);
}

static inline hsa_packet_type_t GetHeaderType(const packet_t* packet) {
    static const packet_word_t cs_header_type_mask = (1ul << HSA_PACKET_HEADER_WIDTH_TYPE) - 1;
    const packet_word_t* header = reinterpret_cast<const packet_word_t*>(packet);
    return static_cast<hsa_packet_type_t>((*header >> HSA_PACKET_HEADER_TYPE) & cs_header_type_mask);
}

static inline const amd_kernel_code_t* GetKernelCode(uint64_t kernel_object) {
    const amd_kernel_code_t* kernel_code = NULL;
    hsa_status_t status =
        gs_OrigLoaderExtTable.hsa_ven_amd_loader_query_host_address(
                reinterpret_cast<const void*>(kernel_object),
                reinterpret_cast<const void**>(&kernel_code));
    if (HSA_STATUS_SUCCESS != status) {
        kernel_code = reinterpret_cast<amd_kernel_code_t*>(kernel_object);
    }
    return kernel_code;
}

static inline const char* GetKernelNameRef(uint64_t addr) {
    std::lock_guard<mutex_t> lck(gs_kernel_name_mutex_);
    const auto it = gs_symbols_map_->find(addr);
    if (it == gs_symbols_map_->end()) {
        fprintf(stderr, "RTG Tracer: kernel addr (0x%lx) is not found\n", addr);
        abort();
    }
    return it->second;
}

// Demangle C++ symbol name
static std::string cpp_demangle(const char* symname) {
    std::string retval;
    std::size_t pos;

    if (RTG_DEMANGLE) {
        size_t size = 0;
        int status = 0;
        char* result = abi::__cxa_demangle(symname, NULL, &size, &status);
        if (result) {
            // caller of __cxa_demangle must free returned buffer
            retval = result;
            free(result);
        }
        else {
            // demangle failed?
            retval = symname;
        }
    }
    else {
        retval = symname;
    }

    // for some reason ' [clone .kd]' appears at the end of some kernels; remove it
    pos = retval.find(" [clone .kd]");
    if (pos != std::string::npos) {
        retval = retval.substr(0, pos);
    }

    // for some reason '.kd' appears at the end of some kernels; remove it
    pos = retval.find(".kd");
    if (pos != std::string::npos) {
        retval = retval.substr(0, pos);
    }

    return retval;
}

static std::string QueryKernelName(uint64_t kernel_object, const amd_kernel_code_t* kernel_code) {
    const char* kernel_symname = GetKernelNameRef(kernel_object);
    if (HCC_PROFILE) {
        return std::string(kernel_symname);
    }
    else {
        return cpp_demangle(kernel_symname);
    }
}


static void hsa_amd_queue_intercept_cb(
        const void* in_packets, uint64_t count,
        uint64_t user_que_idx, void* data,
        hsa_amd_queue_intercept_packet_writer writer)
{
    hsa_status_t status;
    const packet_t* packets_arr = reinterpret_cast<const packet_t*>(in_packets);
    InterceptCallbackData *data_ = reinterpret_cast<InterceptCallbackData*>(data);
    hsa_agent_t agent = data_->agent;
    hsa_queue_t *queue = data_->queue;
    hsa_queue_t *signal_queue = data_->signal_queue;

    if (writer == NULL) {
        fprintf(stderr, "RTG Tracer: fatal, intercept callback missing writer\n");
        exit(EXIT_FAILURE);
    }
    if (count != 1) {
        fprintf(stderr, "RTG Tracer: fatal, intercept callback packet count != 1\n");
        exit(EXIT_FAILURE);
    }

    // Traverse input packets
    for (uint64_t j = 0; j < count; ++j) {
        //uint64_t idx = user_que_idx + j;
        const packet_t* packet = &packets_arr[j];
        bool to_submit = true;
        hsa_packet_type_t type = GetHeaderType(packet);
        hsa_signal_t deleter_signal{0};
        hsa_signal_t new_signal{0};
        hsa_signal_t original_signal{0};
        bool owns_orig_signal = false;

        new_signal = CreateSignal();
        deleter_signal = CreateSignal();

        // Checking for dispatch packet type
        if (type == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
            ++RTG::gs_host_count_dispatches;
            const hsa_kernel_dispatch_packet_t* dispatch_packet =
                reinterpret_cast<const hsa_kernel_dispatch_packet_t*>(packet);

            uint64_t kernel_object = dispatch_packet->kernel_object;
            const amd_kernel_code_t* kernel_code = GetKernelCode(kernel_object);
            const uint64_t kernel_symbol = kernel_code->runtime_loader_kernel_symbol;
            std::string kernel_name = QueryKernelName(kernel_object, kernel_code);

            // find kernel launch from HIP to get correlation id
            // HCC output does not need correlation_id
            uint64_t correlation_id = HCC_PROFILE ? 0 : AgentInfo::Get(agent)->find_op(kernel_name);

            original_signal = dispatch_packet->completion_signal;
            if (!original_signal.handle) {
                owns_orig_signal = true;
                original_signal = CreateSignal();
            }
            const_cast<hsa_kernel_dispatch_packet_t*>(dispatch_packet)->completion_signal = new_signal;
            gs_pool.push(SignalWaiter, new SignalCallbackData(kernel_name, correlation_id, data_, deleter_signal, new_signal, original_signal, owns_orig_signal, dispatch_packet));
        }
        else if (type == HSA_PACKET_TYPE_BARRIER_AND || type == HSA_PACKET_TYPE_BARRIER_OR) {
            ++RTG::gs_host_count_barriers;
            const hsa_barrier_and_packet_t* barrier_packet =
                reinterpret_cast<const hsa_barrier_and_packet_t*>(packet);

            original_signal = barrier_packet->completion_signal;
            if (!original_signal.handle) {
                owns_orig_signal = true;
                original_signal = CreateSignal();
            }
            const_cast<hsa_barrier_and_packet_t*>(barrier_packet)->completion_signal = new_signal;
            gs_pool.push(SignalWaiter, new SignalCallbackData(data_, deleter_signal, new_signal, original_signal, owns_orig_signal, barrier_packet));
        }
        else {
            fprintf(stderr, "RTG Tracer: unrecognized packet type %d\n", type);
            exit(EXIT_FAILURE);
        }

        // Submitting the original packets as if profiling was not enabled
        writer(packet, 1);

        // Submit a new packet just for decrementing the original signal. This is done in a separate queue. Signal packet depends on real packet.
        if (original_signal.handle) {
            submit_to_signal_queue(signal_queue, new_signal, original_signal);
            submit_to_signal_queue(signal_queue, gs_NullSignal, deleter_signal);
        }
        else {
            fprintf(stderr, "RTG Tracer: fatal, intercept callback missing original signal\n");
            exit(EXIT_FAILURE);
        }

    }
}

static hsa_signal_t CreateSignal()
{
    ++gs_host_count_signals;
    hsa_signal_t signal;
    hsa_status_t status = gs_OrigCoreApiTable.hsa_signal_create_fn(1, 0, nullptr, &signal);
    if (status != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "RTG Tracer: hsa_signal_create_fn failed\n");
        exit(EXIT_FAILURE);
    }
    return signal;
}

static void DestroySignal(hsa_signal_t signal)
{
    hsa_status_t status = gs_OrigCoreApiTable.hsa_signal_destroy_fn(signal);
    if (status != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "RTG Tracer: hsa_signal_destroy_fn failed %lu\n", signal.handle);
    }
    ++gs_cb_count_signals;
}

static void* hip_activity_callback(uint32_t cid, activity_record_t* record, const void* data, void* arg)
{
    // The hip_api_data_t is returned from this activity callback; without it, the hip_api_callback does not fire.
    return (hip_api_data_t*)gstl_hip_api_data[cid];
}

static void* hip_api_callback(uint32_t domain, uint32_t cid, const void* data_, void* arg)
{
    hip_api_data_t *data = (hip_api_data_t*)data_;

    std::string kernname_str;
    const char *kernname = NULL;
    hipStream_t stream;

    switch (cid) {
        case HIP_API_ID_hipLaunchCooperativeKernel:
            stream = data->args.hipLaunchCooperativeKernel.stream;
            kernname = hipKernelNameRefByPtr(data->args.hipLaunchCooperativeKernel.f, stream);
            break;
        case HIP_API_ID_hipLaunchKernel:
            stream = data->args.hipLaunchKernel.stream;
            kernname = hipKernelNameRefByPtr(data->args.hipLaunchKernel.function_address, stream);
            break;
        case HIP_API_ID_hipHccModuleLaunchKernel:
            stream = data->args.hipHccModuleLaunchKernel.hStream;
            kernname = hipKernelNameRef(data->args.hipHccModuleLaunchKernel.f);
            break;
        case HIP_API_ID_hipExtModuleLaunchKernel:
            stream = data->args.hipExtModuleLaunchKernel.hStream;
            kernname = hipKernelNameRef(data->args.hipExtModuleLaunchKernel.f);
            break;
        case HIP_API_ID_hipModuleLaunchKernel:
            stream = data->args.hipModuleLaunchKernel.stream;
            kernname = hipKernelNameRef(data->args.hipModuleLaunchKernel.f);
            break;
        case HIP_API_ID_hipExtLaunchKernel:
            stream = data->args.hipExtLaunchKernel.stream;
            kernname = hipKernelNameRefByPtr(data->args.hipExtLaunchKernel.function_address, stream);
            break;
    }

    // if this is a kernel op, kernname is set
    if (kernname) {
        // demangle kernname
        kernname_str = cpp_demangle(kernname);
    }

    if (data->phase == 0) {
        data->correlation_id = gs_correlation_id_counter++;
        gstl_hip_api_tick[cid] = tick();

        // if this is a kernel op, kernname is set
        if (kernname) {
            int ord = hipGetStreamDeviceId(stream);
            if (HCC_PROFILE) {
                // HCC output does not need correlation id
            }
            else {
                AgentInfo::Get(ord)->insert_op({kernname_str,data->correlation_id});
            }
        }
    }
    else {
        uint64_t tick_ = gstl_hip_api_tick[cid];
        uint64_t ticks = tick() - tick_;
        int localStatus = hipPeekAtLastError();

        LOG_HIP(cid, data, localStatus, tick_, ticks, kernname_str, RTG_HIP_API_ARGS);

        // Now that we're done with the api data, zero it for the next time.
        // Otherwise, phase is always wrong because HIP doesn't set the phase to 0 during API start.
        //memset(data, 0, sizeof(hip_api_data_t));
        data->phase = 0;
    }

    return NULL;
}

static void* roctx_callback(uint32_t domain, uint32_t cid, const void* data_, void* arg)
{
    uint64_t new_tick = tick();
    roctx_api_data_t *data = (roctx_api_data_t*)data_;

    switch (cid) {
        case ROCTX_API_ID_roctxMarkA:
            LOG_ROCTX_MARK(gs_correlation_id_counter++, data->args.message, new_tick);
            break;
        case ROCTX_API_ID_roctxRangePushA:
            gstl_roctx_stack.emplace_back(data->args.message, new_tick, gs_correlation_id_counter++);
            break;
        case ROCTX_API_ID_roctxRangePop:
            {
                auto& item = gstl_roctx_stack.back();
                LOG_ROCTX(item.correlation_id, item.message, item.tick, new_tick - item.tick);
                gstl_roctx_stack.pop_back();
            }
            break;
        case ROCTX_API_ID_roctxRangeStartA:
            gstl_roctx_range.emplace(data->args.id, roctx_data_t{data->args.message,new_tick,gs_correlation_id_counter++});
            break;
        case ROCTX_API_ID_roctxRangeStop:
            {
                auto it = gstl_roctx_range.find(data->args.id);
                if (it == gstl_roctx_range.end()) {
                    fprintf(stderr, "RTG Tracer: invalid roctx range: %lu\n", data->args.id);
                    exit(EXIT_FAILURE);
                }
                auto &item = it->second;
                LOG_ROCTX(item.correlation_id, item.message, item.tick, new_tick - item.tick);
                gstl_roctx_range.erase(it);
            }
            break;
    }

    // NOTE: roctx uses strdup() with all messages stored in the roctx_api_data_t ; user must free
    if (data->args.message) {
        free(const_cast<char*>(data->args.message));
    }

    return NULL;
}

static void finalize_once()
{
    fprintf(stderr, "RTG Tracer: Finalizing\n");

    if (!RTG::loaded) {
        return;
    }

    if (RTG_PROFILE || RTG_PROFILE_COPY) {
        fprintf(stderr, "RTG Tracer: host_count_dispatches=%u\n", RTG::gs_host_count_dispatches.load());
        fprintf(stderr, "RTG Tracer:   cb_count_dispatches=%u\n", RTG::gs_cb_count_dispatches.load());
        fprintf(stderr, "RTG Tracer:   host_count_barriers=%u\n", RTG::gs_host_count_barriers.load());
        fprintf(stderr, "RTG Tracer:     cb_count_barriers=%u\n", RTG::gs_cb_count_barriers.load());
        fprintf(stderr, "RTG Tracer:     host_count_copies=%u\n", RTG::gs_host_count_copies.load());
        fprintf(stderr, "RTG Tracer:       cb_count_copies=%u\n", RTG::gs_cb_count_copies.load());
        fprintf(stderr, "RTG Tracer:    host_count_signals=%u\n", RTG::gs_host_count_signals.load());
        fprintf(stderr, "RTG Tracer:      cb_count_signals=%u\n", RTG::gs_cb_count_signals.load());
    }
    for (int i=0; i<5; ++i) {
        if (RTG::gs_host_count_dispatches != RTG::gs_cb_count_dispatches
                || RTG::gs_host_count_barriers != RTG::gs_cb_count_barriers) {
            fprintf(stderr, "RTG Tracer: not all callbacks have completed, waiting... dispatches %u vs %u barriers %u vs %u signals %u vs %u\n",
                    RTG::gs_host_count_dispatches.load(),
                    RTG::gs_cb_count_dispatches.load(),
                    RTG::gs_host_count_barriers.load(),
                    RTG::gs_cb_count_barriers.load(),
                    RTG::gs_host_count_signals.load(),
                    RTG::gs_cb_count_signals.load()
            );
            sleep(2);
        }
    }

    // Unregister ROCTX
    for (int i=0; i<ROCTX_API_ID_NUMBER; ++i) {
        RemoveApiCallback(i);
    }

    if (HCC_PROFILE) {
        fclose(RTG::gs_stream);
    }
    else {
        RTG::gs_out->close();
    }

    RTG::RestoreHsaTable(RTG::gs_OrigHsaTable);
}

std::once_flag flag;
static void finalize()
{
    std::call_once(flag, finalize_once);
}

} // namespace RTG

extern "C" bool OnLoad(void *pTable,
        uint64_t runtimeVersion, uint64_t failedToolCount,
        const char *const *pFailedToolNames)
{
    fprintf(stderr, "RTG Tracer: Loading\n");
    RTG::loaded = true;
    RTG::gs_NullSignal.handle = 0;

    Flag::init_all();

    std::string outname = RTG_FILE_PREFIX;
    auto pid_pos = outname.find("%p");
    if (pid_pos != std::string::npos) {
        outname = outname.substr(0, pid_pos) + RTG::pidstr() + outname.substr(pid_pos+2);
    }

#ifdef RPD_TRACER
    if (RTG_RPD) {
        outname += std::string(".rpd");
        fprintf(stderr, "RTG Tracer: Filename %s\n", outname.c_str());
    }
    else
#endif
    if (RTG_LEGACY_PRINTF) {
        outname += std::string(".txt");
        fprintf(stderr, "RTG Tracer: Filename %s\n", outname.c_str());
    }
    else {
        fprintf(stderr, "RTG Tracer: Output directory %s\n", outname.c_str());
    }

    if (HCC_PROFILE) {
        fprintf(stderr, "RTG Tracer: HCC_PROFILE=2 mode\n");
        RTG::gs_stream = fopen(outname.c_str(), "w");
    }
    else {
#ifdef RPD_TRACER
        if (RTG_RPD) {
            RTG::gs_out = new RtgOutRpd;
        }
        else
#endif
        {
            if (RTG_LEGACY_PRINTF) {
                RTG::gs_out = new RtgOutPrintf;
            }
            else {
                RTG::gs_out = new RtgOutPrintfLockless;
            }
        }
        RTG::gs_out->open(outname);

        // Register HIP APIs
        std::vector<std::string> tokens_keep = RTG::split(RTG_HIP_API_FILTER, ',');
        std::vector<std::string> tokens_prune = RTG::split(RTG_HIP_API_FILTER_OUT, ',');
        std::vector<std::string> tokens_prune_always;
        tokens_prune_always.push_back("hipPeekAtLastError"); // because we need to call it ourselves for return codes
        tokens_prune_always.push_back("hipGetDevice");
        tokens_prune_always.push_back("hipSetDevice");

        for (int i=0; i<HIP_API_ID_NUMBER; ++i) {
            bool keep = false;
            std::string name = hip_api_name(i);
            for (auto tok : tokens_keep) {
                if (tok == "all" || name.find(tok) != std::string::npos) {
                    keep = true;
                    break;
                }
            }
            for (auto tok : tokens_prune) {
                if (name.find(tok) != std::string::npos) {
                    keep = false;
                    break;
                }
            }
            for (auto tok : tokens_prune_always) {
                if (name.find(tok) != std::string::npos) {
                    keep = false;
                    break;
                }
            }

            if (keep) {
                hipRegisterActivityCallback(i, (void*)RTG::hip_activity_callback, NULL);
                hipRegisterApiCallback(i, (void*)RTG::hip_api_callback, NULL);
            }
            if (RTG_VERBOSE == "2") {
                fprintf(stderr, "RTG Tracer: HIP API %s %s\n", keep ? "keep" : "skip", name.c_str());
            }
        }

        // Register ROCTX
        for (int i=0; i<ROCTX_API_ID_NUMBER; ++i) {
            RegisterApiCallback(i, (void*)RTG::roctx_callback, NULL);
        }
    }

    // set which HSA functions we should replace
    RTG::InitEnabledTable(RTG_HSA_API_FILTER, RTG_HSA_API_FILTER_OUT);
    // replace them
    if (!RTG::InitHsaTable(reinterpret_cast<HsaApiTable*>(pTable))) {
        return false;
    }

    // Discover HSA agent info.
    if (HSA_STATUS_SUCCESS != hsa_iterate_agents(RTG::hsa_iterate_agent_cb, nullptr)) {
        fprintf(stderr, "RTG Tracer: could not iterate hsa agents\n");
        exit(EXIT_FAILURE);
    }
    std::string ordinals;
    if (HIP_VISIBLE_DEVICES) {
        ordinals = std::string(HIP_VISIBLE_DEVICES);
    }
    else if (CUDA_VISIBLE_DEVICES) {
        ordinals = std::string(CUDA_VISIBLE_DEVICES);
    }
    if (ordinals[0] != '\0') {
        size_t end, pos = 0;
        std::vector<hsa_agent_t> valid_agents;
        std::set<size_t> valid_indexes;
        do {
            bool deviceIdValid = true;
            end = ordinals.find_first_of(',', pos);
            if (end == std::string::npos) {
                end = ordinals.size();
            }
            std::string strIndex = ordinals.substr(pos, end - pos);
            int index = atoi(strIndex.c_str());
            if (index < 0 ||
                    static_cast<size_t>(index) >= RTG::gs_gpu_agents.size() ||
                    strIndex != std::to_string(index)) {
                deviceIdValid = false;
            }

            if (!deviceIdValid) {
                // Exit the loop as anything to the right of invalid deviceId
                // has to be discarded
                break;
            } else {
                if (valid_indexes.find(index) == valid_indexes.end()) {
                    valid_agents.push_back(RTG::gs_gpu_agents[index]);
                    valid_indexes.insert(index);
                }
            }
            pos = end + 1;
        } while (pos < ordinals.size());
        RTG::gs_gpu_agents = valid_agents;
    }

    RTG::AgentInfo::Init(RTG::gs_gpu_agents);

    std::atexit(RTG::finalize);

    return true;
}

extern "C" void OnUnload()
{
    fprintf(stderr, "RTG Tracer: Unloading\n");
    RTG::finalize();
}

__attribute__((destructor)) static void destroy() {
    fprintf(stderr, "RTG Tracer: Destructing\n");
    RTG::finalize();
}
