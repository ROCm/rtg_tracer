#include <atomic>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
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
#include "missing_ostream_definitions.h"
#define HIP_PROF_HIP_API_STRING 1 // to enable hipApiString in hip_prof_str.h
#include <hip/amd_detail/hip_prof_str.h>

#include <roctracer/roctracer.h>

#include "ctpl_stl.h"

// User options, set using env vars.
#include "flags.h"

#define USE_ATOMIC 1
#define ENABLE_HSA_AMD_MEMORY_LOCK_TO_POOL 0
#define ENABLE_HSA_AMD_RUNTIME_QUEUE_CREATE_REGISTER 0

namespace RTG {

static HsaApiTable* gs_OrigHsaTable;     // The HSA Runtime's original table
static CoreApiTable gs_OrigCoreApiTable; // The HSA Runtime's versions of HSA core API functions
static AmdExtTable gs_OrigExtApiTable;   // The HSA Runtime's versions of HSA ext API functions
static hsa_ven_amd_loader_1_01_pfn_t gs_OrigLoaderExtTable; // The HSA Runtime's versions of HSA loader ext API functions

// Callback for queue intercept.
static void intercept_callback(const void* in_packets, uint64_t count, uint64_t user_que_idx, void* data, hsa_amd_queue_intercept_packet_writer writer);

// thread pool for signal waits
ctpl::thread_pool pool(1);
// thread pool for signal destroy
ctpl::thread_pool signal_pool(1);

#if USE_ATOMIC
typedef std::atomic<unsigned int> counter_t;
#define LOAD(a) a.load()
#else
typedef unsigned int counter_t;
#define LOAD(a) a
#endif

counter_t host_count_dispatches{0};
counter_t host_count_barriers{0};
counter_t host_count_copies{0};
counter_t host_count_signals{0};
counter_t cb_count_dispatches{0};
counter_t cb_count_barriers{0};
counter_t cb_count_copies{0};
counter_t cb_count_signals{0};

// Output stream for all logging
static FILE* stream;

// Support for HIP API callbacks.
static void* hip_activity_callback(uint32_t cid, activity_record_t* record, const void* data, void* arg);
static void* hip_api_callback(uint32_t domain, uint32_t cid, const void* data, void* arg);
// Need to allocate hip_api_data_t, but cannot use new operator due to incomplete default constructors.
// The hip_api_data_t is returned from the activity callback; without it, the API callback does not fire.
thread_local std::vector<char[sizeof(hip_api_data_t)]> hip_api_data(HIP_API_ID_NUMBER);
// Shared cache of HIP API names. Build it during init, used by all threads without mutex.
std::vector<std::string> hip_api_names(HIP_API_ID_NUMBER);
// To preserve state between API start/stop calls, we need to allocate some state.
struct hip_api_state_ {
    uint64_t tick; \
};
thread_local std::vector<struct hip_api_state_> hip_api_state(HIP_API_ID_NUMBER);

// Lookup which HSA functions we want to trace
static std::unordered_map<std::string,bool> enabled_map;

static void InitCoreApiTable(CoreApiTable* table);
static void InitAmdExtTable(AmdExtTable* table);
static void InitEnabledTable(std::string what_to_trace, std::string what_not_to_trace);
static void InitEnabledTableCore(bool value);
static void InitEnabledTableExtApi(bool value);
static inline bool enabled_check(std::string func);

// Executables loading tracking
//typedef std::recursive_mutex mutex_t;
typedef std::mutex mutex_t;
static mutex_t kernel_name_mutex_;
typedef std::unordered_map<uint64_t, const char*> symbols_map_t;
static symbols_map_t* symbols_map_;
static const char* GetKernelNameRef(uint64_t addr);
static hsa_status_t hsa_executable_freeze_interceptor(hsa_executable_t executable, const char *options);
static hsa_status_t executable_symbols_cb(hsa_executable_t exec, hsa_executable_symbol_t symbol, void *data);

// For tracking agents and streams with 0-based index.
static mutex_t agent_mutex_;
struct AgentInfo {
    static int __agent;
    int agent_index;
    int __queue;
    hsa_queue_t *signal_queue;
    std::unordered_map<uint64_t, int> queue_index;
    AgentInfo() : agent_index{__agent++}, __queue{0}, signal_queue{nullptr} {}
};
static std::unordered_map<uint64_t, AgentInfo*> agent_info_map;

int AgentInfo::__agent{0};

inline hsa_device_type_t agent_type(hsa_agent_t x)
{
    hsa_device_type_t r{};
    hsa_agent_get_info(x, HSA_AGENT_INFO_DEVICE, &r);
    return r;
}

static void get_agent_queue_indexes(hsa_agent_t agent, hsa_queue_t *queue, int &agent_index, int &queue_index)
{
    std::lock_guard<mutex_t> lck(agent_mutex_);

    if (agent.handle == 0) {
        fprintf(stderr, "RTG Tracer: null agent handle\n");
    }

    // create agent info if missing
    if (agent_info_map.count(agent.handle) == 0) {
        agent_info_map[agent.handle] = new AgentInfo;
    }

    auto *info = agent_info_map[agent.handle];

    if (info->queue_index.count(queue->id) == 0) {
        info->queue_index[queue->id] = info->__queue++;
    }
    agent_index = info->agent_index;
    queue_index = info->queue_index[queue->id];
}

static hsa_queue_t* get_agent_signal_queue(hsa_agent_t agent)
{
    std::lock_guard<mutex_t> lck(agent_mutex_);

    if (agent.handle == 0) {
        fprintf(stderr, "RTG Tracer: null agent handle\n");
    }

    // create agent info if missing
    if (agent_info_map.count(agent.handle) == 0) {
        agent_info_map[agent.handle] = new AgentInfo;
    }

    auto *info = agent_info_map[agent.handle];

    if (info->signal_queue == nullptr) {
        hsa_status_t status;
        // create a regular queue; this is for our fake signaling queue
        status = gs_OrigCoreApiTable.hsa_queue_create_fn(agent, 2048, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr,
                std::numeric_limits<unsigned int>::max(), std::numeric_limits<unsigned int>::max(), &info->signal_queue);
        if (status != HSA_STATUS_SUCCESS) {
            const char *msg;
            hsa_status_string(status, &msg);
            fprintf(stderr, "RTG Tracer: could not create agent signal queue: %s\n", msg);
            exit(EXIT_FAILURE);
        }
        // make sure profiling is enabled for the newly created queue
        status = gs_OrigExtApiTable.hsa_amd_profiling_set_profiler_enabled_fn(info->signal_queue, true);
        if (status != HSA_STATUS_SUCCESS) {
            const char *msg;
            hsa_status_string(status, &msg);
            fprintf(stderr, "RTG Tracer: could not create agent signal queue with profile: %s\n", msg);
            exit(EXIT_FAILURE);
        }
    }

    return info->signal_queue;
}

struct InterceptCallbackData
{
    InterceptCallbackData(hsa_queue_t *queue, hsa_agent_t agent, hsa_queue_t *signal_queue)
        : queue(queue), agent(agent), signal_queue(signal_queue), agent_index(0), queue_index(0), seq_index(0)
    {
        get_agent_queue_indexes(agent, queue, agent_index, queue_index);
    }
    hsa_queue_t *queue;
    hsa_agent_t agent;
    hsa_queue_t *signal_queue;
    int agent_index;
    int queue_index;
    std::atomic<int> seq_index;
};

// These declarations were getting quite lengthy; moved to separate file to aid readability.
#include "ToStringDefinitions.h"

static inline int pid() {
    return getpid();
}

static inline std::string pidstr() {
    std::ostringstream pid_os;
    pid_os << pid();
    return pid_os.str();
}

static inline std::string tid() {
    std::ostringstream tid_os;
    tid_os << std::this_thread::get_id();
    return tid_os.str();
}

#if 0
static inline suseconds_t tick() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}
#endif
#if 0
static inline uint64_t tick() {
    using namespace std::chrono;
    return high_resolution_clock::now().time_since_epoch() / microseconds(1);
}
#endif
uint64_t tick() {
    struct timespec tp;
    ::clock_gettime(CLOCK_MONOTONIC, &tp);
    return (uint64_t)tp.tv_sec * (1000ULL * 1000ULL * 1000ULL) + (uint64_t)tp.tv_nsec;
}

// dispatch ID
static inline unsigned long did() {
    static unsigned long id = 0;
    return id++;
}

// copied from hcc runtime's HCC_PROFILE=2
#define LOG_PROFILE(start, end, type, tag, msg) \
{\
    std::stringstream sstream;\
    sstream << "profile: " << std::setw(7) << type << ";\t" \
                         << std::setw(40) << tag\
                         << ";\t" << std::fixed << std::setw(6) << std::setprecision(1) << (end-start)/1000.0 << " us;";\
    sstream << "\t" << start << ";\t" << end << ";";\
    sstream << "\t" << "#" << agent_id_ << "." << queue_id_ << "." << seq_num_ << ";"; \
    sstream <<  msg << "\n";\
    fprintf(stream, "%s", sstream.str().c_str());\
}
#define LOG_RPT LOG_PROFILE(start_, stop_, type_, tag_, msg_)
    //std::cerr << sstream.str();\

#define DISABLE_LOGGING 0
#if DISABLE_LOGGING

#define LOG_STATUS_OUT
#define LOG_UINT64_OUT
#define LOG_SIGNAL_OUT
#define LOG_VOID_OUT
#define LOG_DISPATCH_HOST
#define LOG_DISPATCH
#define LOG_BARRIER_HOST
#define LOG_BARRIER
#define LOG_COPY
#define TRACE(...)
#define LOG_STATUS(status) ({ hsa_status_t localStatus = status; localStatus; })
#define LOG_SIGNAL(status) ({ hsa_signal_value_t localStatus = status; localStatus; })
#define LOG_UINT64(status) ({ uint64_t localStatus = status; localStatus; })
#define LOG_UINT32(status) ({ uint32_t localStatus = status; localStatus; })
#define LOG_VOID(status) ({ status; })

#else // DISABLE_LOGGING

#define LOG_STATUS_OUT \
fprintf(stream, "HSA: pid:%d tid:%s %s %s ret=%d @%lu +%lu\n", pid_, tid_.c_str(), func.c_str(), args.c_str(), localStatus, tick_, ticks); fflush(stream);
#define LOG_UINT64_OUT \
fprintf(stream, "HSA: pid:%d tid:%s %s %s ret=%lu @%lu +%lu\n", pid_, tid_.c_str(), func.c_str(), args.c_str(), localStatus, tick_, ticks); fflush(stream);
#define LOG_SIGNAL_OUT \
fprintf(stream, "HSA: pid:%d tid:%s %s %s ret=%ld @%lu +%lu\n", pid_, tid_.c_str(), func.c_str(), args.c_str(), localStatus, tick_, ticks); fflush(stream);
#define LOG_VOID_OUT \
fprintf(stream, "HSA: pid:%d tid:%s %s %s ret=void @%lu +%lu\n", pid_, tid_.c_str(), func.c_str(), args.c_str(), tick_, ticks); fflush(stream);
#define LOG_DISPATCH_HOST \
fprintf(stream, "HSA: pid:%d tid:%s dispatch queue:%lu agent:%lu signal:%lu name:'%s' tick:%lu id:%lu workgroup:{%d,%d,%d} grid:{%d,%d,%d}\n", pid_, tid_.c_str(), queue_->id, agent_.handle, signal_.handle, name_, tick_, id_, packet->workgroup_size_x, packet->workgroup_size_y, packet->workgroup_size_z, packet->grid_size_x, packet->grid_size_y, packet->grid_size_z);
#define LOG_DISPATCH \
fprintf(stream, "HSA: pid:%d tid:%s dispatch queue:%lu agent:%lu signal:%lu name:'%s' start:%lu stop:%lu id:%lu\n", pid_, tid_.c_str(), queue_->id, agent_.handle, signal_.handle, name_, start_, stop_, id_); fflush(stream);
#define LOG_BARRIER_HOST \
fprintf(stream, "HSA: pid:%d tid:%s barrier queue:%lu agent:%lu signal:%lu dep1:%lu dep2:%lu dep3:%lu dep4:%lu dep5:%lu tick:%lu id:%lu\n", pid_, tid_.c_str(), queue_->id, agent_.handle, signal_.handle, dep1, dep2, dep3, dep4, dep5, tick_, id_); fflush(stream);
#define LOG_BARRIER \
fprintf(stream, "HSA: pid:%d tid:%s barrier queue:%lu agent:%lu signal:%lu start:%lu stop:%lu dep1:%lu dep2:%lu dep3:%lu dep4:%lu dep5:%lu id:%lu\n", pid_, tid_.c_str(), queue_->id, agent_.handle, signal_.handle, start_, stop_, data->dep1, data->dep2, data->dep3, data->dep4, data->dep5, data->id_); fflush(stream);
#define LOG_COPY \
fprintf(stream, "HSA: pid:%d tid:%s copy agent:%lu signal:%lu start:%lu stop:%lu dep1:%lu dep2:%lu dep3:%lu dep4:%lu dep5:%lu\n", pid_, tid_.c_str(), agent_.handle, signal_.handle, start_, stop_, data->dep1, data->dep2, data->dep3, data->dep4, data->dep5); fflush(stream);
//#define HIP_API_START \
//fprintf(stream, "HIP: pid:%d tid:%s %s %s @%lu\n", pid_, tid_.c_str(), func.c_str(), args.c_str(), tick_); fflush(stream);
#define LOG_HIP \
fprintf(stream, "HIP: pid:%d tid:%s %s ret=%d @%lu +%lu\n", pid_, tid_.c_str(), func.c_str(), localStatus, tick_, ticks); fflush(stream);
#define LOG_HIP_ARGS \
fprintf(stream, "HIP: pid:%d tid:%s %s %s ret=%d @%lu +%lu\n", pid_, tid_.c_str(), func.c_str(), args, localStatus, tick_, ticks); fflush(stream);

#define TRACE(...) \
    static bool is_enabled = enabled_check(__func__); \
    std::string func; \
    std::string args; \
    int pid_; \
    std::string tid_; \
    uint64_t tick_; \
    if (is_enabled) { \
        func = __func__; \
        args = "(" + ToString(__VA_ARGS__) + ")"; \
        pid_ = pid(); \
        tid_ = tid(); \
        tick_ = tick(); \
    }

#define LOG_STATUS(status)                                                               \
    ({                                                                                   \
        hsa_status_t localStatus = status; /*local copy so status only evaluated once*/  \
        if (is_enabled) {                                                                \
            uint64_t ticks = tick() - tick_;                                             \
            LOG_STATUS_OUT                                                               \
        }                                                                                \
        localStatus;                                                                     \
    })

#define LOG_SIGNAL(status)                                                                     \
    ({                                                                                         \
        hsa_signal_value_t localStatus = status; /*local copy so status only evaluated once*/  \
        if (is_enabled) {                                                                      \
            uint64_t ticks = tick() - tick_;                                                   \
            LOG_SIGNAL_OUT                                                                     \
        }                                                                                      \
        localStatus;                                                                           \
    })

#define LOG_UINT64(status)                                                           \
    ({                                                                               \
        uint64_t localStatus = status; /*local copy so status only evaluated once*/  \
        if (is_enabled) {                                                            \
            uint64_t ticks = tick() - tick_;                                         \
            LOG_UINT64_OUT                                                           \
        }                                                                            \
        localStatus;                                                                 \
    })

#define LOG_UINT32(status)                                                           \
    ({                                                                               \
        uint32_t localStatus = status; /*local copy so status only evaluated once*/  \
        if (is_enabled) {                                                            \
            uint64_t ticks = tick() - tick_;                                         \
            LOG_STATUS_OUT                                                           \
        }                                                                            \
        localStatus;                                                                 \
    })

#define LOG_VOID(status)                                                       \
    ({                                                                         \
        status;                                                                \
        if (is_enabled) {                                                      \
            uint64_t ticks = tick() - tick_;                                   \
            LOG_VOID_OUT                                                       \
        }                                                                      \
    })

#endif // DISABLE_LOGGING

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
	barrier->dep_signal[0] = new_signal;
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
    SignalCallbackData(const char *name, InterceptCallbackData *data, hsa_signal_t signal, hsa_signal_t orig_signal, bool owns_orig_signal, const hsa_kernel_dispatch_packet_t *packet)
        : name(name), data(data), queue(data->queue), agent(data->agent), signal(signal), orig_signal(orig_signal), owns_orig_signal(owns_orig_signal), bytes(0), direction(0),
            is_copy(false), is_barrier(false), dep1(0), dep2(0), dep3(0), dep4(0), dep5(0), id_(did()), seq_num_(data->seq_index++)
    {
        if (RTG_HSA_HOST_DISPATCH) {
            long unsigned tick_ = tick();
            int pid_ = pid();
            std::string tid_ = tid();
            const char *name_ = name;
            hsa_agent_t agent_ = agent;
            hsa_queue_t* queue_ = queue;
            hsa_signal_t signal_ = signal;
            LOG_DISPATCH_HOST
        }
    }

    SignalCallbackData(hsa_queue_t* queue, hsa_agent_t agent, hsa_signal_t signal, hsa_signal_t orig_signal, bool owns_orig_signal, uint32_t num_dep_signals, const hsa_signal_t* dep_signals, size_t bytes, int direction, int seq)
        : name(nullptr), queue(queue), agent(agent), signal(signal), orig_signal(orig_signal), owns_orig_signal(owns_orig_signal), bytes(bytes), direction(direction),
            is_copy(true),
            is_barrier(false),
            dep1(num_dep_signals>0 ? dep_signals[0].handle : 0),
            dep2(num_dep_signals>1 ? dep_signals[1].handle : 0),
            dep3(num_dep_signals>2 ? dep_signals[2].handle : 0),
            dep4(num_dep_signals>3 ? dep_signals[3].handle : 0),
            dep5(num_dep_signals>4 ? dep_signals[4].handle : 0),
            id_(0),
            seq_num_(seq)
    {}

    SignalCallbackData(InterceptCallbackData *data, hsa_signal_t signal, hsa_signal_t orig_signal, bool owns_orig_signal, const hsa_barrier_and_packet_t* packet)
        : name(nullptr), data(data), queue(data->queue), agent(data->agent), signal(signal), orig_signal(orig_signal), owns_orig_signal(owns_orig_signal), bytes(0), direction(0),
            is_copy(false),
            is_barrier(true),
            dep1(packet->dep_signal[0].handle),
            dep2(packet->dep_signal[1].handle),
            dep3(packet->dep_signal[2].handle),
            dep4(packet->dep_signal[3].handle),
            dep5(packet->dep_signal[4].handle),
            id_(did()),
            seq_num_(data->seq_index++)
    {
        if (RTG_HSA_HOST_DISPATCH) {
            long unsigned tick_ = tick();
            int pid_ = pid();
            std::string tid_ = tid();
            const char *name_ = "barrier";
            hsa_agent_t agent_ = agent;
            hsa_queue_t* queue_ = queue;
            hsa_signal_t signal_ = signal;
            LOG_BARRIER_HOST
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

    const char *name;
    InterceptCallbackData *data;
    hsa_queue_t* queue;
    hsa_agent_t agent;
    hsa_signal_t signal;
    hsa_signal_t orig_signal;
    bool owns_orig_signal;
    size_t bytes;
    int direction;
    bool is_copy;
    bool is_barrier;
    long unsigned dep1;
    long unsigned dep2;
    long unsigned dep3;
    long unsigned dep4;
    long unsigned dep5;
    long unsigned start;
    long unsigned stop;
    long unsigned id_;
    int seq_num_;
};

static void SignalDestroyer(int id, hsa_signal_t ours, hsa_signal_t theirs, bool owns_orig_signal)
{
    //fprintf(stderr, "RTG Tracer: SignalDestroyer id=%d ours=%lu theirs=%lu\n", id, ours.handle, theirs.handle);
    gs_OrigCoreApiTable.hsa_signal_wait_relaxed_fn(theirs, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
    hsa_status_t status = gs_OrigCoreApiTable.hsa_signal_destroy_fn(ours);
    ++cb_count_signals;
    if (status != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "RTG Tracer: signal destroy failed %lu\n", ours.handle);
    }
    if (owns_orig_signal) {
        status = gs_OrigCoreApiTable.hsa_signal_destroy_fn(theirs);
        if (status != HSA_STATUS_SUCCESS) {
            fprintf(stderr, "RTG Tracer: signal destroy failed %lu\n", theirs.handle);
        }
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

static bool signal_callback(hsa_signal_value_t value, void* arg)
{
    if (arg != nullptr) {
        SignalCallbackData* data = reinterpret_cast<SignalCallbackData*>(arg);
        bool okay = data->compute_profile();
        if (okay) {
            long unsigned start_ = data->start;
            long unsigned stop_ = data->stop;
            int pid_ = pid();
            std::string tid_ = tid();
            const char *name_ = data->name;
            hsa_agent_t agent_ = data->agent;
            hsa_queue_t* queue_ = data->queue;
            hsa_signal_t signal_ = data->signal;
            if (HCC_PROFILE) {
                const char *type_;
                const char *tag_;
                const char *msg_;
                std::string msgstr;
                int agent_id_ = 0;
                int queue_id_ = 0;
                int seq_num_ = 0;
                if (data->is_barrier) {
                    ++cb_count_barriers;
                    type_ = "barrier";
                    tag_ = "";
                    msg_ = "";
                    agent_id_ = data->data->agent_index;
                    queue_id_ = data->data->queue_index;
                    seq_num_ = data->seq_num_;
                }
                else if (data->is_copy) {
                    get_agent_queue_indexes(agent_, queue_, agent_id_, queue_id_);
                    ++cb_count_copies;
                    type_ = "copy";
                    tag_ = getDirectionString(data->direction);
                    msgstr = getCopyString(data->bytes, start_, stop_);
                    msg_ = msgstr.c_str();
                    //msg_ = "";
                    seq_num_ = data->seq_num_;
                }
                else {
                    ++cb_count_dispatches;
                    type_ = "kernel";
                    tag_ = data->name;
                    msg_ = "";
                    agent_id_ = data->data->agent_index;
                    queue_id_ = data->data->queue_index;
                    seq_num_ = data->seq_num_;
                }
                LOG_RPT
            }
            else {
                if (data->is_barrier) {
                    ++cb_count_barriers;
                    LOG_BARRIER
                }
                else if (data->is_copy) {
                    ++cb_count_copies;
                    LOG_COPY
                }
                else {
                    long unsigned id_ = data->id_;
                    ++cb_count_dispatches;
                    LOG_DISPATCH
                }
            }
        }
        // we created the signal, we must free, but we can't until we know it is no longer needed
        // so wait on the associated original signal
        signal_pool.push(SignalDestroyer, data->signal, data->orig_signal, data->owns_orig_signal);
        delete data;
    }
    return false; // do not re-use callback
}

static void SignalWaiter(int id, SignalCallbackData *data)
{
    //fprintf(stderr, "RTG Tracer: SignalWaiter id=%d signal=%lu\n", id, signal.handle);
    gs_OrigCoreApiTable.hsa_signal_wait_relaxed_fn(data->signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
    signal_callback(0, data);
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
    if (RTG_PROFILE) {
        hsa_status_t status;
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
                *queue, intercept_callback, data);
        // print as usual
        TRACE(agent, size, type, callback, data, private_segment_size, group_segment_size, queue);
        return LOG_STATUS(status);
    }
    else {
        // print as usual
        TRACE(agent, size, type, callback, data, private_segment_size, group_segment_size, queue);
        return LOG_STATUS(gs_OrigCoreApiTable.hsa_queue_create_fn(agent, size, type, callback, data, private_segment_size, group_segment_size, queue));
    }
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
    TRACE(executable, options);
    if (RTG_PROFILE) {
        return LOG_STATUS(hsa_executable_freeze_interceptor(executable, options));
    }
    else {
        return LOG_STATUS(gs_OrigCoreApiTable.hsa_executable_freeze_fn(executable, options));
    }
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

static void InitCoreApiTable(CoreApiTable* table) {
    // Initialize function pointers for Hsa Core Runtime Api's
    table->hsa_init_fn = RTG::hsa_init;
    //table->hsa_shut_down_fn = RTG::hsa_shut_down;
    table->hsa_system_get_info_fn = RTG::hsa_system_get_info;
    table->hsa_system_extension_supported_fn = RTG::hsa_system_extension_supported;
    table->hsa_system_get_extension_table_fn = RTG::hsa_system_get_extension_table;
    table->hsa_iterate_agents_fn = RTG::hsa_iterate_agents;
    table->hsa_agent_get_info_fn = RTG::hsa_agent_get_info;
    table->hsa_agent_get_exception_policies_fn = RTG::hsa_agent_get_exception_policies;
    table->hsa_agent_extension_supported_fn = RTG::hsa_agent_extension_supported;
    table->hsa_queue_create_fn = RTG::hsa_queue_create;
    table->hsa_soft_queue_create_fn = RTG::hsa_soft_queue_create;
    table->hsa_queue_destroy_fn = RTG::hsa_queue_destroy;
    table->hsa_queue_inactivate_fn = RTG::hsa_queue_inactivate;
    table->hsa_queue_load_read_index_scacquire_fn = RTG::hsa_queue_load_read_index_scacquire;
    table->hsa_queue_load_read_index_relaxed_fn = RTG::hsa_queue_load_read_index_relaxed;
    table->hsa_queue_load_write_index_scacquire_fn = RTG::hsa_queue_load_write_index_scacquire;
    table->hsa_queue_load_write_index_relaxed_fn = RTG::hsa_queue_load_write_index_relaxed;
    table->hsa_queue_store_write_index_relaxed_fn = RTG::hsa_queue_store_write_index_relaxed;
    table->hsa_queue_store_write_index_screlease_fn = RTG::hsa_queue_store_write_index_screlease;
    table->hsa_queue_cas_write_index_scacq_screl_fn = RTG::hsa_queue_cas_write_index_scacq_screl;
    table->hsa_queue_cas_write_index_scacquire_fn = RTG::hsa_queue_cas_write_index_scacquire;
    table->hsa_queue_cas_write_index_relaxed_fn = RTG::hsa_queue_cas_write_index_relaxed;
    table->hsa_queue_cas_write_index_screlease_fn = RTG::hsa_queue_cas_write_index_screlease;
    table->hsa_queue_add_write_index_scacq_screl_fn = RTG::hsa_queue_add_write_index_scacq_screl;
    table->hsa_queue_add_write_index_scacquire_fn = RTG::hsa_queue_add_write_index_scacquire;
    table->hsa_queue_add_write_index_relaxed_fn = RTG::hsa_queue_add_write_index_relaxed;
    table->hsa_queue_add_write_index_screlease_fn = RTG::hsa_queue_add_write_index_screlease;
    table->hsa_queue_store_read_index_relaxed_fn = RTG::hsa_queue_store_read_index_relaxed;
    table->hsa_queue_store_read_index_screlease_fn = RTG::hsa_queue_store_read_index_screlease;
    table->hsa_agent_iterate_regions_fn = RTG::hsa_agent_iterate_regions;
    table->hsa_region_get_info_fn = RTG::hsa_region_get_info;
    table->hsa_memory_register_fn = RTG::hsa_memory_register;
    table->hsa_memory_deregister_fn = RTG::hsa_memory_deregister;
    table->hsa_memory_allocate_fn = RTG::hsa_memory_allocate;
    table->hsa_memory_free_fn = RTG::hsa_memory_free;
    table->hsa_memory_copy_fn = RTG::hsa_memory_copy;
    table->hsa_memory_assign_agent_fn = RTG::hsa_memory_assign_agent;
    table->hsa_signal_create_fn = RTG::hsa_signal_create;
    table->hsa_signal_destroy_fn = RTG::hsa_signal_destroy;
    table->hsa_signal_load_relaxed_fn = RTG::hsa_signal_load_relaxed;
    table->hsa_signal_load_scacquire_fn = RTG::hsa_signal_load_scacquire;
    table->hsa_signal_store_relaxed_fn = RTG::hsa_signal_store_relaxed;
    table->hsa_signal_store_screlease_fn = RTG::hsa_signal_store_screlease;
    table->hsa_signal_wait_relaxed_fn = RTG::hsa_signal_wait_relaxed;
    table->hsa_signal_wait_scacquire_fn = RTG::hsa_signal_wait_scacquire;
    table->hsa_signal_and_relaxed_fn = RTG::hsa_signal_and_relaxed;
    table->hsa_signal_and_scacquire_fn = RTG::hsa_signal_and_scacquire;
    table->hsa_signal_and_screlease_fn = RTG::hsa_signal_and_screlease;
    table->hsa_signal_and_scacq_screl_fn = RTG::hsa_signal_and_scacq_screl;
    table->hsa_signal_or_relaxed_fn = RTG::hsa_signal_or_relaxed;
    table->hsa_signal_or_scacquire_fn = RTG::hsa_signal_or_scacquire;
    table->hsa_signal_or_screlease_fn = RTG::hsa_signal_or_screlease;
    table->hsa_signal_or_scacq_screl_fn = RTG::hsa_signal_or_scacq_screl;
    table->hsa_signal_xor_relaxed_fn = RTG::hsa_signal_xor_relaxed;
    table->hsa_signal_xor_scacquire_fn = RTG::hsa_signal_xor_scacquire;
    table->hsa_signal_xor_screlease_fn = RTG::hsa_signal_xor_screlease;
    table->hsa_signal_xor_scacq_screl_fn = RTG::hsa_signal_xor_scacq_screl;
    table->hsa_signal_exchange_relaxed_fn = RTG::hsa_signal_exchange_relaxed;
    table->hsa_signal_exchange_scacquire_fn = RTG::hsa_signal_exchange_scacquire;
    table->hsa_signal_exchange_screlease_fn = RTG::hsa_signal_exchange_screlease;
    table->hsa_signal_exchange_scacq_screl_fn = RTG::hsa_signal_exchange_scacq_screl;
    table->hsa_signal_add_relaxed_fn = RTG::hsa_signal_add_relaxed;
    table->hsa_signal_add_scacquire_fn = RTG::hsa_signal_add_scacquire;
    table->hsa_signal_add_screlease_fn = RTG::hsa_signal_add_screlease;
    table->hsa_signal_add_scacq_screl_fn = RTG::hsa_signal_add_scacq_screl;
    table->hsa_signal_subtract_relaxed_fn = RTG::hsa_signal_subtract_relaxed;
    table->hsa_signal_subtract_scacquire_fn = RTG::hsa_signal_subtract_scacquire;
    table->hsa_signal_subtract_screlease_fn = RTG::hsa_signal_subtract_screlease;
    table->hsa_signal_subtract_scacq_screl_fn = RTG::hsa_signal_subtract_scacq_screl;
    table->hsa_signal_cas_relaxed_fn = RTG::hsa_signal_cas_relaxed;
    table->hsa_signal_cas_scacquire_fn = RTG::hsa_signal_cas_scacquire;
    table->hsa_signal_cas_screlease_fn = RTG::hsa_signal_cas_screlease;
    table->hsa_signal_cas_scacq_screl_fn = RTG::hsa_signal_cas_scacq_screl;

    //===--- Instruction Set Architecture -----------------------------------===//

    table->hsa_isa_from_name_fn = RTG::hsa_isa_from_name;
    // Deprecated since v1.1.
    table->hsa_isa_get_info_fn = RTG::hsa_isa_get_info;
    // Deprecated since v1.1.
    table->hsa_isa_compatible_fn = RTG::hsa_isa_compatible;

    //===--- Code Objects (deprecated) --------------------------------------===//

    // Deprecated since v1.1.
    table->hsa_code_object_serialize_fn = RTG::hsa_code_object_serialize;
    // Deprecated since v1.1.
    table->hsa_code_object_deserialize_fn = RTG::hsa_code_object_deserialize;
    // Deprecated since v1.1.
    table->hsa_code_object_destroy_fn = RTG::hsa_code_object_destroy;
    // Deprecated since v1.1.
    table->hsa_code_object_get_info_fn = RTG::hsa_code_object_get_info;
    // Deprecated since v1.1.
    table->hsa_code_object_get_symbol_fn = RTG::hsa_code_object_get_symbol;
    // Deprecated since v1.1.
    table->hsa_code_symbol_get_info_fn = RTG::hsa_code_symbol_get_info;
    // Deprecated since v1.1.
    table->hsa_code_object_iterate_symbols_fn = RTG::hsa_code_object_iterate_symbols;

    //===--- Executable -----------------------------------------------------===//

    // Deprecated since v1.1.
    table->hsa_executable_create_fn = RTG::hsa_executable_create;
    table->hsa_executable_destroy_fn = RTG::hsa_executable_destroy;
    // Deprecated since v1.1.
    table->hsa_executable_load_code_object_fn = RTG::hsa_executable_load_code_object;
    table->hsa_executable_freeze_fn = RTG::hsa_executable_freeze;
    table->hsa_executable_get_info_fn = RTG::hsa_executable_get_info;
    table->hsa_executable_global_variable_define_fn = RTG::hsa_executable_global_variable_define;
    table->hsa_executable_agent_global_variable_define_fn = RTG::hsa_executable_agent_global_variable_define;
    table->hsa_executable_readonly_variable_define_fn = RTG::hsa_executable_readonly_variable_define;
    table->hsa_executable_validate_fn = RTG::hsa_executable_validate;
    // Deprecated since v1.1.
    table->hsa_executable_get_symbol_fn = RTG::hsa_executable_get_symbol;
    table->hsa_executable_symbol_get_info_fn = RTG::hsa_executable_symbol_get_info;
    // Deprecated since v1.1.
    table->hsa_executable_iterate_symbols_fn = RTG::hsa_executable_iterate_symbols;

    //===--- Runtime Notifications ------------------------------------------===//

    table->hsa_status_string_fn = RTG::hsa_status_string;

    // Start HSA v1.1 additions
    table->hsa_extension_get_name_fn = RTG::hsa_extension_get_name;
    table->hsa_system_major_extension_supported_fn = RTG::hsa_system_major_extension_supported;
    table->hsa_system_get_major_extension_table_fn = RTG::hsa_system_get_major_extension_table;
    table->hsa_agent_major_extension_supported_fn = RTG::hsa_agent_major_extension_supported;
    table->hsa_cache_get_info_fn = RTG::hsa_cache_get_info;
    table->hsa_agent_iterate_caches_fn = RTG::hsa_agent_iterate_caches;
    // Silent store optimization is present in all signal ops when no agents are sleeping.
    table->hsa_signal_silent_store_relaxed_fn = RTG::hsa_signal_store_relaxed;
    table->hsa_signal_silent_store_screlease_fn = RTG::hsa_signal_store_screlease;
    table->hsa_signal_group_create_fn = RTG::hsa_signal_group_create;
    table->hsa_signal_group_destroy_fn = RTG::hsa_signal_group_destroy;
    table->hsa_signal_group_wait_any_scacquire_fn = RTG::hsa_signal_group_wait_any_scacquire;
    table->hsa_signal_group_wait_any_relaxed_fn = RTG::hsa_signal_group_wait_any_relaxed;

    //===--- Instruction Set Architecture - HSA v1.1 additions --------------===//

    table->hsa_agent_iterate_isas_fn = RTG::hsa_agent_iterate_isas;
    table->hsa_isa_get_info_alt_fn = RTG::hsa_isa_get_info_alt;
    table->hsa_isa_get_exception_policies_fn = RTG::hsa_isa_get_exception_policies;
    table->hsa_isa_get_round_method_fn = RTG::hsa_isa_get_round_method;
    table->hsa_wavefront_get_info_fn = RTG::hsa_wavefront_get_info;
    table->hsa_isa_iterate_wavefronts_fn = RTG::hsa_isa_iterate_wavefronts;

    //===--- Code Objects (deprecated) - HSA v1.1 additions -----------------===//

    // Deprecated since v1.1.
    table->hsa_code_object_get_symbol_from_name_fn = RTG::hsa_code_object_get_symbol_from_name;

    //===--- Executable - HSA v1.1 additions --------------------------------===//

    table->hsa_code_object_reader_create_from_file_fn = RTG::hsa_code_object_reader_create_from_file;
    table->hsa_code_object_reader_create_from_memory_fn = RTG::hsa_code_object_reader_create_from_memory;
    table->hsa_code_object_reader_destroy_fn = RTG::hsa_code_object_reader_destroy;
    table->hsa_executable_create_alt_fn = RTG::hsa_executable_create_alt;
    table->hsa_executable_load_program_code_object_fn = RTG::hsa_executable_load_program_code_object;
    table->hsa_executable_load_agent_code_object_fn = RTG::hsa_executable_load_agent_code_object;
    table->hsa_executable_validate_alt_fn = RTG::hsa_executable_validate_alt;
    table->hsa_executable_get_symbol_by_name_fn = RTG::hsa_executable_get_symbol_by_name;
    table->hsa_executable_iterate_agent_symbols_fn = RTG::hsa_executable_iterate_agent_symbols;
    table->hsa_executable_iterate_program_symbols_fn = RTG::hsa_executable_iterate_program_symbols;
}

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

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_memory_async_copy(void* dst, hsa_agent_t dst_agent, const void* src, hsa_agent_t src_agent, size_t size, uint32_t num_dep_signals, const hsa_signal_t* dep_signals, hsa_signal_t completion_signal) {
    ++RTG::host_count_copies;
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
            ++RTG::host_count_signals;
            hsa_signal_t new_signal;
            hsa_status_t status = gs_OrigCoreApiTable.hsa_signal_create_fn(1, 0, nullptr, &new_signal);
            if (status != HSA_STATUS_SUCCESS) {
                fprintf(stderr, "RTG Tracer: hsa_signal_create_fn failed in hsa_amd_memory_async_copy\n");
                exit(EXIT_FAILURE);
            }
            hsa_queue_t *queue = get_agent_signal_queue(agent_to_use);
            uint64_t index = submit_to_signal_queue(queue, new_signal, completion_signal);
            pool.push(SignalWaiter, new SignalCallbackData(queue, src_agent, new_signal, completion_signal, false, num_dep_signals, dep_signals, size, direction, index));
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
    ++RTG::host_count_copies;
    if (RTG_PROFILE_COPY) {
        if (completion_signal.handle == 0) {
            fprintf(stderr, "RTG Tracer: hsa_amd_memory_async_copy_rect no signal\n");
        }
        else {
            if (agent_type(copy_agent) != HSA_DEVICE_TYPE_CPU) {
                ++RTG::host_count_signals;
                hsa_signal_t new_signal;
                hsa_status_t status = gs_OrigCoreApiTable.hsa_signal_create_fn(1, 0, nullptr, &new_signal);
                if (status != HSA_STATUS_SUCCESS) {
                    fprintf(stderr, "RTG Tracer: hsa_amd_signal_create_fn failed in hsa_amd_memory_async_copy_rect\n");
                    exit(EXIT_FAILURE);
                }
                hsa_queue_t *queue = get_agent_signal_queue(copy_agent);
                uint64_t index = submit_to_signal_queue(queue, new_signal, completion_signal);
                pool.push(SignalWaiter, new SignalCallbackData(queue, copy_agent, new_signal, completion_signal, false, num_dep_signals, dep_signals, range->x*range->y, dir, index));
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

#if ENABLE_HSA_AMD_MEMORY_LOCK_TO_POOL
// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_memory_lock_to_pool(void* host_ptr, size_t size, hsa_agent_t* agents, int num_agent, hsa_amd_memory_pool_t pool, uint32_t flags, void** agent_ptr) {
    TRACE(host_ptr, size, agent_ptr, num_agent, pool, flags, agent_ptr);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_memory_lock_to_pool_fn(host_ptr, size, agents, num_agent, pool, flags, agent_ptr));
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
hsa_status_t hsa_amd_pointer_info(void* ptr, hsa_amd_pointer_info_t* info, void* (*alloc)(size_t), uint32_t* num_agents_accessible, hsa_agent_t** accessible) {
    TRACE(ptr, info, alloc, accessible);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_pointer_info_fn(ptr, info, alloc, num_agents_accessible, accessible));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_pointer_info_set_userdata(void* ptr, void* userptr) {
    TRACE(ptr, userptr);
    return LOG_STATUS(gs_OrigExtApiTable.hsa_amd_pointer_info_set_userdata_fn(ptr, userptr));
}

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

static void InitAmdExtTable(AmdExtTable* table) {
    // Initialize function pointers for Amd Extension Api's
    table->hsa_amd_coherency_get_type_fn = RTG::hsa_amd_coherency_get_type;
    table->hsa_amd_coherency_set_type_fn = RTG::hsa_amd_coherency_set_type;
    table->hsa_amd_profiling_set_profiler_enabled_fn = RTG::hsa_amd_profiling_set_profiler_enabled;
    table->hsa_amd_profiling_async_copy_enable_fn = RTG::hsa_amd_profiling_async_copy_enable;
    table->hsa_amd_profiling_get_dispatch_time_fn = RTG::hsa_amd_profiling_get_dispatch_time;
    table->hsa_amd_profiling_get_async_copy_time_fn = RTG::hsa_amd_profiling_get_async_copy_time;
    table->hsa_amd_profiling_convert_tick_to_system_domain_fn = RTG::hsa_amd_profiling_convert_tick_to_system_domain;
    table->hsa_amd_signal_async_handler_fn = RTG::hsa_amd_signal_async_handler;
    table->hsa_amd_async_function_fn = RTG::hsa_amd_async_function;
    table->hsa_amd_signal_wait_any_fn = RTG::hsa_amd_signal_wait_any;
    table->hsa_amd_queue_cu_set_mask_fn = RTG::hsa_amd_queue_cu_set_mask;
    table->hsa_amd_memory_pool_get_info_fn = RTG::hsa_amd_memory_pool_get_info;
    table->hsa_amd_agent_iterate_memory_pools_fn = RTG::hsa_amd_agent_iterate_memory_pools;
    table->hsa_amd_memory_pool_allocate_fn = RTG::hsa_amd_memory_pool_allocate;
    table->hsa_amd_memory_pool_free_fn = RTG::hsa_amd_memory_pool_free;
    table->hsa_amd_memory_async_copy_fn = RTG::hsa_amd_memory_async_copy;
    table->hsa_amd_agent_memory_pool_get_info_fn = RTG::hsa_amd_agent_memory_pool_get_info;
    table->hsa_amd_agents_allow_access_fn = RTG::hsa_amd_agents_allow_access;
    table->hsa_amd_memory_pool_can_migrate_fn = RTG::hsa_amd_memory_pool_can_migrate;
    table->hsa_amd_memory_migrate_fn = RTG::hsa_amd_memory_migrate;
    table->hsa_amd_memory_lock_fn = RTG::hsa_amd_memory_lock;
    table->hsa_amd_memory_unlock_fn = RTG::hsa_amd_memory_unlock;
    table->hsa_amd_memory_fill_fn = RTG::hsa_amd_memory_fill;
    table->hsa_amd_interop_map_buffer_fn = RTG::hsa_amd_interop_map_buffer;
    table->hsa_amd_interop_unmap_buffer_fn = RTG::hsa_amd_interop_unmap_buffer;
    table->hsa_amd_pointer_info_fn = RTG::hsa_amd_pointer_info;
    table->hsa_amd_pointer_info_set_userdata_fn = RTG::hsa_amd_pointer_info_set_userdata;
    table->hsa_amd_ipc_memory_create_fn = RTG::hsa_amd_ipc_memory_create;
    table->hsa_amd_ipc_memory_attach_fn = RTG::hsa_amd_ipc_memory_attach;
    table->hsa_amd_ipc_memory_detach_fn = RTG::hsa_amd_ipc_memory_detach;
    table->hsa_amd_signal_create_fn = RTG::hsa_amd_signal_create;
    table->hsa_amd_ipc_signal_create_fn = RTG::hsa_amd_ipc_signal_create;
    table->hsa_amd_ipc_signal_attach_fn = RTG::hsa_amd_ipc_signal_attach;
    table->hsa_amd_register_system_event_handler_fn = RTG::hsa_amd_register_system_event_handler;
    table->hsa_amd_queue_intercept_create_fn = RTG::hsa_amd_queue_intercept_create;
    table->hsa_amd_queue_intercept_register_fn = RTG::hsa_amd_queue_intercept_register;
    table->hsa_amd_queue_set_priority_fn = RTG::hsa_amd_queue_set_priority;
    table->hsa_amd_memory_async_copy_rect_fn = RTG::hsa_amd_memory_async_copy_rect;
#if ENABLE_HSA_AMD_RUNTIME_QUEUE_CREATE_REGISTER
    table->hsa_amd_runtime_queue_create_register_fn = RTG::hsa_amd_runtime_queue_create_register;
#endif
#if ENABLE_HSA_AMD_MEMORY_LOCK_TO_POOL
    table->hsa_amd_memory_lock_to_pool_fn = RTG::hsa_amd_memory_lock_to_pool;
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
            for (auto &&kv : enabled_map) {
                if (kv.first.find(s) != std::string::npos) {
                    kv.second = true;
                }
            }
        }
    }
    tokens = split(what_not_to_trace, ',');
    for (auto s : tokens) {
        for (auto &&kv : enabled_map) {
            if (kv.first.find(s) != std::string::npos) {
                kv.second = false;
            }
        }
    }
}

static void InitEnabledTableCore(bool value) {
    // Initialize function pointers for Hsa Core Runtime Api's
    enabled_map["hsa_init"] = value;
    //enabled_map["hsa_shut_down"] = value;
    enabled_map["hsa_system_get_info"] = value;
    enabled_map["hsa_system_extension_supported"] = value;
    enabled_map["hsa_system_get_extension_table"] = value;
    enabled_map["hsa_iterate_agents"] = value;
    enabled_map["hsa_agent_get_info"] = value;
    enabled_map["hsa_agent_get_exception_policies"] = value;
    enabled_map["hsa_agent_extension_supported"] = value;
    enabled_map["hsa_queue_create"] = value;
    enabled_map["hsa_soft_queue_create"] = value;
    enabled_map["hsa_queue_destroy"] = value;
    enabled_map["hsa_queue_inactivate"] = value;
    enabled_map["hsa_queue_load_read_index_scacquire"] = value;
    enabled_map["hsa_queue_load_read_index_relaxed"] = value;
    enabled_map["hsa_queue_load_write_index_scacquire"] = value;
    enabled_map["hsa_queue_load_write_index_relaxed"] = value;
    enabled_map["hsa_queue_store_write_index_relaxed"] = value;
    enabled_map["hsa_queue_store_write_index_screlease"] = value;
    enabled_map["hsa_queue_cas_write_index_scacq_screl"] = value;
    enabled_map["hsa_queue_cas_write_index_scacquire"] = value;
    enabled_map["hsa_queue_cas_write_index_relaxed"] = value;
    enabled_map["hsa_queue_cas_write_index_screlease"] = value;
    enabled_map["hsa_queue_add_write_index_scacq_screl"] = value;
    enabled_map["hsa_queue_add_write_index_scacquire"] = value;
    enabled_map["hsa_queue_add_write_index_relaxed"] = value;
    enabled_map["hsa_queue_add_write_index_screlease"] = value;
    enabled_map["hsa_queue_store_read_index_relaxed"] = value;
    enabled_map["hsa_queue_store_read_index_screlease"] = value;
    enabled_map["hsa_agent_iterate_regions"] = value;
    enabled_map["hsa_region_get_info"] = value;
    enabled_map["hsa_memory_register"] = value;
    enabled_map["hsa_memory_deregister"] = value;
    enabled_map["hsa_memory_allocate"] = value;
    enabled_map["hsa_memory_free"] = value;
    enabled_map["hsa_memory_copy"] = value;
    enabled_map["hsa_memory_assign_agent"] = value;
    enabled_map["hsa_signal_create"] = value;
    enabled_map["hsa_signal_destroy"] = value;
    enabled_map["hsa_signal_load_relaxed"] = value;
    enabled_map["hsa_signal_load_scacquire"] = value;
    enabled_map["hsa_signal_store_relaxed"] = value;
    enabled_map["hsa_signal_store_screlease"] = value;
    enabled_map["hsa_signal_wait_relaxed"] = value;
    enabled_map["hsa_signal_wait_scacquire"] = value;
    enabled_map["hsa_signal_and_relaxed"] = value;
    enabled_map["hsa_signal_and_scacquire"] = value;
    enabled_map["hsa_signal_and_screlease"] = value;
    enabled_map["hsa_signal_and_scacq_screl"] = value;
    enabled_map["hsa_signal_or_relaxed"] = value;
    enabled_map["hsa_signal_or_scacquire"] = value;
    enabled_map["hsa_signal_or_screlease"] = value;
    enabled_map["hsa_signal_or_scacq_screl"] = value;
    enabled_map["hsa_signal_xor_relaxed"] = value;
    enabled_map["hsa_signal_xor_scacquire"] = value;
    enabled_map["hsa_signal_xor_screlease"] = value;
    enabled_map["hsa_signal_xor_scacq_screl"] = value;
    enabled_map["hsa_signal_exchange_relaxed"] = value;
    enabled_map["hsa_signal_exchange_scacquire"] = value;
    enabled_map["hsa_signal_exchange_screlease"] = value;
    enabled_map["hsa_signal_exchange_scacq_screl"] = value;
    enabled_map["hsa_signal_add_relaxed"] = value;
    enabled_map["hsa_signal_add_scacquire"] = value;
    enabled_map["hsa_signal_add_screlease"] = value;
    enabled_map["hsa_signal_add_scacq_screl"] = value;
    enabled_map["hsa_signal_subtract_relaxed"] = value;
    enabled_map["hsa_signal_subtract_scacquire"] = value;
    enabled_map["hsa_signal_subtract_screlease"] = value;
    enabled_map["hsa_signal_subtract_scacq_screl"] = value;
    enabled_map["hsa_signal_cas_relaxed"] = value;
    enabled_map["hsa_signal_cas_scacquire"] = value;
    enabled_map["hsa_signal_cas_screlease"] = value;
    enabled_map["hsa_signal_cas_scacq_screl"] = value;

    //===--- Instruction Set Architecture -----------------------------------===//

    enabled_map["hsa_isa_from_name"] = value;
    // Deprecated since v1.1.
    enabled_map["hsa_isa_get_info"] = value;
    // Deprecated since v1.1.
    enabled_map["hsa_isa_compatible"] = value;

    //===--- Code Objects (deprecated) --------------------------------------===//

    // Deprecated since v1.1.
    enabled_map["hsa_code_object_serialize"] = value;
    // Deprecated since v1.1.
    enabled_map["hsa_code_object_deserialize"] = value;
    // Deprecated since v1.1.
    enabled_map["hsa_code_object_destroy"] = value;
    // Deprecated since v1.1.
    enabled_map["hsa_code_object_get_info"] = value;
    // Deprecated since v1.1.
    enabled_map["hsa_code_object_get_symbol"] = value;
    // Deprecated since v1.1.
    enabled_map["hsa_code_symbol_get_info"] = value;
    // Deprecated since v1.1.
    enabled_map["hsa_code_object_iterate_symbols"] = value;

    //===--- Executable -----------------------------------------------------===//

    // Deprecated since v1.1.
    enabled_map["hsa_executable_create"] = value;
    enabled_map["hsa_executable_destroy"] = value;
    // Deprecated since v1.1.
    enabled_map["hsa_executable_load_code_object"] = value;
    enabled_map["hsa_executable_freeze"] = value;
    enabled_map["hsa_executable_get_info"] = value;
    enabled_map["hsa_executable_global_variable_define"] = value;
    enabled_map["hsa_executable_agent_global_variable_define"] = value;
    enabled_map["hsa_executable_readonly_variable_define"] = value;
    enabled_map["hsa_executable_validate"] = value;
    // Deprecated since v1.1.
    enabled_map["hsa_executable_get_symbol"] = value;
    enabled_map["hsa_executable_symbol_get_info"] = value;
    // Deprecated since v1.1.
    enabled_map["hsa_executable_iterate_symbols"] = value;

    //===--- Runtime Notifications ------------------------------------------===//

    enabled_map["hsa_status_string"] = value;

    // Start HSA v1.1 additions
    enabled_map["hsa_extension_get_name"] = value;
    enabled_map["hsa_system_major_extension_supported"] = value;
    enabled_map["hsa_system_get_major_extension_table"] = value;
    enabled_map["hsa_agent_major_extension_supported"] = value;
    enabled_map["hsa_cache_get_info"] = value;
    enabled_map["hsa_agent_iterate_caches"] = value;
    // Silent store optimization is present in all signal ops when no agents are sleeping.
    enabled_map["hsa_signal_store_relaxed"] = value;
    enabled_map["hsa_signal_store_screlease"] = value;
    enabled_map["hsa_signal_group_create"] = value;
    enabled_map["hsa_signal_group_destroy"] = value;
    enabled_map["hsa_signal_group_wait_any_scacquire"] = value;
    enabled_map["hsa_signal_group_wait_any_relaxed"] = value;

    //===--- Instruction Set Architecture - HSA v1.1 additions --------------===//

    enabled_map["hsa_agent_iterate_isas"] = value;
    enabled_map["hsa_isa_get_info_alt"] = value;
    enabled_map["hsa_isa_get_exception_policies"] = value;
    enabled_map["hsa_isa_get_round_method"] = value;
    enabled_map["hsa_wavefront_get_info"] = value;
    enabled_map["hsa_isa_iterate_wavefronts"] = value;

    //===--- Code Objects (deprecated) - HSA v1.1 additions -----------------===//

    // Deprecated since v1.1.
    enabled_map["hsa_code_object_get_symbol_from_name"] = value;

    //===--- Executable - HSA v1.1 additions --------------------------------===//

    enabled_map["hsa_code_object_reader_create_from_file"] = value;
    enabled_map["hsa_code_object_reader_create_from_memory"] = value;
    enabled_map["hsa_code_object_reader_destroy"] = value;
    enabled_map["hsa_executable_create_alt"] = value;
    enabled_map["hsa_executable_load_program_code_object"] = value;
    enabled_map["hsa_executable_load_agent_code_object"] = value;
    enabled_map["hsa_executable_validate_alt"] = value;
    enabled_map["hsa_executable_get_symbol_by_name"] = value;
    enabled_map["hsa_executable_iterate_agent_symbols"] = value;
    enabled_map["hsa_executable_iterate_program_symbols"] = value;
}

static void InitEnabledTableExtApi(bool value) {
    // Initialize function pointers for Amd Extension Api's
    enabled_map["hsa_amd_coherency_get_type"] = value;
    enabled_map["hsa_amd_coherency_set_type"] = value;
    enabled_map["hsa_amd_profiling_set_profiler_enabled"] = value;
    enabled_map["hsa_amd_profiling_async_copy_enable"] = value;
    enabled_map["hsa_amd_profiling_get_dispatch_time"] = value;
    enabled_map["hsa_amd_profiling_get_async_copy_time"] = value;
    enabled_map["hsa_amd_profiling_convert_tick_to_system_domain"] = value;
    enabled_map["hsa_amd_signal_async_handler"] = value;
    enabled_map["hsa_amd_async_function"] = value;
    enabled_map["hsa_amd_signal_wait_any"] = value;
    enabled_map["hsa_amd_queue_cu_set_mask"] = value;
    enabled_map["hsa_amd_memory_pool_get_info"] = value;
    enabled_map["hsa_amd_agent_iterate_memory_pools"] = value;
    enabled_map["hsa_amd_memory_pool_allocate"] = value;
    enabled_map["hsa_amd_memory_pool_free"] = value;
    enabled_map["hsa_amd_memory_async_copy"] = value;
    enabled_map["hsa_amd_agent_memory_pool_get_info"] = value;
    enabled_map["hsa_amd_agents_allow_access"] = value;
    enabled_map["hsa_amd_memory_pool_can_migrate"] = value;
    enabled_map["hsa_amd_memory_migrate"] = value;
    enabled_map["hsa_amd_memory_lock"] = value;
    enabled_map["hsa_amd_memory_unlock"] = value;
    enabled_map["hsa_amd_memory_fill"] = value;
    enabled_map["hsa_amd_interop_map_buffer"] = value;
    enabled_map["hsa_amd_interop_unmap_buffer"] = value;
    enabled_map["hsa_amd_pointer_info"] = value;
    enabled_map["hsa_amd_pointer_info_set_userdata"] = value;
    enabled_map["hsa_amd_ipc_memory_create"] = value;
    enabled_map["hsa_amd_ipc_memory_attach"] = value;
    enabled_map["hsa_amd_ipc_memory_detach"] = value;
    enabled_map["hsa_amd_signal_create"] = value;
    enabled_map["hsa_amd_ipc_signal_create"] = value;
    enabled_map["hsa_amd_ipc_signal_attach"] = value;
    enabled_map["hsa_amd_register_system_event_handler"] = value;
    enabled_map["hsa_amd_queue_intercept_create"] = value;
    enabled_map["hsa_amd_queue_intercept_register"] = value;
    enabled_map["hsa_amd_queue_set_priority"] = value;
    enabled_map["hsa_amd_memory_async_copy_rect"] = value;
#if ENABLE_HSA_AMD_RUNTIME_QUEUE_CREATE_REGISTER
    enabled_map["hsa_amd_runtime_queue_create_register"] = value;
#endif
#if ENABLE_HSA_AMD_MEMORY_LOCK_TO_POOL
    enabled_map["hsa_amd_memory_lock_to_pool"] = value;
#endif
}

static inline bool enabled_check(std::string func)
{
    return enabled_map[func];
}

// shorthands
typedef hsa_ven_amd_aqlprofile_pfn_t pfn_t;
typedef hsa_ven_amd_aqlprofile_event_t event_t;
typedef hsa_ven_amd_aqlprofile_parameter_t parameter_t;
typedef hsa_ven_amd_aqlprofile_profile_t profile_t;
typedef hsa_ext_amd_aql_pm4_packet_t packet_t;
typedef uint32_t packet_word_t;
typedef uint64_t timestamp_t;
static const packet_word_t header_type_mask = (1ul << HSA_PACKET_HEADER_WIDTH_TYPE) - 1;
static const char* kernel_none_;

#define CHECK_STATUS(msg, status) do {  \
    if (status != HSA_STATUS_SUCCESS) { \
        fprintf(stderr, msg);           \
        abort();                        \
    }                                   \
} while (false)

static hsa_status_t executable_symbols_cb(hsa_executable_t exec, hsa_executable_symbol_t symbol, void *data) {
    hsa_symbol_kind_t value = (hsa_symbol_kind_t)0;
    hsa_status_t status = gs_OrigCoreApiTable.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &value);
    CHECK_STATUS("Error in getting symbol info", status);
    if (value == HSA_SYMBOL_KIND_KERNEL) {
        uint64_t addr = 0;
        uint32_t len = 0;
        status = gs_OrigCoreApiTable.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &addr);
        CHECK_STATUS("Error in getting kernel object", status);
        status = gs_OrigCoreApiTable.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &len);
        CHECK_STATUS("Error in getting name len", status);
        char *name = new char[len + 1];
        status = gs_OrigCoreApiTable.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name);
        CHECK_STATUS("Error in getting kernel name", status);
        name[len] = 0;
        auto ret = symbols_map_->insert({addr, name});
        if (ret.second == false) {
            delete[] ret.first->second;
            ret.first->second = name;
        }
    }
    return HSA_STATUS_SUCCESS;
}

static hsa_status_t hsa_executable_freeze_interceptor(hsa_executable_t executable, const char *options) {
    std::lock_guard<mutex_t> lck(kernel_name_mutex_);
    if (symbols_map_ == NULL) symbols_map_ = new symbols_map_t;
    hsa_status_t status = gs_OrigCoreApiTable.hsa_executable_iterate_symbols_fn(executable, executable_symbols_cb, NULL);
    CHECK_STATUS("Error in iterating executable symbols", status);
    return gs_OrigCoreApiTable.hsa_executable_freeze_fn(executable, options);
}

static inline hsa_packet_type_t GetHeaderType(const packet_t* packet) {
    const packet_word_t* header = reinterpret_cast<const packet_word_t*>(packet);
    return static_cast<hsa_packet_type_t>((*header >> HSA_PACKET_HEADER_TYPE) & header_type_mask);
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
    std::lock_guard<mutex_t> lck(kernel_name_mutex_);
    const auto it = symbols_map_->find(addr);
    if (it == symbols_map_->end()) {
        fprintf(stderr, "RTG Tracer: kernel addr (0x%lx) is not found\n", addr);
        abort();
    }
    return it->second;
}

// Demangle C++ symbol name
static const char* cpp_demangle(const char* symname) {
    size_t size = 0;
    int status;
    const char* ret = abi::__cxa_demangle(symname, NULL, &size, &status);
    return (ret != 0) ? ret : strdup(symname);
}

static const char* QueryKernelName(uint64_t kernel_object, const amd_kernel_code_t* kernel_code) {
    const char* kernel_symname = GetKernelNameRef(kernel_object);
    if (HCC_PROFILE) {
        return kernel_symname;
    }
    else {
        return cpp_demangle(kernel_symname);
    }
}


static void intercept_callback(
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
        hsa_signal_t new_signal{0};
        hsa_signal_t original_signal{0};
        bool owns_orig_signal = false;

        ++host_count_signals;
        hsa_status_t status = gs_OrigCoreApiTable.hsa_signal_create_fn(1, 0, nullptr, &new_signal);
        if (status != HSA_STATUS_SUCCESS) {
            fprintf(stderr, "RTG Tracer: failed to allocate signal\n");
            exit(EXIT_FAILURE);
        }

        // Checking for dispatch packet type
        if (type == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
            ++RTG::host_count_dispatches;
            const hsa_kernel_dispatch_packet_t* dispatch_packet =
                reinterpret_cast<const hsa_kernel_dispatch_packet_t*>(packet);

            uint64_t kernel_object = dispatch_packet->kernel_object;
            const amd_kernel_code_t* kernel_code = GetKernelCode(kernel_object);
            const uint64_t kernel_symbol = kernel_code->runtime_loader_kernel_symbol;
            const char* kernel_name = QueryKernelName(kernel_object, kernel_code);

            original_signal = dispatch_packet->completion_signal;
            if (!original_signal.handle) {
                owns_orig_signal = true;
                status = gs_OrigCoreApiTable.hsa_signal_create_fn(1, 0, nullptr, &original_signal);
                if (status != HSA_STATUS_SUCCESS) {
                    fprintf(stderr, "RTG Tracer: failed to allocate signal\n");
                    exit(EXIT_FAILURE);
                }
            }
            const_cast<hsa_kernel_dispatch_packet_t*>(dispatch_packet)->completion_signal = new_signal;
            pool.push(SignalWaiter, new SignalCallbackData(kernel_name, data_, new_signal, original_signal, owns_orig_signal, dispatch_packet));
        }
        else if (type == HSA_PACKET_TYPE_BARRIER_AND || type == HSA_PACKET_TYPE_BARRIER_OR) {
            ++RTG::host_count_barriers;
            const hsa_barrier_and_packet_t* barrier_packet =
                reinterpret_cast<const hsa_barrier_and_packet_t*>(packet);

            original_signal = barrier_packet->completion_signal;
            if (!original_signal.handle) {
                owns_orig_signal = true;
                status = gs_OrigCoreApiTable.hsa_signal_create_fn(1, 0, nullptr, &original_signal);
                if (status != HSA_STATUS_SUCCESS) {
                    fprintf(stderr, "RTG Tracer: failed to allocate signal\n");
                    exit(EXIT_FAILURE);
                }
            }
            const_cast<hsa_barrier_and_packet_t*>(barrier_packet)->completion_signal = new_signal;
            pool.push(SignalWaiter, new SignalCallbackData(data_, new_signal, original_signal, owns_orig_signal, barrier_packet));
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
        }
        else {
            fprintf(stderr, "RTG Tracer: fatal, intercept callback missing original signal\n");
            exit(EXIT_FAILURE);
        }

    }
}

static void* hip_activity_callback(uint32_t cid, activity_record_t* record, const void* data, void* arg)
{
    return (hip_api_data_t*)hip_api_data[cid];
}

static void* hip_api_callback(uint32_t domain, uint32_t cid, const void* data_, void* arg)
{
    hip_api_data_t *data = (hip_api_data_t*)data_;
    if (data->phase == 0) {
        // hack for now, stash tick in the correlation ID
        data->correlation_id = tick();
    }
    else {
        std::string &func = hip_api_names[cid];
        int pid_ = pid();
        std::string tid_ = tid();
        const char* args = hipApiString((hip_api_id_t)cid, data);
        uint64_t tick_ = data->correlation_id;
        uint64_t ticks = tick() - tick_;                                   \
        int localStatus = 0;
        if (RTG_HIP_API_ARGS) {
            LOG_HIP_ARGS
        }
        else {
            LOG_HIP
        }
        // Now that we're done with the api data, zero it for the next time.
        // Otherwise, phase is always wrong because HIP doesn't set the phase to 0 during API start.
        memset(data, 0, sizeof(hip_api_data_t));
    }
    //fprintf(stderr, "HI FROM hip_api_callback domain=%u cid=%u data=%p arg=%p correlation_id=%lu phase=%u name=%s\n",
    //        domain, cid, data, arg, data->correlation_id, data->phase, hip_api_names[cid].c_str());
    return NULL;
}

} // namespace RTG

extern "C" bool OnLoad(void *pTable,
        uint64_t runtimeVersion, uint64_t failedToolCount,
        const char *const *pFailedToolNames)
{
    fprintf(stderr, "RTG Tracer: Loading\n");

    Flag::init_all();

    std::string outname = RTG_FILENAME;
    // PID is needed to avoid clashses in multi-process use cases
    outname += ".";
    outname += RTG::pidstr();
    fprintf(stderr, "RTG Tracer: Filename %s\n", outname.c_str());
    RTG::stream = fopen(outname.c_str(), "w");

    if (HCC_PROFILE) {
        fprintf(stderr, "RTG Tracer: HCC_PROFILE=2 mode\n");
    }
    else {
        RTG::InitEnabledTable(RTG_HSA_API_FILTER, RTG_HSA_API_FILTER_OUT);

        std::vector<std::string> tokens_keep = RTG::split(RTG_HIP_API_FILTER, ',');
        std::vector<std::string> tokens_prune = RTG::split(RTG_HIP_API_FILTER_OUT, ',');
        for (int i=0; i<HIP_API_ID_NUMBER; ++i) {
            bool keep = false;
            std::string name = hip_api_name(i);
            RTG::hip_api_names[i] = name;
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
            if (keep) {
                hipRegisterActivityCallback(i, (void*)RTG::hip_activity_callback, NULL);
                hipRegisterApiCallback(i, (void*)RTG::hip_api_callback, NULL);
            }
            if (RTG_VERBOSE) {
                fprintf(stderr, "RTG Tracer: HIP API %s %s\n", keep ? "keep" : "skip", name.c_str());
            }
        }
    }

    return RTG::InitHsaTable(reinterpret_cast<HsaApiTable*>(pTable));
}

extern "C" void OnUnload()
{
    fprintf(stderr, "RTG Tracer: Unloading\n");
    RTG::RestoreHsaTable(RTG::gs_OrigHsaTable);
    fclose(RTG::stream);
}

__attribute__((destructor)) static void destroy() {
    fprintf(stderr, "RTG Tracer: Destructing\n");
    if (RTG_PROFILE || RTG_PROFILE_COPY) {
        fprintf(stderr, "RTG Tracer: host_count_dispatches=%u\n", LOAD(RTG::host_count_dispatches));
        fprintf(stderr, "RTG Tracer:   cb_count_dispatches=%u\n", LOAD(RTG::cb_count_dispatches));
        fprintf(stderr, "RTG Tracer:   host_count_barriers=%u\n", LOAD(RTG::host_count_barriers));
        fprintf(stderr, "RTG Tracer:     cb_count_barriers=%u\n", LOAD(RTG::cb_count_barriers));
        fprintf(stderr, "RTG Tracer:     host_count_copies=%u\n", LOAD(RTG::host_count_copies));
        fprintf(stderr, "RTG Tracer:       cb_count_copies=%u\n", LOAD(RTG::cb_count_copies));
        fprintf(stderr, "RTG Tracer:    host_count_signals=%u\n", LOAD(RTG::host_count_signals));
        fprintf(stderr, "RTG Tracer:      cb_count_signals=%u\n", LOAD(RTG::cb_count_signals));
    }
    for (int i=0; i<5; ++i) {
        if (RTG::host_count_dispatches != RTG::cb_count_dispatches
                || RTG::host_count_barriers != RTG::cb_count_barriers) {
            fprintf(stderr, "RTG Tracer: not all callbacks have completed, waiting... dispatches %u vs %u barriers %u vs %u signals %u vs %u\n",
                    LOAD(RTG::host_count_dispatches),
                    LOAD(RTG::cb_count_dispatches),
                    LOAD(RTG::host_count_barriers),
                    LOAD(RTG::cb_count_barriers),
                    LOAD(RTG::host_count_signals),
                    LOAD(RTG::cb_count_signals)
            );
            sleep(2);
        }
    }
}
