#include <cstdio>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

//#include <hsa/hsa.h>
//#include <hsa/hsa_ext_image.h>
//#include <hsa/hsa_ext_amd.h>
//#include <hsa/hsa_ext_finalize.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_loader.h>
//#include <hsakmt.h>

#define ENABLE_HSA_AMD_MEMORY_LOCK_TO_POOL 0
#define ENABLE_HSA_AMD_RUNTIME_QUEUE_CREATE_REGISTER 0

// The HSA Runtime's versions of HSA core API functions
static CoreApiTable gs_OrigCoreApiTable;

// The HSA Runtime's versions of HSA ext API functions
static AmdExtTable gs_OrigExtApiTable;

// The HSA Runtime's versions of HSA loader ext API functions
static hsa_ven_amd_loader_1_01_pfn_t gs_OrigLoaderExtTable;

static void InitCoreApiTable(CoreApiTable* table);
static void InitAmdExtTable(AmdExtTable* table);

// Helper functions to convert function arguments into strings.
// Handles POD data types as well as enumerations.
// The implementation uses C++11 variadic templates and template specialization.

// This is the default which works for most types:
template <typename T>
inline std::string ToString(T v) {
    std::ostringstream ss;
    ss << v;
    return ss.str();
};

// Catch empty arguments case
inline std::string ToString() { return (""); }

// C++11 variadic template - peels off first argument, converts to string, and calls itself again to
// peel the next arg. Strings are automatically separated by comma+space.
template <typename T, typename... Args>
inline std::string ToString(T first, Args... args) {
    return ToString(first) + ", " + ToString(args...);
}

static inline int pid() {
    return getpid();
}

static inline std::string tid() {
    std::ostringstream os;
    os << std::this_thread::get_id();
    return os.str();
}

static inline suseconds_t tick() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

#define API_TRACE(...) \
    std::string apiStr = std::string(__func__) + " (" + ToString(__VA_ARGS__) + ')'; \
    std::string tidStr = tid(); \
    fprintf(stderr, "<<hsa-api pid:%d tid:%s @%lu %s\n", pid(), tidStr.c_str(), tick(), apiStr.c_str());

#define LOG_STATUS(status)                                                                          \
    ({                                                                                              \
        hipError_t localHipStatus = hipStatus; /*local copy so hipStatus only evaluated once*/      \
        tls_lastHipError = localHipStatus;                                                          \
                                                                                                    \
        if ((COMPILE_HIP_TRACE_API & 0x2) && HIP_TRACE_API & (1 << TRACE_ALL)) {                    \
            auto ticks = getTicks() - hipApiStartTick;                                              \
            fprintf(stderr, "  %ship-api pid:%d tid:%d.%lu %-30s ret=%2d (%s)>> +%lu ns%s\n",       \
                    (localHipStatus == 0) ? API_COLOR : KRED, tls_tidInfo.pid(), tls_tidInfo.tid(), \
                    tls_tidInfo.apiSeqNum(), __func__, localHipStatus,                              \
                    ihipErrorString(localHipStatus), ticks, API_COLOR_END);                         \
        }                                                                                           \
        if (HIP_PROFILE_API) {                                                                      \
            MARKER_END();                                                                           \
        }                                                                                           \
        localHipStatus;                                                                             \
    })



bool InitHsaTable(HsaApiTable* pTable)
{
    if (pTable == nullptr) {
        fprintf(stderr, "RTG HSA Tracer: HSA Runtime provided a nullptr API Table");
        return false;
    }

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
        fprintf(stderr, "RTG HSA Tracer: Cannot get loader extension function table");
        return false;
    }

    InitCoreApiTable(pTable->core_);
    InitAmdExtTable(pTable->amd_ext_);

    return true;
}

extern "C" bool OnLoad(void *pTable,
        uint64_t runtimeVersion, uint64_t failedToolCount,
        const char *const *pFailedToolNames)
{
    fprintf(stderr, "RTG HSA Tracer: Loading\n");
    return InitHsaTable(reinterpret_cast<HsaApiTable*>(pTable));
}

extern "C" void OnUnload()
{
    fprintf(stderr, "RTG HSA Tracer: Unloading\n");
}

hsa_status_t rtg_hsa_init() {
    return gs_OrigCoreApiTable.hsa_init_fn();
}

hsa_status_t rtg_hsa_shut_down() {
    return gs_OrigCoreApiTable.hsa_shut_down_fn();
}

hsa_status_t rtg_hsa_system_get_info(hsa_system_info_t attribute, void* value) {
    return gs_OrigCoreApiTable.hsa_system_get_info_fn(attribute, value);
}

hsa_status_t rtg_hsa_extension_get_name(uint16_t extension, const char** name) {
    return gs_OrigCoreApiTable.hsa_extension_get_name_fn(extension, name);
}

hsa_status_t rtg_hsa_system_extension_supported(uint16_t extension, uint16_t version_major, uint16_t version_minor, bool* result) {
    return gs_OrigCoreApiTable.hsa_system_extension_supported_fn(extension, version_major, version_minor, result);
}

hsa_status_t rtg_hsa_system_major_extension_supported(uint16_t extension, uint16_t version_major, uint16_t* version_minor, bool* result) {
    return gs_OrigCoreApiTable.hsa_system_major_extension_supported_fn(extension, version_major, version_minor, result);
}

hsa_status_t rtg_hsa_system_get_extension_table(uint16_t extension, uint16_t version_major, uint16_t version_minor, void* table) {
    return gs_OrigCoreApiTable.hsa_system_get_extension_table_fn( extension, version_major, version_minor, table);
}

hsa_status_t rtg_hsa_system_get_major_extension_table(uint16_t extension, uint16_t version_major, size_t table_length, void* table) {
    return gs_OrigCoreApiTable.hsa_system_get_major_extension_table_fn(extension, version_major, table_length, table);
}

hsa_status_t rtg_hsa_iterate_agents(hsa_status_t (*callback)(hsa_agent_t agent, void* data), void* data) {
    return gs_OrigCoreApiTable.hsa_iterate_agents_fn(callback, data);
}

hsa_status_t rtg_hsa_agent_get_info(hsa_agent_t agent, hsa_agent_info_t attribute, void* value) {
    return gs_OrigCoreApiTable.hsa_agent_get_info_fn(agent, attribute, value);
}

hsa_status_t rtg_hsa_agent_get_exception_policies(hsa_agent_t agent, hsa_profile_t profile, uint16_t* mask) {
    return gs_OrigCoreApiTable.hsa_agent_get_exception_policies_fn(agent, profile, mask);
}

hsa_status_t rtg_hsa_cache_get_info(hsa_cache_t cache, hsa_cache_info_t attribute, void* value) {
    return gs_OrigCoreApiTable.hsa_cache_get_info_fn(cache, attribute, value);
}

hsa_status_t rtg_hsa_agent_iterate_caches(hsa_agent_t agent, hsa_status_t (*callback)(hsa_cache_t cache, void* data), void* value) {
    return gs_OrigCoreApiTable.hsa_agent_iterate_caches_fn(agent, callback, value);
}

hsa_status_t rtg_hsa_agent_extension_supported(uint16_t extension, hsa_agent_t agent, uint16_t version_major, uint16_t version_minor, bool* result) {
    return gs_OrigCoreApiTable.hsa_agent_extension_supported_fn(extension, agent, version_major, version_minor, result);
}

hsa_status_t rtg_hsa_agent_major_extension_supported(uint16_t extension, hsa_agent_t agent, uint16_t version_major, uint16_t* version_minor, bool* result) {
    return gs_OrigCoreApiTable.hsa_agent_major_extension_supported_fn(extension, agent, version_major, version_minor, result);
}

hsa_status_t rtg_hsa_queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type, void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data), void* data, uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t** queue) {
    return gs_OrigCoreApiTable.hsa_queue_create_fn(agent, size, type, callback, data, private_segment_size, group_segment_size, queue);
}

hsa_status_t rtg_hsa_soft_queue_create(hsa_region_t region, uint32_t size, hsa_queue_type32_t type, uint32_t features, hsa_signal_t completion_signal, hsa_queue_t** queue) {
    return gs_OrigCoreApiTable.hsa_soft_queue_create_fn(region, size, type, features, completion_signal, queue);
}

hsa_status_t rtg_hsa_queue_destroy(hsa_queue_t* queue) {
    return gs_OrigCoreApiTable.hsa_queue_destroy_fn(queue);
}

hsa_status_t rtg_hsa_queue_inactivate(hsa_queue_t* queue) {
    return gs_OrigCoreApiTable.hsa_queue_inactivate_fn(queue);
}

uint64_t rtg_hsa_queue_load_read_index_scacquire(const hsa_queue_t* queue) {
    return gs_OrigCoreApiTable.hsa_queue_load_read_index_scacquire_fn(queue);
}

uint64_t rtg_hsa_queue_load_read_index_relaxed(const hsa_queue_t* queue) {
    return gs_OrigCoreApiTable.hsa_queue_load_read_index_relaxed_fn(queue);
}

uint64_t rtg_hsa_queue_load_write_index_scacquire(const hsa_queue_t* queue) {
    return gs_OrigCoreApiTable.hsa_queue_load_write_index_scacquire_fn(queue);
}

uint64_t rtg_hsa_queue_load_write_index_relaxed(const hsa_queue_t* queue) {
    return gs_OrigCoreApiTable.hsa_queue_load_write_index_relaxed_fn(queue);
}

void rtg_hsa_queue_store_write_index_relaxed(const hsa_queue_t* queue, uint64_t value) {
    return gs_OrigCoreApiTable.hsa_queue_store_write_index_relaxed_fn(queue, value);
}

void rtg_hsa_queue_store_write_index_screlease(const hsa_queue_t* queue, uint64_t value) {
    return gs_OrigCoreApiTable.hsa_queue_store_write_index_screlease_fn(queue, value);
}

uint64_t rtg_hsa_queue_cas_write_index_scacq_screl(const hsa_queue_t* queue, uint64_t expected, uint64_t value) {
    return gs_OrigCoreApiTable.hsa_queue_cas_write_index_scacq_screl_fn(queue, expected, value);
}

uint64_t rtg_hsa_queue_cas_write_index_scacquire(const hsa_queue_t* queue, uint64_t expected, uint64_t value) {
    return gs_OrigCoreApiTable.hsa_queue_cas_write_index_scacquire_fn(queue, expected, value);
}

uint64_t rtg_hsa_queue_cas_write_index_relaxed(const hsa_queue_t* queue, uint64_t expected, uint64_t value) {
    return gs_OrigCoreApiTable.hsa_queue_cas_write_index_relaxed_fn(queue, expected, value);
}

uint64_t rtg_hsa_queue_cas_write_index_screlease(const hsa_queue_t* queue, uint64_t expected, uint64_t value) {
    return gs_OrigCoreApiTable.hsa_queue_cas_write_index_screlease_fn(queue, expected, value);
}

uint64_t rtg_hsa_queue_add_write_index_scacq_screl(const hsa_queue_t* queue, uint64_t value) {
    return gs_OrigCoreApiTable.hsa_queue_add_write_index_scacq_screl_fn(queue, value);
}

uint64_t rtg_hsa_queue_add_write_index_scacquire(const hsa_queue_t* queue, uint64_t value) {
    return gs_OrigCoreApiTable.hsa_queue_add_write_index_scacquire_fn(queue, value);
}

uint64_t rtg_hsa_queue_add_write_index_relaxed(const hsa_queue_t* queue, uint64_t value) {
    return gs_OrigCoreApiTable.hsa_queue_add_write_index_relaxed_fn(queue, value);
}

uint64_t rtg_hsa_queue_add_write_index_screlease(const hsa_queue_t* queue, uint64_t value) {
    return gs_OrigCoreApiTable.hsa_queue_add_write_index_screlease_fn(queue, value);
}

void rtg_hsa_queue_store_read_index_relaxed(const hsa_queue_t* queue, uint64_t value) {
    return gs_OrigCoreApiTable.hsa_queue_store_read_index_relaxed_fn(queue, value);
}

void rtg_hsa_queue_store_read_index_screlease(const hsa_queue_t* queue, uint64_t value) {
    return gs_OrigCoreApiTable.hsa_queue_store_read_index_screlease_fn(queue, value);
}

hsa_status_t rtg_hsa_agent_iterate_regions(hsa_agent_t agent, hsa_status_t (*callback)(hsa_region_t region, void* data), void* data) {
    return gs_OrigCoreApiTable.hsa_agent_iterate_regions_fn(agent, callback, data);
}

hsa_status_t rtg_hsa_region_get_info(hsa_region_t region, hsa_region_info_t attribute, void* value) {
    return gs_OrigCoreApiTable.hsa_region_get_info_fn(region, attribute, value);
}

hsa_status_t rtg_hsa_memory_register(void* address, size_t size) {
    return gs_OrigCoreApiTable.hsa_memory_register_fn(address, size);
}

hsa_status_t rtg_hsa_memory_deregister(void* address, size_t size) {
    return gs_OrigCoreApiTable.hsa_memory_deregister_fn(address, size);
}

hsa_status_t rtg_hsa_memory_allocate(hsa_region_t region, size_t size, void** ptr) {
    return gs_OrigCoreApiTable.hsa_memory_allocate_fn(region, size, ptr);
}

hsa_status_t rtg_hsa_memory_free(void* ptr) {
    return gs_OrigCoreApiTable.hsa_memory_free_fn(ptr);
}

hsa_status_t rtg_hsa_memory_copy(void* dst, const void* src, size_t size) {
    return gs_OrigCoreApiTable.hsa_memory_copy_fn(dst, src, size);
}

hsa_status_t rtg_hsa_memory_assign_agent(void* ptr, hsa_agent_t agent, hsa_access_permission_t access) {
    return gs_OrigCoreApiTable.hsa_memory_assign_agent_fn(ptr, agent, access);
}

hsa_status_t rtg_hsa_signal_create(hsa_signal_value_t initial_value, uint32_t num_consumers, const hsa_agent_t* consumers, hsa_signal_t* signal) {
    return gs_OrigCoreApiTable.hsa_signal_create_fn(initial_value, num_consumers, consumers, signal);
}

hsa_status_t rtg_hsa_signal_destroy(hsa_signal_t signal) {
    return gs_OrigCoreApiTable.hsa_signal_destroy_fn(signal);
}

hsa_signal_value_t rtg_hsa_signal_load_relaxed(hsa_signal_t signal) {
    return gs_OrigCoreApiTable.hsa_signal_load_relaxed_fn(signal);
}

hsa_signal_value_t rtg_hsa_signal_load_scacquire(hsa_signal_t signal) {
    return gs_OrigCoreApiTable.hsa_signal_load_scacquire_fn(signal);
}

void rtg_hsa_signal_store_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_store_relaxed_fn(signal, value);
}

void rtg_hsa_signal_store_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_store_screlease_fn(signal, value);
}

void rtg_hsa_signal_silent_store_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_silent_store_relaxed_fn(signal, value);
}

void rtg_hsa_signal_silent_store_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_silent_store_screlease_fn(signal, value);
}

hsa_signal_value_t rtg_hsa_signal_wait_relaxed(hsa_signal_t signal, hsa_signal_condition_t condition, hsa_signal_value_t compare_value, uint64_t timeout_hint, hsa_wait_state_t wait_expectancy_hint) {
    return gs_OrigCoreApiTable.hsa_signal_wait_relaxed_fn(signal, condition, compare_value, timeout_hint, wait_expectancy_hint);
}

hsa_signal_value_t rtg_hsa_signal_wait_scacquire(hsa_signal_t signal, hsa_signal_condition_t condition, hsa_signal_value_t compare_value, uint64_t timeout_hint, hsa_wait_state_t wait_expectancy_hint) {
    return gs_OrigCoreApiTable.hsa_signal_wait_scacquire_fn(signal, condition, compare_value, timeout_hint, wait_expectancy_hint);
}

hsa_status_t rtg_hsa_signal_group_create(uint32_t num_signals, const hsa_signal_t* signals, uint32_t num_consumers, const hsa_agent_t* consumers, hsa_signal_group_t* signal_group) {
    return gs_OrigCoreApiTable.hsa_signal_group_create_fn(num_signals, signals, num_consumers, consumers, signal_group);
}

hsa_status_t rtg_hsa_signal_group_destroy(hsa_signal_group_t signal_group) {
    return gs_OrigCoreApiTable.hsa_signal_group_destroy_fn(signal_group);
}

hsa_status_t rtg_hsa_signal_group_wait_any_relaxed(hsa_signal_group_t signal_group, const hsa_signal_condition_t* conditions, const hsa_signal_value_t* compare_values, hsa_wait_state_t wait_state_hint, hsa_signal_t* signal, hsa_signal_value_t* value) {
    return gs_OrigCoreApiTable.hsa_signal_group_wait_any_relaxed_fn(signal_group, conditions, compare_values, wait_state_hint, signal, value);
}

hsa_status_t rtg_hsa_signal_group_wait_any_scacquire(hsa_signal_group_t signal_group, const hsa_signal_condition_t* conditions, const hsa_signal_value_t* compare_values, hsa_wait_state_t wait_state_hint, hsa_signal_t* signal, hsa_signal_value_t* value) {
    return gs_OrigCoreApiTable.hsa_signal_group_wait_any_scacquire_fn( signal_group, conditions, compare_values, wait_state_hint, signal, value);
}

void rtg_hsa_signal_and_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_and_relaxed_fn(signal, value);
}

void rtg_hsa_signal_and_scacquire(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_and_scacquire_fn(signal, value);
}

void rtg_hsa_signal_and_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_and_screlease_fn(signal, value);
}

void rtg_hsa_signal_and_scacq_screl(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_and_scacq_screl_fn(signal, value);
}

void rtg_hsa_signal_or_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_or_relaxed_fn(signal, value);
}

void rtg_hsa_signal_or_scacquire(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_or_scacquire_fn(signal, value);
}

void rtg_hsa_signal_or_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_or_screlease_fn(signal, value);
}

void rtg_hsa_signal_or_scacq_screl(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_or_scacq_screl_fn(signal, value);
}

void rtg_hsa_signal_xor_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_xor_relaxed_fn(signal, value);
}

void rtg_hsa_signal_xor_scacquire(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_xor_scacquire_fn(signal, value);
}

void rtg_hsa_signal_xor_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_xor_screlease_fn(signal, value);
}

void rtg_hsa_signal_xor_scacq_screl(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_xor_scacq_screl_fn(signal, value);
}

void rtg_hsa_signal_add_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_add_relaxed_fn(signal, value);
}

void rtg_hsa_signal_add_scacquire(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_add_scacquire_fn(signal, value);
}

void rtg_hsa_signal_add_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_add_screlease_fn(signal, value);
}

void rtg_hsa_signal_add_scacq_screl(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_add_scacq_screl_fn(signal, value);
}

void rtg_hsa_signal_subtract_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_subtract_relaxed_fn(signal, value);
}

void rtg_hsa_signal_subtract_scacquire(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_subtract_scacquire_fn(signal, value);
}

void rtg_hsa_signal_subtract_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_subtract_screlease_fn(signal, value);
}

void rtg_hsa_signal_subtract_scacq_screl(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_subtract_scacq_screl_fn(signal, value);
}

hsa_signal_value_t rtg_hsa_signal_exchange_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_exchange_relaxed_fn(signal, value);
}

hsa_signal_value_t rtg_hsa_signal_exchange_scacquire(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_exchange_scacquire_fn(signal, value);
}

hsa_signal_value_t rtg_hsa_signal_exchange_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_exchange_screlease_fn(signal, value);
}

hsa_signal_value_t rtg_hsa_signal_exchange_scacq_screl(hsa_signal_t signal, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_exchange_scacq_screl_fn(signal, value);
}

hsa_signal_value_t rtg_hsa_signal_cas_relaxed(hsa_signal_t signal, hsa_signal_value_t expected, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_cas_relaxed_fn(signal, expected, value);
}

hsa_signal_value_t rtg_hsa_signal_cas_scacquire(hsa_signal_t signal, hsa_signal_value_t expected, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_cas_scacquire_fn(signal, expected, value);
}

hsa_signal_value_t rtg_hsa_signal_cas_screlease(hsa_signal_t signal, hsa_signal_value_t expected, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_cas_screlease_fn(signal, expected, value);
}

hsa_signal_value_t rtg_hsa_signal_cas_scacq_screl(hsa_signal_t signal, hsa_signal_value_t expected, hsa_signal_value_t value) {
    return gs_OrigCoreApiTable.hsa_signal_cas_scacq_screl_fn(signal, expected, value);
}

//===--- Instruction Set Architecture -------------------------------------===//

hsa_status_t rtg_hsa_isa_from_name(const char *name, hsa_isa_t *isa) {
    return gs_OrigCoreApiTable.hsa_isa_from_name_fn(name, isa);
}

hsa_status_t rtg_hsa_agent_iterate_isas(hsa_agent_t agent, hsa_status_t (*callback)(hsa_isa_t isa, void *data), void *data) {
    return gs_OrigCoreApiTable.hsa_agent_iterate_isas_fn(agent, callback, data);
}

/* deprecated */
hsa_status_t rtg_hsa_isa_get_info(hsa_isa_t isa, hsa_isa_info_t attribute, uint32_t index, void *value) {
    return gs_OrigCoreApiTable.hsa_isa_get_info_fn(isa, attribute, index, value);
}

hsa_status_t rtg_hsa_isa_get_info_alt( hsa_isa_t isa, hsa_isa_info_t attribute, void *value) {
    return gs_OrigCoreApiTable.hsa_isa_get_info_alt_fn(isa, attribute, value);
}

hsa_status_t rtg_hsa_isa_get_exception_policies( hsa_isa_t isa, hsa_profile_t profile, uint16_t *mask) {
    return gs_OrigCoreApiTable.hsa_isa_get_exception_policies_fn(isa, profile, mask);
}

hsa_status_t rtg_hsa_isa_get_round_method( hsa_isa_t isa, hsa_fp_type_t fp_type, hsa_flush_mode_t flush_mode, hsa_round_method_t *round_method) {
    return gs_OrigCoreApiTable.hsa_isa_get_round_method_fn(isa, fp_type, flush_mode, round_method);
}

hsa_status_t rtg_hsa_wavefront_get_info( hsa_wavefront_t wavefront, hsa_wavefront_info_t attribute, void *value) {
    return gs_OrigCoreApiTable.hsa_wavefront_get_info_fn(wavefront, attribute, value);
}

hsa_status_t rtg_hsa_isa_iterate_wavefronts(hsa_isa_t isa, hsa_status_t (*callback)(hsa_wavefront_t wavefront, void *data), void *data) {
    return gs_OrigCoreApiTable.hsa_isa_iterate_wavefronts_fn(isa, callback, data);
}

/* deprecated */
hsa_status_t rtg_hsa_isa_compatible(hsa_isa_t code_object_isa, hsa_isa_t agent_isa, bool *result) {
    return gs_OrigCoreApiTable.hsa_isa_compatible_fn(code_object_isa, agent_isa, result);
}

//===--- Code Objects (deprecated) ----------------------------------------===//

/* deprecated */
hsa_status_t rtg_hsa_code_object_serialize(hsa_code_object_t code_object, hsa_status_t (*alloc_callback)(size_t size, hsa_callback_data_t data, void **address), hsa_callback_data_t callback_data, const char *options, void **serialized_code_object, size_t *serialized_code_object_size) {
    return gs_OrigCoreApiTable.hsa_code_object_serialize_fn(code_object, alloc_callback, callback_data, options, serialized_code_object, serialized_code_object_size);
}

/* deprecated */
hsa_status_t rtg_hsa_code_object_deserialize(void *serialized_code_object, size_t serialized_code_object_size, const char *options, hsa_code_object_t *code_object) {
    return gs_OrigCoreApiTable.hsa_code_object_deserialize_fn(serialized_code_object, serialized_code_object_size, options, code_object);
}

/* deprecated */
hsa_status_t rtg_hsa_code_object_destroy(hsa_code_object_t code_object) {
    return gs_OrigCoreApiTable.hsa_code_object_destroy_fn(code_object);
}

/* deprecated */
hsa_status_t rtg_hsa_code_object_get_info(hsa_code_object_t code_object, hsa_code_object_info_t attribute, void *value) {
    return gs_OrigCoreApiTable.hsa_code_object_get_info_fn(code_object, attribute, value);
}

/* deprecated */
hsa_status_t rtg_hsa_code_object_get_symbol(hsa_code_object_t code_object, const char *symbol_name, hsa_code_symbol_t *symbol) {
    return gs_OrigCoreApiTable.hsa_code_object_get_symbol_fn(code_object, symbol_name, symbol);
}

/* deprecated */
hsa_status_t rtg_hsa_code_object_get_symbol_from_name(hsa_code_object_t code_object, const char *module_name, const char *symbol_name, hsa_code_symbol_t *symbol) {
    return gs_OrigCoreApiTable.hsa_code_object_get_symbol_from_name_fn(code_object, module_name, symbol_name, symbol);
}

/* deprecated */
hsa_status_t rtg_hsa_code_symbol_get_info(hsa_code_symbol_t code_symbol, hsa_code_symbol_info_t attribute, void *value) {
    return gs_OrigCoreApiTable.hsa_code_symbol_get_info_fn(code_symbol, attribute, value);
}

/* deprecated */
hsa_status_t rtg_hsa_code_object_iterate_symbols(hsa_code_object_t code_object, hsa_status_t (*callback)(hsa_code_object_t code_object, hsa_code_symbol_t symbol, void *data), void *data) {
    return gs_OrigCoreApiTable.hsa_code_object_iterate_symbols_fn(code_object, callback, data);
}

//===--- Executable -------------------------------------------------------===//

hsa_status_t rtg_hsa_code_object_reader_create_from_file( hsa_file_t file, hsa_code_object_reader_t *code_object_reader) {
    return gs_OrigCoreApiTable.hsa_code_object_reader_create_from_file_fn(file, code_object_reader);
}

hsa_status_t rtg_hsa_code_object_reader_create_from_memory(const void *code_object, size_t size, hsa_code_object_reader_t *code_object_reader) {
    return gs_OrigCoreApiTable.hsa_code_object_reader_create_from_memory_fn(code_object, size, code_object_reader);
}

hsa_status_t rtg_hsa_code_object_reader_destroy(hsa_code_object_reader_t code_object_reader) {
    return gs_OrigCoreApiTable.hsa_code_object_reader_destroy_fn(code_object_reader);
}

/* deprecated */
hsa_status_t rtg_hsa_executable_create(hsa_profile_t profile, hsa_executable_state_t executable_state, const char *options, hsa_executable_t *executable) {
    return gs_OrigCoreApiTable.hsa_executable_create_fn(profile, executable_state, options, executable);
}

hsa_status_t rtg_hsa_executable_create_alt(hsa_profile_t profile, hsa_default_float_rounding_mode_t default_float_rounding_mode, const char *options, hsa_executable_t *executable) {
    return gs_OrigCoreApiTable.hsa_executable_create_alt_fn(profile, default_float_rounding_mode, options, executable);
}

hsa_status_t rtg_hsa_executable_destroy(hsa_executable_t executable) {
    return gs_OrigCoreApiTable.hsa_executable_destroy_fn(executable);
}

/* deprecated */
hsa_status_t rtg_hsa_executable_load_code_object(hsa_executable_t executable, hsa_agent_t agent, hsa_code_object_t code_object, const char *options) {
    return gs_OrigCoreApiTable.hsa_executable_load_code_object_fn(executable, agent, code_object, options);
}

hsa_status_t rtg_hsa_executable_load_program_code_object(hsa_executable_t executable, hsa_code_object_reader_t code_object_reader, const char *options, hsa_loaded_code_object_t *loaded_code_object) {
    return gs_OrigCoreApiTable.hsa_executable_load_program_code_object_fn(executable, code_object_reader, options, loaded_code_object);
}

hsa_status_t rtg_hsa_executable_load_agent_code_object(hsa_executable_t executable, hsa_agent_t agent, hsa_code_object_reader_t code_object_reader, const char *options, hsa_loaded_code_object_t *loaded_code_object) {
    return gs_OrigCoreApiTable.hsa_executable_load_agent_code_object_fn(executable, agent, code_object_reader, options, loaded_code_object);
}

hsa_status_t rtg_hsa_executable_freeze(hsa_executable_t executable, const char *options) {
    return gs_OrigCoreApiTable.hsa_executable_freeze_fn(executable, options);
}

hsa_status_t rtg_hsa_executable_get_info(hsa_executable_t executable, hsa_executable_info_t attribute, void *value) {
    return gs_OrigCoreApiTable.hsa_executable_get_info_fn(executable, attribute, value);
}

hsa_status_t rtg_hsa_executable_global_variable_define(hsa_executable_t executable, const char *variable_name, void *address) {
    return gs_OrigCoreApiTable.hsa_executable_global_variable_define_fn(executable, variable_name, address);
}

hsa_status_t rtg_hsa_executable_agent_global_variable_define(hsa_executable_t executable, hsa_agent_t agent, const char *variable_name, void *address) {
    return gs_OrigCoreApiTable.hsa_executable_agent_global_variable_define_fn(executable, agent, variable_name, address);
}

hsa_status_t rtg_hsa_executable_readonly_variable_define(hsa_executable_t executable, hsa_agent_t agent, const char *variable_name, void *address) {
    return gs_OrigCoreApiTable.hsa_executable_readonly_variable_define_fn(executable, agent, variable_name, address);
}

hsa_status_t rtg_hsa_executable_validate(hsa_executable_t executable, uint32_t *result) {
    return gs_OrigCoreApiTable.hsa_executable_validate_fn(executable, result);
}

hsa_status_t rtg_hsa_executable_validate_alt(hsa_executable_t executable, const char *options, uint32_t *result) {
    return gs_OrigCoreApiTable.hsa_executable_validate_alt_fn(executable, options, result);
}

/* deprecated */
hsa_status_t rtg_hsa_executable_get_symbol(hsa_executable_t executable, const char *module_name, const char *symbol_name, hsa_agent_t agent, int32_t call_convention, hsa_executable_symbol_t *symbol) {
    return gs_OrigCoreApiTable.hsa_executable_get_symbol_fn(executable, module_name, symbol_name, agent, call_convention, symbol);
}

hsa_status_t rtg_hsa_executable_get_symbol_by_name(hsa_executable_t executable, const char *symbol_name, const hsa_agent_t *agent, hsa_executable_symbol_t *symbol) {
    return gs_OrigCoreApiTable.hsa_executable_get_symbol_by_name_fn(executable, symbol_name, agent, symbol);
}

hsa_status_t rtg_hsa_executable_symbol_get_info(hsa_executable_symbol_t executable_symbol, hsa_executable_symbol_info_t attribute, void *value) {
    return gs_OrigCoreApiTable.hsa_executable_symbol_get_info_fn(executable_symbol, attribute, value);
}

/* deprecated */
hsa_status_t rtg_hsa_executable_iterate_symbols(hsa_executable_t executable, hsa_status_t (*callback)(hsa_executable_t executable, hsa_executable_symbol_t symbol, void *data), void *data) {
    return gs_OrigCoreApiTable.hsa_executable_iterate_symbols_fn(executable, callback, data);
}

hsa_status_t rtg_hsa_executable_iterate_agent_symbols(hsa_executable_t executable, hsa_agent_t agent, hsa_status_t (*callback)(hsa_executable_t exec, hsa_agent_t agent, hsa_executable_symbol_t symbol, void *data), void *data) {
    return gs_OrigCoreApiTable.hsa_executable_iterate_agent_symbols_fn(executable, agent, callback, data);
}

hsa_status_t rtg_hsa_executable_iterate_program_symbols(hsa_executable_t executable, hsa_status_t (*callback)(hsa_executable_t exec, hsa_executable_symbol_t symbol, void *data), void *data) {
    return gs_OrigCoreApiTable.hsa_executable_iterate_program_symbols_fn(executable, callback, data);
}

//===--- Runtime Notifications --------------------------------------------===//

hsa_status_t rtg_hsa_status_string(hsa_status_t status, const char **status_string) {
    return gs_OrigCoreApiTable.hsa_status_string_fn(status, status_string);
}

static void InitCoreApiTable(CoreApiTable* table) {
    // Initialize function pointers for Hsa Core Runtime Api's
    table->hsa_init_fn = rtg_hsa_init;
    table->hsa_shut_down_fn = rtg_hsa_shut_down;
    table->hsa_system_get_info_fn = rtg_hsa_system_get_info;
    table->hsa_system_extension_supported_fn = rtg_hsa_system_extension_supported;
    table->hsa_system_get_extension_table_fn = rtg_hsa_system_get_extension_table;
    table->hsa_iterate_agents_fn = rtg_hsa_iterate_agents;
    table->hsa_agent_get_info_fn = rtg_hsa_agent_get_info;
    table->hsa_agent_get_exception_policies_fn = rtg_hsa_agent_get_exception_policies;
    table->hsa_agent_extension_supported_fn = rtg_hsa_agent_extension_supported;
    table->hsa_queue_create_fn = rtg_hsa_queue_create;
    table->hsa_soft_queue_create_fn = rtg_hsa_soft_queue_create;
    table->hsa_queue_destroy_fn = rtg_hsa_queue_destroy;
    table->hsa_queue_inactivate_fn = rtg_hsa_queue_inactivate;
    table->hsa_queue_load_read_index_scacquire_fn = rtg_hsa_queue_load_read_index_scacquire;
    table->hsa_queue_load_read_index_relaxed_fn = rtg_hsa_queue_load_read_index_relaxed;
    table->hsa_queue_load_write_index_scacquire_fn = rtg_hsa_queue_load_write_index_scacquire;
    table->hsa_queue_load_write_index_relaxed_fn = rtg_hsa_queue_load_write_index_relaxed;
    table->hsa_queue_store_write_index_relaxed_fn = rtg_hsa_queue_store_write_index_relaxed;
    table->hsa_queue_store_write_index_screlease_fn = rtg_hsa_queue_store_write_index_screlease;
    table->hsa_queue_cas_write_index_scacq_screl_fn = rtg_hsa_queue_cas_write_index_scacq_screl;
    table->hsa_queue_cas_write_index_scacquire_fn = rtg_hsa_queue_cas_write_index_scacquire;
    table->hsa_queue_cas_write_index_relaxed_fn = rtg_hsa_queue_cas_write_index_relaxed;
    table->hsa_queue_cas_write_index_screlease_fn = rtg_hsa_queue_cas_write_index_screlease;
    table->hsa_queue_add_write_index_scacq_screl_fn = rtg_hsa_queue_add_write_index_scacq_screl;
    table->hsa_queue_add_write_index_scacquire_fn = rtg_hsa_queue_add_write_index_scacquire;
    table->hsa_queue_add_write_index_relaxed_fn = rtg_hsa_queue_add_write_index_relaxed;
    table->hsa_queue_add_write_index_screlease_fn = rtg_hsa_queue_add_write_index_screlease;
    table->hsa_queue_store_read_index_relaxed_fn = rtg_hsa_queue_store_read_index_relaxed;
    table->hsa_queue_store_read_index_screlease_fn = rtg_hsa_queue_store_read_index_screlease;
    table->hsa_agent_iterate_regions_fn = rtg_hsa_agent_iterate_regions;
    table->hsa_region_get_info_fn = rtg_hsa_region_get_info;
    table->hsa_memory_register_fn = rtg_hsa_memory_register;
    table->hsa_memory_deregister_fn = rtg_hsa_memory_deregister;
    table->hsa_memory_allocate_fn = rtg_hsa_memory_allocate;
    table->hsa_memory_free_fn = rtg_hsa_memory_free;
    table->hsa_memory_copy_fn = rtg_hsa_memory_copy;
    table->hsa_memory_assign_agent_fn = rtg_hsa_memory_assign_agent;
    table->hsa_signal_create_fn = rtg_hsa_signal_create;
    table->hsa_signal_destroy_fn = rtg_hsa_signal_destroy;
    table->hsa_signal_load_relaxed_fn = rtg_hsa_signal_load_relaxed;
    table->hsa_signal_load_scacquire_fn = rtg_hsa_signal_load_scacquire;
    table->hsa_signal_store_relaxed_fn = rtg_hsa_signal_store_relaxed;
    table->hsa_signal_store_screlease_fn = rtg_hsa_signal_store_screlease;
    table->hsa_signal_wait_relaxed_fn = rtg_hsa_signal_wait_relaxed;
    table->hsa_signal_wait_scacquire_fn = rtg_hsa_signal_wait_scacquire;
    table->hsa_signal_and_relaxed_fn = rtg_hsa_signal_and_relaxed;
    table->hsa_signal_and_scacquire_fn = rtg_hsa_signal_and_scacquire;
    table->hsa_signal_and_screlease_fn = rtg_hsa_signal_and_screlease;
    table->hsa_signal_and_scacq_screl_fn = rtg_hsa_signal_and_scacq_screl;
    table->hsa_signal_or_relaxed_fn = rtg_hsa_signal_or_relaxed;
    table->hsa_signal_or_scacquire_fn = rtg_hsa_signal_or_scacquire;
    table->hsa_signal_or_screlease_fn = rtg_hsa_signal_or_screlease;
    table->hsa_signal_or_scacq_screl_fn = rtg_hsa_signal_or_scacq_screl;
    table->hsa_signal_xor_relaxed_fn = rtg_hsa_signal_xor_relaxed;
    table->hsa_signal_xor_scacquire_fn = rtg_hsa_signal_xor_scacquire;
    table->hsa_signal_xor_screlease_fn = rtg_hsa_signal_xor_screlease;
    table->hsa_signal_xor_scacq_screl_fn = rtg_hsa_signal_xor_scacq_screl;
    table->hsa_signal_exchange_relaxed_fn = rtg_hsa_signal_exchange_relaxed;
    table->hsa_signal_exchange_scacquire_fn = rtg_hsa_signal_exchange_scacquire;
    table->hsa_signal_exchange_screlease_fn = rtg_hsa_signal_exchange_screlease;
    table->hsa_signal_exchange_scacq_screl_fn = rtg_hsa_signal_exchange_scacq_screl;
    table->hsa_signal_add_relaxed_fn = rtg_hsa_signal_add_relaxed;
    table->hsa_signal_add_scacquire_fn = rtg_hsa_signal_add_scacquire;
    table->hsa_signal_add_screlease_fn = rtg_hsa_signal_add_screlease;
    table->hsa_signal_add_scacq_screl_fn = rtg_hsa_signal_add_scacq_screl;
    table->hsa_signal_subtract_relaxed_fn = rtg_hsa_signal_subtract_relaxed;
    table->hsa_signal_subtract_scacquire_fn = rtg_hsa_signal_subtract_scacquire;
    table->hsa_signal_subtract_screlease_fn = rtg_hsa_signal_subtract_screlease;
    table->hsa_signal_subtract_scacq_screl_fn = rtg_hsa_signal_subtract_scacq_screl;
    table->hsa_signal_cas_relaxed_fn = rtg_hsa_signal_cas_relaxed;
    table->hsa_signal_cas_scacquire_fn = rtg_hsa_signal_cas_scacquire;
    table->hsa_signal_cas_screlease_fn = rtg_hsa_signal_cas_screlease;
    table->hsa_signal_cas_scacq_screl_fn = rtg_hsa_signal_cas_scacq_screl;

    //===--- Instruction Set Architecture -----------------------------------===//

    table->hsa_isa_from_name_fn = rtg_hsa_isa_from_name;
    // Deprecated since v1.1.
    table->hsa_isa_get_info_fn = rtg_hsa_isa_get_info;
    // Deprecated since v1.1.
    table->hsa_isa_compatible_fn = rtg_hsa_isa_compatible;

    //===--- Code Objects (deprecated) --------------------------------------===//

    // Deprecated since v1.1.
    table->hsa_code_object_serialize_fn = rtg_hsa_code_object_serialize;
    // Deprecated since v1.1.
    table->hsa_code_object_deserialize_fn = rtg_hsa_code_object_deserialize;
    // Deprecated since v1.1.
    table->hsa_code_object_destroy_fn = rtg_hsa_code_object_destroy;
    // Deprecated since v1.1.
    table->hsa_code_object_get_info_fn = rtg_hsa_code_object_get_info;
    // Deprecated since v1.1.
    table->hsa_code_object_get_symbol_fn = rtg_hsa_code_object_get_symbol;
    // Deprecated since v1.1.
    table->hsa_code_symbol_get_info_fn = rtg_hsa_code_symbol_get_info;
    // Deprecated since v1.1.
    table->hsa_code_object_iterate_symbols_fn = rtg_hsa_code_object_iterate_symbols;

    //===--- Executable -----------------------------------------------------===//

    // Deprecated since v1.1.
    table->hsa_executable_create_fn = rtg_hsa_executable_create;
    table->hsa_executable_destroy_fn = rtg_hsa_executable_destroy;
    // Deprecated since v1.1.
    table->hsa_executable_load_code_object_fn = rtg_hsa_executable_load_code_object;
    table->hsa_executable_freeze_fn = rtg_hsa_executable_freeze;
    table->hsa_executable_get_info_fn = rtg_hsa_executable_get_info;
    table->hsa_executable_global_variable_define_fn = rtg_hsa_executable_global_variable_define;
    table->hsa_executable_agent_global_variable_define_fn = rtg_hsa_executable_agent_global_variable_define;
    table->hsa_executable_readonly_variable_define_fn = rtg_hsa_executable_readonly_variable_define;
    table->hsa_executable_validate_fn = rtg_hsa_executable_validate;
    // Deprecated since v1.1.
    table->hsa_executable_get_symbol_fn = rtg_hsa_executable_get_symbol;
    table->hsa_executable_symbol_get_info_fn = rtg_hsa_executable_symbol_get_info;
    // Deprecated since v1.1.
    table->hsa_executable_iterate_symbols_fn = rtg_hsa_executable_iterate_symbols;

    //===--- Runtime Notifications ------------------------------------------===//

    table->hsa_status_string_fn = rtg_hsa_status_string;

    // Start HSA v1.1 additions
    table->hsa_extension_get_name_fn = rtg_hsa_extension_get_name;
    table->hsa_system_major_extension_supported_fn = rtg_hsa_system_major_extension_supported;
    table->hsa_system_get_major_extension_table_fn = rtg_hsa_system_get_major_extension_table;
    table->hsa_agent_major_extension_supported_fn = rtg_hsa_agent_major_extension_supported;
    table->hsa_cache_get_info_fn = rtg_hsa_cache_get_info;
    table->hsa_agent_iterate_caches_fn = rtg_hsa_agent_iterate_caches;
    // Silent store optimization is present in all signal ops when no agents are sleeping.
    table->hsa_signal_silent_store_relaxed_fn = rtg_hsa_signal_store_relaxed;
    table->hsa_signal_silent_store_screlease_fn = rtg_hsa_signal_store_screlease;
    table->hsa_signal_group_create_fn = rtg_hsa_signal_group_create;
    table->hsa_signal_group_destroy_fn = rtg_hsa_signal_group_destroy;
    table->hsa_signal_group_wait_any_scacquire_fn = rtg_hsa_signal_group_wait_any_scacquire;
    table->hsa_signal_group_wait_any_relaxed_fn = rtg_hsa_signal_group_wait_any_relaxed;

    //===--- Instruction Set Architecture - HSA v1.1 additions --------------===//

    table->hsa_agent_iterate_isas_fn = rtg_hsa_agent_iterate_isas;
    table->hsa_isa_get_info_alt_fn = rtg_hsa_isa_get_info_alt;
    table->hsa_isa_get_exception_policies_fn = rtg_hsa_isa_get_exception_policies;
    table->hsa_isa_get_round_method_fn = rtg_hsa_isa_get_round_method;
    table->hsa_wavefront_get_info_fn = rtg_hsa_wavefront_get_info;
    table->hsa_isa_iterate_wavefronts_fn = rtg_hsa_isa_iterate_wavefronts;

    //===--- Code Objects (deprecated) - HSA v1.1 additions -----------------===//

    // Deprecated since v1.1.
    table->hsa_code_object_get_symbol_from_name_fn = rtg_hsa_code_object_get_symbol_from_name;

    //===--- Executable - HSA v1.1 additions --------------------------------===//

    table->hsa_code_object_reader_create_from_file_fn = rtg_hsa_code_object_reader_create_from_file;
    table->hsa_code_object_reader_create_from_memory_fn = rtg_hsa_code_object_reader_create_from_memory;
    table->hsa_code_object_reader_destroy_fn = rtg_hsa_code_object_reader_destroy;
    table->hsa_executable_create_alt_fn = rtg_hsa_executable_create_alt;
    table->hsa_executable_load_program_code_object_fn = rtg_hsa_executable_load_program_code_object;
    table->hsa_executable_load_agent_code_object_fn = rtg_hsa_executable_load_agent_code_object;
    table->hsa_executable_validate_alt_fn = rtg_hsa_executable_validate_alt;
    table->hsa_executable_get_symbol_by_name_fn = rtg_hsa_executable_get_symbol_by_name;
    table->hsa_executable_iterate_agent_symbols_fn = rtg_hsa_executable_iterate_agent_symbols;
    table->hsa_executable_iterate_program_symbols_fn = rtg_hsa_executable_iterate_program_symbols;
}

/*
 * Following set of functions are bundled as AMD Extension Apis
 */

// Pass through stub functions
hsa_status_t rtg_hsa_amd_coherency_get_type(hsa_agent_t agent, hsa_amd_coherency_type_t* type) {
    return gs_OrigExtApiTable.hsa_amd_coherency_get_type_fn(agent, type);
}

// Pass through stub functions
hsa_status_t rtg_hsa_amd_coherency_set_type(hsa_agent_t agent, hsa_amd_coherency_type_t type) {
    return gs_OrigExtApiTable.hsa_amd_coherency_set_type_fn(agent, type);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_profiling_set_profiler_enabled(hsa_queue_t* queue, int enable) {
    return gs_OrigExtApiTable.hsa_amd_profiling_set_profiler_enabled_fn(queue, enable);
}

hsa_status_t rtg_hsa_amd_profiling_async_copy_enable(bool enable) {
    return gs_OrigExtApiTable.hsa_amd_profiling_async_copy_enable_fn(enable);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_profiling_get_dispatch_time(hsa_agent_t agent, hsa_signal_t signal, hsa_amd_profiling_dispatch_time_t* time) {
    return gs_OrigExtApiTable.hsa_amd_profiling_get_dispatch_time_fn(agent, signal, time);
}

hsa_status_t rtg_hsa_amd_profiling_get_async_copy_time(hsa_signal_t hsa_signal, hsa_amd_profiling_async_copy_time_t* time) {
    return gs_OrigExtApiTable.hsa_amd_profiling_get_async_copy_time_fn(hsa_signal, time);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_profiling_convert_tick_to_system_domain(hsa_agent_t agent, uint64_t agent_tick, uint64_t* system_tick) {
    return gs_OrigExtApiTable.hsa_amd_profiling_convert_tick_to_system_domain_fn(agent, agent_tick, system_tick);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_signal_async_handler(hsa_signal_t signal, hsa_signal_condition_t cond, hsa_signal_value_t value, hsa_amd_signal_handler handler, void* arg) {
    return gs_OrigExtApiTable.hsa_amd_signal_async_handler_fn(signal, cond, value, handler, arg);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_async_function(void (*callback)(void* arg), void* arg) {
    return gs_OrigExtApiTable.hsa_amd_async_function_fn(callback, arg);
}

// Mirrors Amd Extension Apis
uint32_t rtg_hsa_amd_signal_wait_any(uint32_t signal_count, hsa_signal_t* signals, hsa_signal_condition_t* conds, hsa_signal_value_t* values, uint64_t timeout_hint, hsa_wait_state_t wait_hint, hsa_signal_value_t* satisfying_value) {
    return gs_OrigExtApiTable.hsa_amd_signal_wait_any_fn(signal_count, signals, conds, values, timeout_hint, wait_hint, satisfying_value);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_queue_cu_set_mask(const hsa_queue_t* queue, uint32_t num_cu_mask_count, const uint32_t* cu_mask) {
    return gs_OrigExtApiTable.hsa_amd_queue_cu_set_mask_fn(queue, num_cu_mask_count, cu_mask);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_memory_pool_get_info(hsa_amd_memory_pool_t memory_pool, hsa_amd_memory_pool_info_t attribute, void* value) {
    return gs_OrigExtApiTable.hsa_amd_memory_pool_get_info_fn(memory_pool, attribute, value);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_agent_iterate_memory_pools(hsa_agent_t agent, hsa_status_t (*callback)(hsa_amd_memory_pool_t memory_pool, void* data), void* data) {
    return gs_OrigExtApiTable.hsa_amd_agent_iterate_memory_pools_fn(agent, callback, data);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t memory_pool, size_t size, uint32_t flags, void** ptr) {
    return gs_OrigExtApiTable.hsa_amd_memory_pool_allocate_fn(memory_pool, size, flags, ptr);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_memory_pool_free(void* ptr) {
    return gs_OrigExtApiTable.hsa_amd_memory_pool_free_fn(ptr);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_memory_async_copy(void* dst, hsa_agent_t dst_agent, const void* src, hsa_agent_t src_agent, size_t size, uint32_t num_dep_signals, const hsa_signal_t* dep_signals, hsa_signal_t completion_signal) {
    return gs_OrigExtApiTable.hsa_amd_memory_async_copy_fn(dst, dst_agent, src, src_agent, size, num_dep_signals, dep_signals, completion_signal);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_memory_async_copy_rect(const hsa_pitched_ptr_t* dst, const hsa_dim3_t* dst_offset, const hsa_pitched_ptr_t* src, const hsa_dim3_t* src_offset, const hsa_dim3_t* range, hsa_agent_t copy_agent, hsa_amd_copy_direction_t dir, uint32_t num_dep_signals, const hsa_signal_t* dep_signals, hsa_signal_t completion_signal) {
    return gs_OrigExtApiTable.hsa_amd_memory_async_copy_rect_fn(dst, dst_offset, src, src_offset, range, copy_agent, dir, num_dep_signals, dep_signals, completion_signal);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_agent_memory_pool_get_info(hsa_agent_t agent, hsa_amd_memory_pool_t memory_pool, hsa_amd_agent_memory_pool_info_t attribute, void* value) {
    return gs_OrigExtApiTable.hsa_amd_agent_memory_pool_get_info_fn(agent, memory_pool, attribute, value);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_agents_allow_access(uint32_t num_agents, const hsa_agent_t* agents, const uint32_t* flags, const void* ptr) {
    return gs_OrigExtApiTable.hsa_amd_agents_allow_access_fn(num_agents, agents, flags, ptr);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_memory_pool_can_migrate(hsa_amd_memory_pool_t src_memory_pool, hsa_amd_memory_pool_t dst_memory_pool, bool* result) {
    return gs_OrigExtApiTable.hsa_amd_memory_pool_can_migrate_fn(src_memory_pool, dst_memory_pool, result);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_memory_migrate(const void* ptr, hsa_amd_memory_pool_t memory_pool, uint32_t flags) {
    return gs_OrigExtApiTable.hsa_amd_memory_migrate_fn(ptr, memory_pool, flags);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_memory_lock(void* host_ptr, size_t size, hsa_agent_t* agents, int num_agent, void** agent_ptr) {
    return gs_OrigExtApiTable.hsa_amd_memory_lock_fn(host_ptr, size, agents, num_agent, agent_ptr);
}

#if ENABLE_HSA_AMD_MEMORY_LOCK_TO_POOL
// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_memory_lock_to_pool(void* host_ptr, size_t size, hsa_agent_t* agents, int num_agent, hsa_amd_memory_pool_t pool, uint32_t flags, void** agent_ptr) {
    return gs_OrigExtApiTable.hsa_amd_memory_lock_to_pool_fn(host_ptr, size, agents, num_agent, pool, flags, agent_ptr);
}
#endif

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_memory_unlock(void* host_ptr) {
    return gs_OrigExtApiTable.hsa_amd_memory_unlock_fn(host_ptr);

}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_memory_fill(void* ptr, uint32_t value, size_t count) {
    return gs_OrigExtApiTable.hsa_amd_memory_fill_fn(ptr, value, count);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_interop_map_buffer(uint32_t num_agents, hsa_agent_t* agents, int interop_handle, uint32_t flags, size_t* size, void** ptr, size_t* metadata_size, const void** metadata) {
    return gs_OrigExtApiTable.hsa_amd_interop_map_buffer_fn(num_agents, agents, interop_handle, flags, size, ptr, metadata_size, metadata);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_interop_unmap_buffer(void* ptr) {
    return gs_OrigExtApiTable.hsa_amd_interop_unmap_buffer_fn(ptr);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_image_create(hsa_agent_t agent, const hsa_ext_image_descriptor_t *image_descriptor, const hsa_amd_image_descriptor_t *image_layout, const void *image_data, hsa_access_permission_t access_permission, hsa_ext_image_t *image) {
    return gs_OrigExtApiTable.hsa_amd_image_create_fn(agent, image_descriptor, image_layout, image_data, access_permission, image);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_pointer_info(void* ptr, hsa_amd_pointer_info_t* info, void* (*alloc)(size_t), uint32_t* num_agents_accessible, hsa_agent_t** accessible) {
    return gs_OrigExtApiTable.hsa_amd_pointer_info_fn(ptr, info, alloc, num_agents_accessible, accessible);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_pointer_info_set_userdata(void* ptr, void* userptr) {
    return gs_OrigExtApiTable.hsa_amd_pointer_info_set_userdata_fn(ptr, userptr);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_ipc_memory_create(void* ptr, size_t len, hsa_amd_ipc_memory_t* handle) {
    return gs_OrigExtApiTable.hsa_amd_ipc_memory_create_fn(ptr, len, handle);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_ipc_memory_attach(const hsa_amd_ipc_memory_t* ipc, size_t len, uint32_t num_agents, const hsa_agent_t* mapping_agents, void** mapped_ptr) {
    return gs_OrigExtApiTable.hsa_amd_ipc_memory_attach_fn(ipc, len, num_agents, mapping_agents, mapped_ptr);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_ipc_memory_detach(void* mapped_ptr) {
    return gs_OrigExtApiTable.hsa_amd_ipc_memory_detach_fn(mapped_ptr);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_signal_create(hsa_signal_value_t initial_value, uint32_t num_consumers, const hsa_agent_t* consumers, uint64_t attributes, hsa_signal_t* signal) {
    return gs_OrigExtApiTable.hsa_amd_signal_create_fn(initial_value, num_consumers, consumers, attributes, signal);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_ipc_signal_create(hsa_signal_t signal, hsa_amd_ipc_signal_t* handle) {
    return gs_OrigExtApiTable.hsa_amd_ipc_signal_create_fn(signal, handle);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_ipc_signal_attach(const hsa_amd_ipc_signal_t* handle, hsa_signal_t* signal) {
    return gs_OrigExtApiTable.hsa_amd_ipc_signal_attach_fn(handle, signal);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_register_system_event_handler(hsa_amd_system_event_callback_t callback, void* data) {
    return gs_OrigExtApiTable.hsa_amd_register_system_event_handler_fn(callback, data);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_queue_intercept_create(hsa_agent_t agent_handle, uint32_t size, hsa_queue_type32_t type, void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data), void* data, uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t** queue) {
    return gs_OrigExtApiTable.hsa_amd_queue_intercept_create_fn(agent_handle, size, type, callback, data, private_segment_size, group_segment_size, queue);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_queue_intercept_register(hsa_queue_t* queue, hsa_amd_queue_intercept_handler callback, void* user_data) {
    return gs_OrigExtApiTable.hsa_amd_queue_intercept_register_fn(queue, callback, user_data);
}

// Mirrors Amd Extension Apis
hsa_status_t rtg_hsa_amd_queue_set_priority(hsa_queue_t* queue, hsa_amd_queue_priority_t priority) {
    return gs_OrigExtApiTable.hsa_amd_queue_set_priority_fn(queue, priority);
}
static void InitAmdExtTable(AmdExtTable* table) {
    // Initialize function pointers for Amd Extension Api's
    table->hsa_amd_coherency_get_type_fn = rtg_hsa_amd_coherency_get_type;
    table->hsa_amd_coherency_set_type_fn = rtg_hsa_amd_coherency_set_type;
    table->hsa_amd_profiling_set_profiler_enabled_fn = rtg_hsa_amd_profiling_set_profiler_enabled;
    table->hsa_amd_profiling_async_copy_enable_fn = rtg_hsa_amd_profiling_async_copy_enable;
    table->hsa_amd_profiling_get_dispatch_time_fn = rtg_hsa_amd_profiling_get_dispatch_time;
    table->hsa_amd_profiling_get_async_copy_time_fn = rtg_hsa_amd_profiling_get_async_copy_time;
    table->hsa_amd_profiling_convert_tick_to_system_domain_fn = rtg_hsa_amd_profiling_convert_tick_to_system_domain;
    table->hsa_amd_signal_async_handler_fn = rtg_hsa_amd_signal_async_handler;
    table->hsa_amd_async_function_fn = rtg_hsa_amd_async_function;
    table->hsa_amd_signal_wait_any_fn = rtg_hsa_amd_signal_wait_any;
    table->hsa_amd_queue_cu_set_mask_fn = rtg_hsa_amd_queue_cu_set_mask;
    table->hsa_amd_memory_pool_get_info_fn = rtg_hsa_amd_memory_pool_get_info;
    table->hsa_amd_agent_iterate_memory_pools_fn = rtg_hsa_amd_agent_iterate_memory_pools;
    table->hsa_amd_memory_pool_allocate_fn = rtg_hsa_amd_memory_pool_allocate;
    table->hsa_amd_memory_pool_free_fn = rtg_hsa_amd_memory_pool_free;
    table->hsa_amd_memory_async_copy_fn = rtg_hsa_amd_memory_async_copy;
    table->hsa_amd_agent_memory_pool_get_info_fn = rtg_hsa_amd_agent_memory_pool_get_info;
    table->hsa_amd_agents_allow_access_fn = rtg_hsa_amd_agents_allow_access;
    table->hsa_amd_memory_pool_can_migrate_fn = rtg_hsa_amd_memory_pool_can_migrate;
    table->hsa_amd_memory_migrate_fn = rtg_hsa_amd_memory_migrate;
    table->hsa_amd_memory_lock_fn = rtg_hsa_amd_memory_lock;
    table->hsa_amd_memory_unlock_fn = rtg_hsa_amd_memory_unlock;
    table->hsa_amd_memory_fill_fn = rtg_hsa_amd_memory_fill;
    table->hsa_amd_interop_map_buffer_fn = rtg_hsa_amd_interop_map_buffer;
    table->hsa_amd_interop_unmap_buffer_fn = rtg_hsa_amd_interop_unmap_buffer;
    table->hsa_amd_pointer_info_fn = rtg_hsa_amd_pointer_info;
    table->hsa_amd_pointer_info_set_userdata_fn = rtg_hsa_amd_pointer_info_set_userdata;
    table->hsa_amd_ipc_memory_create_fn = rtg_hsa_amd_ipc_memory_create;
    table->hsa_amd_ipc_memory_attach_fn = rtg_hsa_amd_ipc_memory_attach;
    table->hsa_amd_ipc_memory_detach_fn = rtg_hsa_amd_ipc_memory_detach;
    table->hsa_amd_signal_create_fn = rtg_hsa_amd_signal_create;
    table->hsa_amd_ipc_signal_create_fn = rtg_hsa_amd_ipc_signal_create;
    table->hsa_amd_ipc_signal_attach_fn = rtg_hsa_amd_ipc_signal_attach;
    table->hsa_amd_register_system_event_handler_fn = rtg_hsa_amd_register_system_event_handler;
    table->hsa_amd_queue_intercept_create_fn = rtg_hsa_amd_queue_intercept_create;
    table->hsa_amd_queue_intercept_register_fn = rtg_hsa_amd_queue_intercept_register;
    table->hsa_amd_queue_set_priority_fn = rtg_hsa_amd_queue_set_priority;
    table->hsa_amd_memory_async_copy_rect_fn = rtg_hsa_amd_memory_async_copy_rect;
#if ENABLE_HSA_AMD_RUNTIME_QUEUE_CREATE_REGISTER
    table->hsa_amd_runtime_queue_create_register_fn = rtg_hsa_amd_runtime_queue_create_register;
#endif
#if ENABLE_HSA_AMD_MEMORY_LOCK_TO_POOL
    table->hsa_amd_memory_lock_to_pool_fn = rtg_hsa_amd_memory_lock_to_pool;
#endif
}

