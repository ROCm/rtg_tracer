/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/

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

template <>
inline std::string ToString(hsa_agent_t agent) {
    return "agent_" + ToString(agent.handle);
}

template <>
inline std::string ToString(hsa_cache_t cache) {
    return "cache_" + ToString(cache.handle);
}

template <>
inline std::string ToString(hsa_cache_info_t attribute) {
    return "_omitted_";
}

template <>
inline std::string ToString(hsa_region_t region) {
    return "region_" + ToString(region.handle);
}

template <>
inline std::string ToString(hsa_signal_t signal) {
    return "signal_" + ToString(signal.handle);
}

template <>
inline std::string ToString(hsa_signal_group_t group) {
    return "signal_group_" + ToString(group.handle);
}

template <>
inline std::string ToString(hsa_wavefront_t wavefront) {
    return "wavefront_" + ToString(wavefront.handle);
}

template <>
inline std::string ToString(hsa_isa_t isa) {
    return "isa_" + ToString(isa.handle);
}

template <>
inline std::string ToString(hsa_callback_data_t data) {
    return "callback_data_" + ToString(data.handle);
}

template <>
inline std::string ToString(hsa_code_object_t obj) {
    return "code_object_" + ToString(obj.handle);
}

template <>
inline std::string ToString(hsa_code_symbol_t symbol) {
    return "code_symbol_" + ToString(symbol.handle);
}

template <>
inline std::string ToString(hsa_code_object_reader_t reader) {
    return "code_object_reader_" + ToString(reader.handle);
}

template <>
inline std::string ToString(hsa_executable_t executable) {
    return "executable_" + ToString(executable.handle);
}

template <>
inline std::string ToString(hsa_executable_symbol_t symbol) {
    return "executable_symbol_" + ToString(symbol.handle);
}

template <>
inline std::string ToString(hsa_amd_memory_pool_t pool) {
    return "amd_memory_pool_" + ToString(pool.handle);
}

// Catch empty arguments case
inline std::string ToString() { return (""); }

// C++11 variadic template - peels off first argument, converts to string, and calls itself again to
// peel the next arg. Strings are automatically separated by comma+space.
template <typename T, typename... Args>
inline std::string ToString(T first, Args... args) {
    return ToString(first) + ", " + ToString(args...);
}

