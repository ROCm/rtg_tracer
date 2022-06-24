#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cxxabi.h>
#include <dlfcn.h>

#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include <hip/hip_runtime.h>
//#include <hip/hip_runtime_api.h>

#define GUARD_SIZE (2 * 1024 * 1024)
#define GUARD_INT 0xdeadbeef
#define GUARD_INT_COUNT (GUARD_SIZE / sizeof(int))
static std::mutex gs_allocations_mutex_; // protects gs_allocations
static std::unordered_map<void*, size_t> gs_allocations;
static std::unordered_map<hipStream_t, int*> gs_stream_guard_ptr;
constexpr int blocksPerGrid = 256;
constexpr int threadsPerBlock = 256;

static thread_local bool this_launch_is_our_guard_launch(false);

#define OUT std::cerr
//#define OUT std::cout
static std::string LABEL = "HIP GUARD: ";
#define ERR(msg) OUT << LABEL << __func__ << ": " << msg << std::endl
#define RPT(msg) OUT << LABEL << msg << std::endl
#if DEBUG == 1
#define LOG(msg) OUT << LABEL << __func__ << ": " << msg << std::endl
#define TRACE(msg) OUT << LABEL << __func__ << ": " << msg << std::endl
#else
#define LOG(msg)
#define TRACE(msg)
#endif

template <typename T, int size>
struct Array {
    T data[size];

    __host__ __device__ T operator[](int i) const {
        return data[i];
    }
    __host__ __device__ T& operator[](int i) {
        return data[i];
    }
    Array() = default;
    Array(const Array&) = default;
    Array& operator=(const Array&) = default;
    __host__ __device__ Array(T x) {
        for (int i = 0; i < size; i++) {
            data[i] = x;
        }
    }
};

// C++ symbol demangle
static inline const char* cxx_demangle(const char* symbol) {
  size_t funcnamesize;
  int status;
  const char* ret = (symbol != NULL) ? abi::__cxa_demangle(symbol, NULL, &funcnamesize, &status) : symbol;
  return (ret != NULL) ? ret : symbol;
}

static void store_ptr(void* ptr, size_t size)
{
    TRACE("ptr=" << ptr << " size=" << size);
    {
        std::lock_guard<std::mutex> lock(gs_allocations_mutex_);
        if (gs_allocations.find(ptr) != gs_allocations.end()) {
            ERR("collision");
        }
        gs_allocations[ptr] = size;
    }

    int *guard = (int*)(((char*)(ptr)) + size);
    LOG("guard=" << guard);
    auto status = hipMemsetD32(guard, GUARD_INT, GUARD_INT_COUNT);
    if (hipSuccess != status) {
        ERR("memset guard failed: " << hipGetErrorString(status));
    }
}

static void release_ptr(void* ptr)
{
    TRACE("ptr=" << ptr);
    {
        std::lock_guard<std::mutex> lock(gs_allocations_mutex_);
        if (gs_allocations.find(ptr) == gs_allocations.end()) {
            ERR("not found");
        }
        else {
            gs_allocations.erase(ptr);
        }
    }
}

static int* get_stream_guard_ptr(hipStream_t stream)
{
    TRACE("stream=" << stream);
#if 0
    std::lock_guard<std::mutex> lock(gs_allocations_mutex_);
    if (gs_stream_guard_ptr.find(stream) == gs_stream_guard_ptr.end()) {
        int *ptr = NULL;
        if (hipSuccess != hipHostMalloc(&ptr, sizeof(int), 0)) {
            ERR("hipHostMalloc failed");
        }
        gs_stream_guard_ptr[stream] = ptr;
    }
    return gs_stream_guard_ptr[stream];
#else
    int *ptr = NULL;
    auto status = hipHostMalloc(&ptr, sizeof(int), 0);
    if (hipSuccess != status) {
        ERR("hipHostMalloc failed: " << hipGetErrorString(status));
    }
    *ptr = 0;
    return ptr;
#endif
}

static void release_stream_guard_ptr(void *ptr) {
    TRACE("ptr=" << ptr);
    auto status = hipHostFree(ptr);
    if (hipSuccess != status) {
        ERR("hipHostFree failed: " << hipGetErrorString(status));
    }
}

hipError_t hipMalloc(void** ptr, size_t size)
{
    TRACE("**ptr=" << ptr << " size=" << size);
    typedef hipError_t (*fptr)(void** ptr, size_t size);
    static fptr orig = NULL;

    if (orig == NULL) {
        orig = (fptr)dlsym(RTLD_NEXT, __func__);
        if (orig == NULL) {
            ERR("dlsym: " << dlerror());
            return hipErrorUnknown;
        }
    }

    auto new_size = size + GUARD_SIZE;
    auto status = orig(ptr, new_size);
    if (hipSuccess != status) {
        ERR(hipGetErrorString(status));
    }
    LOG("*ptr=" << *ptr);

    store_ptr(*ptr, size);
    return status;
}

hipError_t hipExtMallocWithFlags(void** ptr, size_t size, unsigned int flags)
{
    TRACE("**ptr=" << ptr << " size=" << size << " flags=" << flags);
    typedef hipError_t (*fptr)(void** ptr, size_t size, unsigned int flags);
    static fptr orig = NULL;

    if (orig == NULL) {
        orig = (fptr)dlsym(RTLD_NEXT, __func__);
        if (orig == NULL) {
            ERR("dlsym: " << dlerror());
            return hipErrorUnknown;
        }
    }

    auto new_size = size + GUARD_SIZE;
    auto status = orig(ptr, new_size, flags);
    if (hipSuccess != status) {
        ERR(hipGetErrorString(status));
    }
    LOG("*ptr=" << *ptr);

    store_ptr(*ptr, size);
    return status;
}

hipError_t hipMallocManaged(void** ptr, size_t size, unsigned int flags)
{
    TRACE("**ptr=" << ptr << " size=" << size << " flags=" << flags);
    typedef hipError_t (*fptr)(void** ptr, size_t size, unsigned int flags);
    static fptr orig = NULL;

    if (orig == NULL) {
        orig = (fptr)dlsym(RTLD_NEXT, __func__);
        if (orig == NULL) {
            ERR("dlsym: " << dlerror());
            return hipErrorUnknown;
        }
    }

    auto new_size = size + GUARD_SIZE;
    auto status = orig(ptr, new_size, flags);
    if (hipSuccess != status) {
        ERR(hipGetErrorString(status));
    }
    LOG("*ptr=" << *ptr);

    store_ptr(*ptr, size);
    return status;
}

hipError_t hipFree(void* ptr)
{
    TRACE("ptr=" << ptr);
    typedef hipError_t (*fptr)(void* ptr);
    static fptr orig = NULL;

    if (orig == NULL) {
        orig = (fptr)dlsym(RTLD_NEXT, __func__);
        if (orig == NULL) {
            ERR("dlsym: " << dlerror());
            return hipErrorUnknown;
        }
    }

    release_ptr(ptr);

    auto status = orig(ptr);
    if (hipSuccess != status) {
        ERR(hipGetErrorString(status));
    }

    return status;
}

// guard check
// check up to 256 guard pages per kernel launch
// each block is assigned 1 page, all threads check their own indices, then block reduce to get answer
// answer is pinned host memory
// guard page is reset during check
__global__ void guard_check(Array<int*,blocksPerGrid> guards, int num_guards, int *answer)
{
    int bdx = blockIdx.x;
    int idx = threadIdx.x;
    __shared__ int r[threadsPerBlock];

    if (bdx < num_guards)
    {
        int *guard = guards[bdx];
        r[idx] = 0;
        for (int i = 0; i < GUARD_INT_COUNT; i += threadsPerBlock) {
            if (i < GUARD_INT_COUNT) {
                r[idx] += (guard[i] != GUARD_INT);
                guard[i] = GUARD_INT; // reset guard value
            }
        }
        __syncthreads();
        for (int size = threadsPerBlock / 2; size > 0; size /= 2) {
            if (idx < size) {
                r[idx] += r[idx + size];
            }
            __syncthreads();
        }
        if (idx == 0) {
            *answer += r[0];
        }
    }
}

struct CheckAnswerData {
    int *answer;
    std::string kernel_name;
};

void check_answer(hipStream_t stream, hipError_t status, void* userData)
{
    TRACE("stream=" << stream << " status=" << status << " userData=" << userData);
    CheckAnswerData *data = reinterpret_cast<CheckAnswerData*>(userData);
    if (*(data->answer) != 0) {
        RPT("out of bounds found: " << data->kernel_name);
    }
    delete data;
}

void free_answer(hipStream_t stream, hipError_t status, void* userData)
{
    // must be in separate thread or hangs the runtime
    TRACE("stream=" << stream << " status=" << status << " userData=" << userData);
    auto th = std::thread(release_stream_guard_ptr, userData);
    th.detach();
}

void launch_guard_check(std::string kernel_name, hipStream_t stream)
{
    // protect against recursive calls
    this_launch_is_our_guard_launch = true;

    hipError_t status = hipSuccess;
    int *answer = get_stream_guard_ptr(stream);
    Array<int*,blocksPerGrid> guards;
    for (int i=0; i<blocksPerGrid; i++) {
        guards[i] = NULL;
    }
    {
        std::lock_guard<std::mutex> lock(gs_allocations_mutex_);
        int i = 0;
        for (auto& it: gs_allocations) {
            void *ptr = it.first;
            size_t size = it.second;
            int *guard = (int*)((char*)ptr + size);
            guards[i++] = guard;
            LOG("guard_check prep for i=" << i << " ptr=" << ptr << " guard=" << guard << " size=" << size);
            if (i == blocksPerGrid) {
                LOG("guard_check full");
                guard_check<<<dim3(blocksPerGrid), dim3(threadsPerBlock), 0, stream>>>(guards, blocksPerGrid, answer);
                status = hipStreamAddCallback(stream, check_answer, new CheckAnswerData{answer,kernel_name}, 0);
                if (hipSuccess != status) {
                    ERR("hipStreamAddCallback failed: " << hipGetErrorString(status));
                }
                i = 0;
            }
        }
        if (i > 0) {
            LOG("guard_check partial i=" << i);
            guard_check<<<dim3(i), dim3(threadsPerBlock), 0, stream>>>(guards, i, answer);
            status = hipStreamAddCallback(stream, check_answer, new CheckAnswerData{answer,kernel_name}, 0);
            if (hipSuccess != status) {
                ERR("hipStreamAddCallback failed: " << hipGetErrorString(status));
            }
        }
        status = hipStreamAddCallback(stream, free_answer, answer, 0);
        if (hipSuccess != status) {
            ERR("hipStreamAddCallback failed: " << hipGetErrorString(status));
        }
    }
}

#define ARGS_HIPLAUNCHKERNEL  const void* function_address, dim3 numBlocks, dim3 dimBlocks, void** args, size_t sharedMemBytes, hipStream_t stream
#define ARGS_HIPLAUNCHKERNEL_             function_address,      numBlocks,      dimBlocks,        args,        sharedMemBytes,             stream
hipError_t hipLaunchKernel(ARGS_HIPLAUNCHKERNEL)
{
    TRACE("");
    typedef hipError_t (*fptr)(ARGS_HIPLAUNCHKERNEL);
    static fptr orig = NULL;

    if (orig == NULL) {
        orig = (fptr)dlsym(RTLD_NEXT, __func__);
        if (orig == NULL) {
            ERR("dlsym: " << dlerror());
            return hipErrorUnknown;
        }
    }

    auto ret = orig(ARGS_HIPLAUNCHKERNEL_);

    if (this_launch_is_our_guard_launch) return ret;

    std::string kernel_name = cxx_demangle(hipKernelNameRefByPtr(function_address, stream));

    launch_guard_check(kernel_name, stream);

    return ret;
}


//hipLaunchCooperativeKernelMultiDevice TODO
//hipExtLaunchMultiKernelMultiDevice TODO

#define ARGS_HIPEXTLAUNCHKERNEL  const void* function_address, dim3 numBlocks, dim3 dimBlocks, void** args, size_t sharedMemBytes, hipStream_t stream, hipEvent_t startEvent, hipEvent_t stopEvent, int flags
#define ARGS_HIPEXTLAUNCHKERNEL_             function_address,      numBlocks,      dimBlocks,        args,        sharedMemBytes,             stream,            startEvent,            stopEvent,     flags
extern "C" hipError_t hipExtLaunchKernel(ARGS_HIPEXTLAUNCHKERNEL)
{
    TRACE("");
    typedef hipError_t (*fptr)(ARGS_HIPEXTLAUNCHKERNEL);
    static fptr orig = NULL;

    if (orig == NULL) {
        orig = (fptr)dlsym(RTLD_NEXT, __func__);
        if (orig == NULL) {
            ERR("dlsym: " << dlerror());
            return hipErrorUnknown;
        }
    }

    auto ret = orig(ARGS_HIPEXTLAUNCHKERNEL_);

    if (this_launch_is_our_guard_launch) return ret;

    std::string kernel_name = cxx_demangle(hipKernelNameRefByPtr(function_address, stream));

    launch_guard_check(kernel_name, stream);

    return ret;
}


#define ARGS_HIPLAUNCHCOOPERATIVEKERNEL  const void* f, dim3 gridDim, dim3 blockDimX, void** kernelParams, unsigned int sharedMemBytes, hipStream_t stream
#define ARGS_HIPLAUNCHCOOPERATIVEKERNEL_             f,      gridDim,      blockDimX,        kernelParams,              sharedMemBytes,             stream
hipError_t hipLaunchCooperativeKernel(ARGS_HIPLAUNCHCOOPERATIVEKERNEL)
{
    TRACE("");
    typedef hipError_t (*fptr)(ARGS_HIPLAUNCHCOOPERATIVEKERNEL);
    static fptr orig = NULL;

    if (orig == NULL) {
        orig = (fptr)dlsym(RTLD_NEXT, __func__);
        if (orig == NULL) {
            ERR("dlsym: " << dlerror());
            return hipErrorUnknown;
        }
    }

    auto ret = orig(ARGS_HIPLAUNCHCOOPERATIVEKERNEL_);

    if (this_launch_is_our_guard_launch) return ret;

    std::string kernel_name = cxx_demangle(hipKernelNameRefByPtr(f, stream));

    launch_guard_check(kernel_name, stream);

    return ret;
}


#define ARGS_HIPHCCMODULELAUNCHKERNEL  hipFunction_t f, uint32_t globalWorkSizeX, uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ, uint32_t localWorkSizeX, uint32_t localWorkSizeY, uint32_t localWorkSizeZ, size_t sharedMemBytes, hipStream_t hStream, void** kernelParams, void** extra, hipEvent_t startEvent, hipEvent_t stopEvent
#define ARGS_HIPHCCMODULELAUNCHKERNEL_               f,          globalWorkSizeX,          globalWorkSizeY,          globalWorkSizeZ,          localWorkSizeX,          localWorkSizeY,          localWorkSizeZ,        sharedMemBytes,             hStream,        kernelParams,        extra,            startEvent,            stopEvent
hipError_t hipHccModuleLaunchKernel(ARGS_HIPHCCMODULELAUNCHKERNEL)
{
    TRACE("");
    typedef hipError_t (*fptr)(ARGS_HIPHCCMODULELAUNCHKERNEL);
    static fptr orig = NULL;

    if (orig == NULL) {
        orig = (fptr)dlsym(RTLD_NEXT, __func__);
        if (orig == NULL) {
            ERR("dlsym: " << dlerror());
            return hipErrorUnknown;
        }
    }

    auto ret = orig(ARGS_HIPHCCMODULELAUNCHKERNEL_);

    if (this_launch_is_our_guard_launch) return ret;

    std::string kernel_name = cxx_demangle(hipKernelNameRef(f));

    launch_guard_check(kernel_name, hStream);

    return ret;
}


#define ARGS_HIPMODULELAUNCHKERNEL  hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream, void** kernelParams, void** extra
#define ARGS_HIPMODULELAUNCHKERNEL_               f,              gridDimX,              gridDimY,              gridDimZ,              blockDimX,              blockDimY,              blockDimZ,              sharedMemBytes,             stream,        kernelParams,        extra
hipError_t hipModuleLaunchKernel(ARGS_HIPMODULELAUNCHKERNEL)
{
    TRACE("");
    typedef hipError_t (*fptr)(ARGS_HIPMODULELAUNCHKERNEL);
    static fptr orig = NULL;

    if (orig == NULL) {
        orig = (fptr)dlsym(RTLD_NEXT, __func__);
        if (orig == NULL) {
            ERR("dlsym: " << dlerror());
            return hipErrorUnknown;
        }
    }

    auto ret = orig(ARGS_HIPMODULELAUNCHKERNEL_);

    if (this_launch_is_our_guard_launch) return ret;

    std::string kernel_name = cxx_demangle(hipKernelNameRef(f));

    launch_guard_check(kernel_name, stream);

    return ret;
}


#define ARGS_HIPEXTMODULELAUNCHKERNEL  hipFunction_t f, uint32_t globalWorkSizeX, uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ, uint32_t localWorkSizeX, uint32_t localWorkSizeY, uint32_t localWorkSizeZ, size_t sharedMemBytes, hipStream_t hStream, void** kernelParams, void** extra, hipEvent_t startEvent, hipEvent_t stopEvent, uint32_t flags
#define ARGS_HIPEXTMODULELAUNCHKERNEL_               f,          globalWorkSizeX,          globalWorkSizeY,          globalWorkSizeZ,          localWorkSizeX,          localWorkSizeY,          localWorkSizeZ,        sharedMemBytes,             hStream,        kernelParams,        extra,            startEvent,            stopEvent,          flags
hipError_t hipExtModuleLaunchKernel(ARGS_HIPEXTMODULELAUNCHKERNEL)
{
    TRACE("");
    typedef hipError_t (*fptr)(ARGS_HIPEXTMODULELAUNCHKERNEL);
    static fptr orig = NULL;

    if (orig == NULL) {
        //orig = (fptr)dlsym(RTLD_NEXT, __func__);
        orig = (fptr)dlsym(RTLD_NEXT, "_Z24hipExtModuleLaunchKernelP18ihipModuleSymbol_tjjjjjjmP12ihipStream_tPPvS4_P11ihipEvent_tS6_j");
        if (orig == NULL) {
            ERR("dlsym: " << dlerror());
            return hipErrorUnknown;
        }
    }

    auto ret = orig(ARGS_HIPEXTMODULELAUNCHKERNEL_);

    if (this_launch_is_our_guard_launch) return ret;

    std::string kernel_name = cxx_demangle(hipKernelNameRef(f));

    launch_guard_check(kernel_name, hStream);

    return ret;
}

// hipMemcpy et al TODO
