#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
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

static std::string LABEL = "HIP GUARD: ";
#define ERR(msg) std::cerr << LABEL << __func__ << ": " << msg << std::endl
#define LOG(msg) std::cerr << LABEL << __func__ << ": " << msg << std::endl
#define TRACE(msg) std::cerr << LABEL << __func__ << ": " << msg << std::endl

#define ARGS_HIPMALLOC  const void* function_address, dim3 numBlocks, dim3 dimBlocks, void** args, size_t sharedMemBytes, hipStream_t stream
#define ARGS_HIPMALLOC_             function_address,      numBlocks,      dimBlocks,        args,        sharedMemBytes,             stream

static void save_ptr(void* ptr, size_t size)
{
    TRACE("ptr=" << ptr << " size=" << size);
    {
        std::lock_guard<std::mutex> lock(gs_allocations_mutex_);
        if (gs_allocations.find(ptr) != gs_allocations.end()) {
            ERR("collision");
        }
        gs_allocations[ptr] = size;
    }

    int *guard = (int*)((char*)(ptr) + size);
    auto status = hipMemsetD32(guard, GUARD_INT, GUARD_INT_COUNT);
    if (hipSuccess != status) {
        ERR("memset guard failed: " << hipGetErrorString(status));
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
        orig = (fptr)dlsym(RTLD_NEXT, "hipMalloc");
        if (orig == NULL) {
            ERR("dlsym: " << dlerror());
            return hipErrorUnknown;
        }
    }

    auto new_size = size + GUARD_SIZE;
    auto status = orig(ptr, new_size);
    if (hipSuccess != status) {
        ERR("hipMalloc: " << hipGetErrorString(status));
    }
    save_ptr(ptr, size);
    return status;
}

__global__ void guard_check(int *guards[blocksPerGrid], int num_guards, int *answer)
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
            *answer = r[0];
        }
    }
}

void check_answer(hipStream_t stream, hipError_t status, void* userData)
{
    TRACE("stream=" << stream << " status=" << status << " userData=" << userData);
    int *answer = reinterpret_cast<int*>(userData);
    if (*answer != 0) {
        ERR("out of bounds found");
    }
}

void free_answer(hipStream_t stream, hipError_t status, void* userData)
{
    TRACE("stream=" << stream << " status=" << status << " userData=" << userData);
    auto th = std::thread(release_stream_guard_ptr, userData);
    th.detach();
}

hipError_t hipLaunchKernel(ARGS_HIPMALLOC)
{
    TRACE("");
    typedef hipError_t (*fptr)(ARGS_HIPMALLOC);
    static fptr orig = NULL;

    if (orig == NULL) {
        orig = (fptr)dlsym(RTLD_NEXT, "hipLaunchKernel");
        if (orig == NULL) {
            ERR("dlsym: " << dlerror());
            return hipErrorUnknown;
        }
    }

    auto ret = orig(ARGS_HIPMALLOC_);

    if (this_launch_is_our_guard_launch) return ret;

    // guard check
    // check up to 256 guard pages per kernel launch
    // each block is assigned 1 page
    hipError_t status = hipSuccess;
    int *answer = get_stream_guard_ptr(stream);
    int *guards[blocksPerGrid];
    for (int i=0; i<blocksPerGrid; i++) {
        guards[i] = NULL;
    }
    {
        std::lock_guard<std::mutex> lock(gs_allocations_mutex_);
        int i = 0;
        for (auto& it: gs_allocations) {
            auto ptr = it.first;
            auto size = it.second;
            int *guard = (int*)((char*)ptr + size);
            guards[i++] = guard;
            LOG("guard_check prep for ptr=" << ptr << " size=" << size);
            if (i == blocksPerGrid) {
                LOG("guard_check");
                this_launch_is_our_guard_launch = true;
                //guard_check<<<dim3(blocksPerGrid), dim3(threadsPerBlock), 0, stream>>>(guards, blocksPerGrid, answer);
                hipLaunchKernelGGL(guard_check, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, stream, guards, blocksPerGrid, answer);
                status = hipStreamAddCallback(stream, check_answer, answer, 0);
                if (hipSuccess != status) {
                    ERR("hipStreamAddCallback failed: " << hipGetErrorString(status));
                }
                i = 0;
            }
        }
        if (i > 0) {
            LOG("guard_check");
            this_launch_is_our_guard_launch = true;
            //guard_check<<<dim3(blocksPerGrid), dim3(threadsPerBlock), 0, stream>>>(guards, i, answer);
            hipLaunchKernelGGL(guard_check, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, stream, guards, blocksPerGrid, answer);
            if (hipSuccess != status) {
                ERR("hipStreamAddCallback failed: " << hipGetErrorString(status));
            }
        }
        else {
            LOG("WTF");
        }
        if (hipSuccess != status) {
            ERR("hipStreamAddCallback failed: " << hipGetErrorString(status));
        }
    }

    return ret;
}
