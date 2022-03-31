/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#include <iostream>
#include <sstream>

#include <hip/hip_runtime_api.h>

#define DUMMY_DECL(T)  \
std::ostream& operator<<(std::ostream& out, const T obj);

#define DUMMY_IMPL(T)  \
std::ostream& operator<<(std::ostream& out, const T obj) { \
    return out << #T; \
}

DUMMY_DECL(HIP_MEMCPY3D)
DUMMY_DECL(hip_Memcpy2D)
DUMMY_DECL(hipPitchedPtr)
DUMMY_DECL(hipExtent)
DUMMY_DECL(hipArray)
DUMMY_DECL(hipLaunchParams)
DUMMY_DECL(hipChannelFormatDesc)
DUMMY_DECL(HIP_ARRAY_DESCRIPTOR)
DUMMY_DECL(hipDeviceProp_t)
DUMMY_DECL(hipIpcEventHandle_t)
DUMMY_DECL(hipIpcMemHandle_t)
DUMMY_DECL(hipMemcpy3DParms)
DUMMY_DECL(HIP_ARRAY3D_DESCRIPTOR)
DUMMY_DECL(hipFuncAttributes)
DUMMY_DECL(hipPointerAttribute_t)

std::ostream& operator<<(std::ostream& out, const dim3 obj);

#if (HIP_VERSION_MAJOR == 4 && HIP_VERSION_MINOR >= 3) || HIP_VERSION_MAJOR > 4
DUMMY_DECL(hipExternalMemoryHandleDesc)
DUMMY_DECL(hipExternalMemoryBufferDesc)
#endif

#if (HIP_VERSION_MAJOR == 4 && HIP_VERSION_MINOR >= 4) || HIP_VERSION_MAJOR > 4
DUMMY_DECL(hipResourceDesc)
DUMMY_DECL(hipKernelNodeParams)
DUMMY_DECL(hipExternalSemaphoreWaitParams)
DUMMY_DECL(hipMemsetParams)
DUMMY_DECL(hipExternalSemaphoreHandleDesc)
DUMMY_DECL(hipExternalSemaphoreSignalParams)
DUMMY_DECL(textureReference)
DUMMY_DECL(hipMipmappedArray)
#define HIP_API_ID_NUMBER HIP_API_ID_LAST
#endif

#if (HIP_VERSION_MAJOR == 5 && HIP_VERSION_MINOR >= 0) || HIP_VERSION_MAJOR > 5
DUMMY_DECL(hipHostNodeParams)
#endif
