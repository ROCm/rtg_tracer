#include <iostream>
#include <sstream>

#define DUMMY(T)  \
std::ostream& operator<<(std::ostream& out, const T obj) { \
    return out << #T; \
}

DUMMY(HIP_MEMCPY3D)
DUMMY(hip_Memcpy2D)
DUMMY(hipPitchedPtr)
DUMMY(hipExtent)
DUMMY(hipArray)
DUMMY(hipLaunchParams)
DUMMY(hipChannelFormatDesc)
DUMMY(HIP_ARRAY_DESCRIPTOR)
DUMMY(hipDeviceProp_t)
DUMMY(hipIpcEventHandle_t)
DUMMY(hipIpcMemHandle_t)
DUMMY(hipMemcpy3DParms)
DUMMY(HIP_ARRAY3D_DESCRIPTOR)
DUMMY(hipFuncAttributes)
DUMMY(hipPointerAttribute_t)

std::ostream& operator<<(std::ostream& out, const dim3 obj) {
    return out << "{" << obj.x << "," << obj.y << "," << obj.z << "}";
}

#if (HIP_VERSION_MAJOR == 4 && HIP_VERSION_MINOR >= 3) || HIP_VERSION_MAJOR > 4
DUMMY(hipExternalMemoryHandleDesc)
DUMMY(hipExternalMemoryBufferDesc)
#endif

#if (HIP_VERSION_MAJOR == 4 && HIP_VERSION_MINOR >= 4) || HIP_VERSION_MAJOR > 4
DUMMY(hipResourceDesc)
DUMMY(hipKernelNodeParams)
DUMMY(hipExternalSemaphoreWaitParams)
DUMMY(hipMemsetParams)
DUMMY(hipExternalSemaphoreHandleDesc)
DUMMY(hipExternalSemaphoreSignalParams)
DUMMY(textureReference)
DUMMY(hipMipmappedArray)
#define HIP_API_ID_NUMBER HIP_API_ID_LAST
#endif

#if (HIP_VERSION_MAJOR == 5 && HIP_VERSION_MINOR >= 0) || HIP_VERSION_MAJOR > 5
DUMMY(hipHostNodeParams)
#endif
