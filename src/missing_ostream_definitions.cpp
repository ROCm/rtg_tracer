/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
#include <iostream>
#include <sstream>

#include <hip/hip_runtime_api.h>

#include "missing_ostream_definitions.h"

DUMMY_IMPL(HIP_MEMCPY3D)
DUMMY_IMPL(hip_Memcpy2D)
DUMMY_IMPL(hipPitchedPtr)
DUMMY_IMPL(hipExtent)
DUMMY_IMPL(hipArray)
DUMMY_IMPL(hipLaunchParams)
DUMMY_IMPL(hipChannelFormatDesc)
DUMMY_IMPL(HIP_ARRAY_DESCRIPTOR)
DUMMY_IMPL(hipDeviceProp_t)
DUMMY_IMPL(hipIpcEventHandle_t)
DUMMY_IMPL(hipIpcMemHandle_t)
DUMMY_IMPL(hipMemcpy3DParms)
DUMMY_IMPL(HIP_ARRAY3D_DESCRIPTOR)
DUMMY_IMPL(hipFuncAttributes)
DUMMY_IMPL(hipPointerAttribute_t)

std::ostream& operator<<(std::ostream& out, const dim3 obj) {
    return out << "{" << obj.x << "," << obj.y << "," << obj.z << "}";
}

#if (HIP_VERSION_MAJOR == 4 && HIP_VERSION_MINOR >= 3) || HIP_VERSION_MAJOR > 4
DUMMY_IMPL(hipExternalMemoryHandleDesc)
DUMMY_IMPL(hipExternalMemoryBufferDesc)
#endif

#if (HIP_VERSION_MAJOR == 4 && HIP_VERSION_MINOR >= 4) || HIP_VERSION_MAJOR > 4
DUMMY_IMPL(hipResourceDesc)
DUMMY_IMPL(hipKernelNodeParams)
DUMMY_IMPL(hipExternalSemaphoreWaitParams)
DUMMY_IMPL(hipMemsetParams)
DUMMY_IMPL(hipExternalSemaphoreHandleDesc)
DUMMY_IMPL(hipExternalSemaphoreSignalParams)
DUMMY_IMPL(textureReference)
DUMMY_IMPL(hipMipmappedArray)
#define HIP_API_ID_NUMBER HIP_API_ID_LAST
#endif

#if (HIP_VERSION_MAJOR == 5 && HIP_VERSION_MINOR >= 0) || HIP_VERSION_MAJOR > 5
DUMMY_IMPL(hipHostNodeParams)
#endif

#if (HIP_VERSION_MAJOR == 5 && HIP_VERSION_MINOR >= 2) || HIP_VERSION_MAJOR > 5
DUMMY_IMPL(hipUUID)
DUMMY_IMPL(hipKernelNodeAttrValue)
DUMMY_IMPL(hipMemAllocationProp)
DUMMY_IMPL(hipMemLocation)
DUMMY_IMPL(hipArrayMapInfo)
DUMMY_IMPL(hipMemPoolProps)
DUMMY_IMPL(hipMemPoolPtrExportData)
DUMMY_IMPL(hipMemAccessDesc)
#endif
