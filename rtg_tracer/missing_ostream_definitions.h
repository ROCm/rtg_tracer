#include <iostream>
#include <sstream>

std::ostream& operator<<(std::ostream& out, const HIP_MEMCPY3D obj) {
    return out << "HIP_MEMCPY3D";
}

std::ostream& operator<<(std::ostream& out, const hip_Memcpy2D obj) {
    return out << "hip_Memcpy2D";
}

std::ostream& operator<<(std::ostream& out, const dim3 obj) {
    return out << "{" << obj.x << "," << obj.y << "," << obj.z << "}";
}

std::ostream& operator<<(std::ostream& out, const hipPitchedPtr obj) {
    return out << "hipPitchedPtr";
}

std::ostream& operator<<(std::ostream& out, const hipExtent obj) {
    return out << "hipExtent";
}

std::ostream& operator<<(std::ostream& out, const hipArray obj) {
    return out << "hipArray";
}

std::ostream& operator<<(std::ostream& out, const hipLaunchParams obj) {
    return out << "hipLaunchParams";
}

std::ostream& operator<<(std::ostream& out, const hipChannelFormatDesc obj) {
    return out << "hipChannelFormatDesc";
}

std::ostream& operator<<(std::ostream& out, const HIP_ARRAY_DESCRIPTOR obj) {
    return out << "HIP_ARRAY_DESCRIPTOR";
}

std::ostream& operator<<(std::ostream& out, const hipDeviceProp_t obj) {
    return out << "hipDeviceProp_t";
}

std::ostream& operator<<(std::ostream& out, const hipIpcEventHandle_t obj) {
    return out << "hipIpcEventHandle_t";
}

std::ostream& operator<<(std::ostream& out, const hipIpcMemHandle_t obj) {
    return out << "hipIpcMemHandle_t";
}

std::ostream& operator<<(std::ostream& out, const hipMemcpy3DParms obj) {
    return out << "hipMemcpy3DParms";
}

std::ostream& operator<<(std::ostream& out, const HIP_ARRAY3D_DESCRIPTOR obj) {
    return out << "HIP_ARRAY3D_DESCRIPTOR";
}

std::ostream& operator<<(std::ostream& out, const hipFuncAttributes obj) {
    return out << "hipFuncAttributes";
}

std::ostream& operator<<(std::ostream& out, const hipPointerAttribute_t obj) {
    return out << "hipPointerAttribute_t";
}

#if (HIP_VERSION_MAJOR == 4 && HIP_VERSION_MINOR >= 3) || HIP_VERSION_MAJOR > 4
std::ostream& operator<<(std::ostream& out, const hipExternalMemoryHandleDesc obj) {
    return out << "hipExternalMemoryHandleDesc";
}

std::ostream& operator<<(std::ostream& out, const hipExternalMemoryBufferDesc obj) {
    return out << "hipExternalMemoryBufferDesc";
}
#endif

