#include "MonteRayManagedMemory.hh"

#include "MonteRayDefinitions.hh"

namespace MonteRay {

CUDAHOST_CALLABLE_MEMBER
void*
ManagedMemoryBase::operator new(size_t len) {
#ifndef __CUDA_ARCH__
    if( debug ) {
        std::cout << "Debug: AllocBase:new -- Custom new operator, size=" << len << "\n";
    }
#endif
    return MONTERAYHOSTALLOC(len, isManagedMemory, std::string("ManagedMemoryBase::new()") );
}

CUDAHOST_CALLABLE_MEMBER
void*
ManagedMemoryBase::operator new[](size_t len) {
#ifndef __CUDA_ARCH__
    if( debug ) {
        std::cout << "Debug: AllocBase:new[] -- Custom new[] operator, size=" << len << "\n";
    }
#endif
    return MONTERAYHOSTALLOC(len, isManagedMemory, std::string("ManagedMemoryBase::new[]"));
}

// TPB Does this due what we think it does?  Need to check sizeof(*this)
CUDAHOST_CALLABLE_MEMBER
void
ManagedMemoryBase::copyToGPU(const cuda::StreamPointer& stream, MonteRayGPUProps device ) {
#ifdef __CUDACC__
    cudaMemAdvise(this, sizeof(*this), cudaMemAdviseSetReadMostly, device.deviceID);
    if( device.deviceProps->concurrentManagedAccess ) {
        cudaMemPrefetchAsync(this, sizeof(*this), device.deviceID, *stream );
    }
#endif
}

CUDAHOST_CALLABLE_MEMBER
void
ManagedMemoryBase::copyToCPU(const cuda::StreamPointer& stream) {
#ifdef __CUDACC__
    cudaMemPrefetchAsync(this, sizeof(*this), cudaCpuDeviceId, *stream );
#endif
}

CUDAHOST_CALLABLE_MEMBER
void
ManagedMemoryBase::operator delete(void* ptr) {
#ifndef __CUDA_ARCH__
     if( debug ) {
         std::cout << "Debug: Custom delete operator.\n";
     }
#endif
     MonteRayHostFree(ptr, true);
}

} // end namespace
