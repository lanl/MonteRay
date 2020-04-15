#ifndef MONTERAYMANAGEDMEMORY_HH_
#define MONTERAYMANAGEDMEMORY_HH_

#include "MonteRayMemory.hh"
#include "MonteRayTypes.hh"
#include "StreamAndEvent.hh"

namespace MonteRay {

class ManagedMemoryBase {
public:

    CUDAHOST_CALLABLE_MEMBER static void* operator new(size_t len);

    CUDAHOST_CALLABLE_MEMBER static void* operator new[](size_t len);

    CUDAHOST_CALLABLE_MEMBER void copyToGPU(const cuda::StreamPointer& stream = {}, MonteRayGPUProps device = MonteRayGPUProps() );

    CUDAHOST_CALLABLE_MEMBER void copyToCPU(const cuda::StreamPointer& stream = {});

    CUDAHOST_CALLABLE_MEMBER static void operator delete(void* ptr);

    static const bool debug = false;
    static const bool isManagedMemory = true;
};

} // end namespace



#endif /* MONTERAYMANAGEDMEMORY_HH_ */
