#ifndef MONTERAYMANAGEDMEMORY_HH_
#define MONTERAYMANAGEDMEMORY_HH_

#include "MonteRayMemory.hh"
#include "MonteRayTypes.hh"

//class cudaStream_t;

namespace MonteRay {

class ManagedMemoryBase {
public:

    CUDAHOST_CALLABLE_MEMBER static void* operator new(size_t len);

    CUDAHOST_CALLABLE_MEMBER static void* operator new[](size_t len);

    CUDAHOST_CALLABLE_MEMBER void copyToGPU(cudaStream_t stream = NULL, MonteRayGPUProps device = MonteRayGPUProps() );

    CUDAHOST_CALLABLE_MEMBER void copyToCPU(cudaStream_t stream = NULL);

    CUDAHOST_CALLABLE_MEMBER static void operator delete(void* ptr);

    static const bool debug = false;
    static const bool isManagedMemory = true;
};

} // end namespace



#endif /* MONTERAYMANAGEDMEMORY_HH_ */
