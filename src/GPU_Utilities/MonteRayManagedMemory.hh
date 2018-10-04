#ifndef MONTERAYMANAGEDMEMORY_HH_
#define MONTERAYMANAGEDMEMORY_HH_

#include "MonteRayMemory.hh"
#include "MonteRayTypes.hh"

#ifndef __CUDACC__
// forward declare cudaStream_t
class cudaStream_t;
#endif

namespace MonteRay {

class ManagedMemoryBase {
public:

    CUDAHOST_CALLABLE_MEMBER static void* operator new(size_t len);

    CUDAHOST_CALLABLE_MEMBER static void* operator new[](size_t len);

    CUDAHOST_CALLABLE_MEMBER void copyToGPU(cudaStream_t* stream = nullptr, MonteRayGPUProps device = MonteRayGPUProps() );

    CUDAHOST_CALLABLE_MEMBER void copyToCPU(cudaStream_t* stream = nullptr);

    CUDAHOST_CALLABLE_MEMBER static void operator delete(void* ptr);

    static const bool debug = false;
    static const bool isManagedMemory = true;
};

} // end namespace



#endif /* MONTERAYMANAGEDMEMORY_HH_ */
