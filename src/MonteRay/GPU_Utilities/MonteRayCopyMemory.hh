#ifndef MONTERAYCOPYMEMORY_HH_
#define MONTERAYCOPYMEMORY_HH_
#include <cstring>

#include "MonteRayMemory.hh"
#include "MonteRayTypes.hh"

namespace MonteRay {

template<class Derived>
class CopyMemoryBase {
public:

    CUDAHOST_CALLABLE_MEMBER CopyMemoryBase();

    CUDAHOST_CALLABLE_MEMBER virtual ~CopyMemoryBase();

    CUDAHOST_CALLABLE_MEMBER virtual void init() = 0;
    CUDAHOST_CALLABLE_MEMBER virtual void copy(const Derived* rhs) = 0;
    CUDAHOST_CALLABLE_MEMBER virtual std::string className() = 0;

    CUDAHOST_CALLABLE_MEMBER static void* operator new(size_t len);

    CUDAHOST_CALLABLE_MEMBER static void* operator new[](size_t len);

    CUDAHOST_CALLABLE_MEMBER virtual void copyToGPU(void);

    CUDAHOST_CALLABLE_MEMBER virtual void copyToCPU();

    CUDAHOST_CALLABLE_MEMBER virtual void deleteGPUMemory();

    CUDAHOST_CALLABLE_MEMBER static void operator delete(void* ptr);

    const Derived* getDevicePtr(void) const { return devicePtr; }
    const Derived* getPtr(void) const { return static_cast<const Derived*>(this); }


    Derived* intermediatePtr = NULL;
    Derived* devicePtr = NULL;

    bool isCudaIntermediate = false;

    static const bool isManagedMemory = false;

#ifndef NDEBUG
    static const bool debug = false;
#endif
};

} // end namespace

#endif /* MONTERAYCOPYMEMORY_HH_ */
