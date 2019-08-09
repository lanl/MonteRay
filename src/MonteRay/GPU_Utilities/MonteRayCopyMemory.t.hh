#ifndef MONTERAYCOPYMEMORY_T_HH_
#define MONTERAYCOPYMEMORY_T_HH_

#include "MonteRayCopyMemory.hh"
#include "GPUErrorCheck.hh"
#include "MonteRayParallelAssistant.hh"

namespace MonteRay {

template<class Derived>
CUDAHOST_CALLABLE_MEMBER
CopyMemoryBase<Derived>::CopyMemoryBase(){

#ifdef DEBUG
    if( debug ) {
        std::cout << "Debug: CopyMemoryBase::CopyMemoryBase() -- allocating " << sizeof( Derived ) << " bytes\n";
    }
#endif

#ifdef __CUDACC__
    devicePtr = (Derived*) MONTERAYDEVICEALLOC( sizeof( Derived ), intermediatePtr->Derived::className() + std::string("::devicePtr") );
#endif
    intermediatePtr = (Derived*) MONTERAYHOSTALLOC(sizeof( Derived ), isManagedMemory, intermediatePtr->Derived::className() + std::string("::intermediatePtr"));
    intermediatePtr->Derived::init();
    intermediatePtr->isCudaIntermediate = true;
}

template<class Derived>
CUDAHOST_CALLABLE_MEMBER
CopyMemoryBase<Derived>::~CopyMemoryBase(){

    if( ! isCudaIntermediate ) {

#ifdef DEBUG
        if( debug ) std::cout << "Debug: CopyMemoryBase::~CopyMemoryBase() -- calling intermediatePtr->Derived::~Derived()\n";
#endif

        intermediatePtr->Derived::~Derived();

#ifdef DEBUG
        if( debug ) std::cout << "Debug: CopyMemoryBase::~CopyMemoryBase() -- calling deleteGPUMemory()\n";
#endif
        deleteGPUMemory();
    }
}

template<class Derived>
CUDAHOST_CALLABLE_MEMBER
void*
CopyMemoryBase<Derived>::operator new(size_t len) {

#ifndef __CUDA_ARCH__
#ifdef DEBUG
    if( debug ) {
        std::cout << "Debug: CopyMemoryBase::new -- Custom new operator, size=" << len << "\n";
    }
#endif
#endif
    return MONTERAYHOSTALLOC(len, isManagedMemory, std::string("MonteRayCopyMemory::new()") );
}

template<class Derived>
CUDAHOST_CALLABLE_MEMBER
void*
CopyMemoryBase<Derived>::operator new[](size_t len) {

#ifndef __CUDA_ARCH__
#ifdef DEBUG
    if( debug ) {
        std::cout << "Debug: CopyMemoryBase::new[] -- Custom new[] operator, size=" << len << "\n";
    }
#endif
#endif
    return MONTERAYHOSTALLOC(len, isManagedMemory, std::string("MonteRayCopyMemory::::new[]") );
}

template<class Derived>
CUDAHOST_CALLABLE_MEMBER
void
CopyMemoryBase<Derived>::copyToGPU(void) {

#ifdef DEBUG
    if( debug ) {
        std::cout << "Debug: CopyMemoryBase::copyToGPU() \n";
    }
#endif

#ifdef __CUDACC__
    if( ! MonteRay::isWorkGroupMaster() ) return;
    Derived* ptr = ( Derived* ) this;
    intermediatePtr->Derived::copy( ptr );
    intermediatePtr->isCudaIntermediate = false;
    CUDA_CHECK_RETURN( cudaMemcpy(devicePtr, intermediatePtr, sizeof(Derived), cudaMemcpyHostToDevice));
    intermediatePtr->isCudaIntermediate = true;
#endif
}

template<class Derived>
CUDAHOST_CALLABLE_MEMBER
void
CopyMemoryBase<Derived>::copyToCPU() {
#ifdef __CUDACC__
    CUDA_CHECK_RETURN( cudaMemcpy(intermediatePtr, devicePtr, sizeof(Derived), cudaMemcpyDeviceToHost));
    intermediatePtr->isCudaIntermediate = true;
    Derived* ptr = ( Derived* ) this;
    ptr->Derived::copy( intermediatePtr );
#endif
}

template<class Derived>
CUDAHOST_CALLABLE_MEMBER
void
CopyMemoryBase<Derived>::deleteGPUMemory(){

#ifdef DEBUG
    if( debug ) std::cout << "Debug: CopyMemoryBase::deleteGPUMemory -- calling MonteRayHostFree( intermediatePtr )\n";
#endif

    MonteRayHostFree(intermediatePtr, isManagedMemory);
#ifdef __CUDACC__
#ifdef DEBUG
    if( debug ) std::cout << "Debug: CopyMemoryBase::deleteGPUMemory -- calling MonteRayDeviceFree( devicePtr )\n";
#endif
    MonteRayDeviceFree( devicePtr );
#endif
}

template<class Derived>
CUDAHOST_CALLABLE_MEMBER
void
CopyMemoryBase<Derived>::operator delete(void* ptr) {

#ifndef __CUDA_ARCH__
#ifdef DEBUG
    if( debug ) {
        std::cout << "Debug: CopyMemoryBase::delete -- Custom delete operator.\n";
    }
#endif
#endif
    MonteRayHostFree(ptr, isManagedMemory);
}

} /* end namespace */

#endif /* MONTERAYCOPYMEMORY_T_HH_ */
