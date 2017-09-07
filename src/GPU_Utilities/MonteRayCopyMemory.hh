#ifndef MONTERAYMANAGEDMEMORY_HH_
#define MONTERAYMANAGEDMEMORY_HH_
#include <cstring>

#include "GPUErrorCheck.hh"
#include <MonteRayMemory.hh>

namespace MonteRay {

template<typename Derived>
class CopyMemoryBase {
public:

	CUDAHOST_CALLABLE_MEMBER CopyMemoryBase(){
		if( debug ) {
			std::cout << "Debug: CopyMemoryBase::CopyMemoryBase() -- allocating " << sizeof( Derived ) << " bytes\n";
		}
		devicePtr = (Derived*) MonteRayDeviceAlloc( sizeof( Derived ) );
		intermediatePtr = (Derived*) MonteRayHostAlloc(sizeof( Derived ), isManagedMemory);
		intermediatePtr->Derived::init();
		intermediatePtr->isCudaIntermediate = true;
	}

	CUDAHOST_CALLABLE_MEMBER virtual ~CopyMemoryBase(){
		deleteGPUMemory();
	}

	CUDAHOST_CALLABLE_MEMBER virtual void init() = 0;
	CUDAHOST_CALLABLE_MEMBER virtual void copy(const Derived* rhs) = 0;

	CUDAHOST_CALLABLE_MEMBER static void* operator new(size_t len) {

#ifndef __CUDA_ARCH__
		if( debug ) {
			std::cout << "Debug: CopyMemoryBase::new -- Custom new operator, size=" << len << "\n";
		}
#endif
		return MonteRayHostAlloc(len, isManagedMemory );
	}

	CUDAHOST_CALLABLE_MEMBER static void* operator new[](size_t len) {

#ifndef __CUDA_ARCH__
		if( debug ) {
			std::cout << "Debug: CopyMemoryBase::new[] -- Custom new[] operator, size=" << len << "\n";
		}
#endif
		return MonteRayHostAlloc(len, isManagedMemory );
	}

	CUDAHOST_CALLABLE_MEMBER virtual void copyToGPU(void) {
		if( debug ) {
			std::cout << "Debug: CopyMemoryBase::copyToGPU() \n";
		}
#ifdef __CUDACC__
		Derived* ptr = ( Derived* ) this;
		intermediatePtr->Derived::copy( ptr );
		intermediatePtr->isCudaIntermediate = false;
		CUDA_CHECK_RETURN( cudaMemcpy(devicePtr, intermediatePtr, sizeof(Derived), cudaMemcpyHostToDevice));
		intermediatePtr->isCudaIntermediate = true;
#else
		throw std::runtime_error( "CopyMemoryBase::copyToGPU -- not valid without CUDA.");
#endif
	}

	CUDAHOST_CALLABLE_MEMBER virtual void copyToCPU() {
#ifdef __CUDACC__
		CUDA_CHECK_RETURN( cudaMemcpy(intermediatePtr, devicePtr, sizeof(Derived), cudaMemcpyDeviceToHost));
		intermediatePtr->isCudaIntermediate = true;
		Derived* ptr = ( Derived* ) this;
		ptr->Derived::copy( intermediatePtr );
#else
		throw std::runtime_error( "CopyMemoryBase::copyToGPU -- not valid without CUDA.");
#endif
	}

	CUDAHOST_CALLABLE_MEMBER virtual void deleteGPUMemory(){
		MonteRayHostFree(intermediatePtr, isManagedMemory);
		MonteRayDeviceFree( devicePtr );
	}

	CUDAHOST_CALLABLE_MEMBER static void operator delete(void* ptr) {

#ifndef __CUDA_ARCH__
		if( debug ) {
			std::cout << "Debug: CopyMemoryBase::delete -- Custom delete operator.\n";
		}
#endif
		MonteRayHostFree(ptr, isManagedMemory);
	}

	Derived* intermediatePtr = NULL;
	Derived* devicePtr = NULL;

	bool isCudaIntermediate = false;

	static const bool isManagedMemory = false;
	static const bool debug = false;
};

} // end namespace



#endif /* MONTERAYMANAGEDMEMORY_HH_ */
