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
#ifdef __CUDACC__
		devicePtr = (Derived*) MONTERAYDEVICEALLOC( sizeof( Derived ), intermediatePtr->Derived::className() + std::string("::devicePtr") );
#endif
		intermediatePtr = (Derived*) MONTERAYHOSTALLOC(sizeof( Derived ), isManagedMemory, intermediatePtr->Derived::className() + std::string("::intermediatePtr"));
		intermediatePtr->Derived::init();
		intermediatePtr->isCudaIntermediate = true;

	}

	CUDAHOST_CALLABLE_MEMBER virtual ~CopyMemoryBase(){

		if( ! isCudaIntermediate ) {
			if( debug ) std::cout << "Debug: CopyMemoryBase::~CopyMemoryBase() -- calling intermediatePtr->Derived::~Derived()\n";
			intermediatePtr->Derived::~Derived();
			if( debug ) std::cout << "Debug: CopyMemoryBase::~CopyMemoryBase() -- calling deleteGPUMemory()\n";
			deleteGPUMemory();
		}

	}

	CUDAHOST_CALLABLE_MEMBER virtual void init() = 0;
	CUDAHOST_CALLABLE_MEMBER virtual void copy(const Derived* rhs) = 0;
	CUDAHOST_CALLABLE_MEMBER virtual std::string className() = 0;

	CUDAHOST_CALLABLE_MEMBER static void* operator new(size_t len) {

#ifndef __CUDA_ARCH__
		if( debug ) {
			std::cout << "Debug: CopyMemoryBase::new -- Custom new operator, size=" << len << "\n";
		}
#endif
		return MONTERAYHOSTALLOC(len, isManagedMemory, std::string("MonteRayCopyMemory::new()") );
	}

	CUDAHOST_CALLABLE_MEMBER static void* operator new[](size_t len) {

#ifndef __CUDA_ARCH__
		if( debug ) {
			std::cout << "Debug: CopyMemoryBase::new[] -- Custom new[] operator, size=" << len << "\n";
		}
#endif
		return MONTERAYHOSTALLOC(len, isManagedMemory, std::string("MonteRayCopyMemory::::new[]") );
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
#endif
	}

	CUDAHOST_CALLABLE_MEMBER virtual void copyToCPU() {
#ifdef __CUDACC__
		CUDA_CHECK_RETURN( cudaMemcpy(intermediatePtr, devicePtr, sizeof(Derived), cudaMemcpyDeviceToHost));
		intermediatePtr->isCudaIntermediate = true;
		Derived* ptr = ( Derived* ) this;
		ptr->Derived::copy( intermediatePtr );
#endif
	}

	CUDAHOST_CALLABLE_MEMBER virtual void deleteGPUMemory(){
		if( debug ) std::cout << "Debug: CopyMemoryBase::deleteGPUMemory -- calling MonteRayHostFree( intermediatePtr )\n";
		MonteRayHostFree(intermediatePtr, isManagedMemory);
#ifdef __CUDACC__
		if( debug ) std::cout << "Debug: CopyMemoryBase::deleteGPUMemory -- calling MonteRayDeviceFree( devicePtr )\n";
		MonteRayDeviceFree( devicePtr );
#endif
	}

	CUDAHOST_CALLABLE_MEMBER static void operator delete(void* ptr) {

#ifndef __CUDA_ARCH__
		if( debug ) {
			std::cout << "Debug: CopyMemoryBase::delete -- Custom delete operator.\n";
		}
#endif
		MonteRayHostFree(ptr, isManagedMemory);
	}

	const Derived* getDevicePtr(void) const { return devicePtr; }
	const Derived* getPtr(void) const { return static_cast<const Derived*>(this); }


	Derived* intermediatePtr = NULL;
	Derived* devicePtr = NULL;

	bool isCudaIntermediate = false;

	static const bool isManagedMemory = false;
	static const bool debug = false;
};

} // end namespace



#endif /* MONTERAYMANAGEDMEMORY_HH_ */
