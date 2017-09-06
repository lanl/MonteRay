#ifndef MONTERAYMANAGEDMEMORY_HH_
#define MONTERAYMANAGEDMEMORY_HH_
#include <cstring>

#include "GPUErrorCheck.hh"
#include <MonteRayMemory.hh>


namespace MonteRay {

template<typename Derived>
class CopyMemoryBase {
public:

	CopyMemoryBase(){
		if( debug ) {
			std::cout << "Debug: CopyMemoryBase::CopyMemoryBase() -- allocating " << sizeof( Derived ) << " bytes\n";
		}
		devicePtr = (Derived*) MonteRayDeviceAlloc( sizeof( Derived ) );
		intermediatePtr = (Derived*) MonteRayHostAlloc(sizeof( Derived ), false);
	}
	virtual ~CopyMemoryBase(){
		deleteGPUMemory();
	}

	static void* operator new(size_t len) {

#ifndef __CUDA_ARCH__
		if( debug ) {
			std::cout << "Debug: CopyMemoryBase::new -- Custom new operator, size=" << len << "\n";
		}
#endif
		return MonteRayHostAlloc(len, false);
	}

	static void* operator new[](size_t len) {

#ifndef __CUDA_ARCH__
		if( debug ) {
			std::cout << "Debug: CopyMemoryBase::new[] -- Custom new[] operator, size=" << len << "\n";
		}
#endif
		return MonteRayHostAlloc(len, false);
	}

	virtual void initialize() {
		std::memcpy( intermediatePtr, (Derived*) this, sizeof( Derived ) );
	}

	virtual void copyToGPU(cudaStream_t stream = NULL, MonteRayGPUProps device = MonteRayGPUProps() ) {
#ifdef __CUDACC__
		CUDA_CHECK_RETURN( cudaMemcpy(devicePtr, intermediatePtr, sizeof(Derived), cudaMemcpyHostToDevice));
#else
		throw std::runtime_error( "CopyMemoryBase::copyToGPU -- not valid without CUDA.");
#endif
	}

	virtual void copyToCPU(cudaStream_t stream = NULL, MonteRayGPUProps device = MonteRayGPUProps() ) {
#ifdef __CUDACC__
		Derived* copy = (Derived*) malloc(sizeof(Derived));
		CUDA_CHECK_RETURN( cudaMemcpy(copy, devicePtr, sizeof(Derived), cudaMemcpyDeviceToHost));
		copy->intermediatePtr = intermediatePtr;
		copy->devicePtr = devicePtr;
		std::memcpy( (Derived*) this, copy, sizeof( Derived ) );
#else
		throw std::runtime_error( "CopyMemoryBase::copyToGPU -- not valid without CUDA.");
#endif
	}

	virtual void deleteGPUMemory(){
		MonteRayHostFree(intermediatePtr, false);
		MonteRayDeviceFree( devicePtr );
	}

	static void operator delete(void* ptr) {

#ifndef __CUDA_ARCH__
		if( debug ) {
			std::cout << "Debug: CopyMemoryBase::delete -- Custom delete operator.\n";
		}
#endif
		MonteRayHostFree(ptr, false);
	}

	static const bool debug = false;

	Derived* intermediatePtr = NULL;
	Derived* devicePtr = NULL;
};

} // end namespace



#endif /* MONTERAYMANAGEDMEMORY_HH_ */
