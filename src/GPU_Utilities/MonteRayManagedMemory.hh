#ifndef MONTERAYMANAGEDMEMORY_HH_
#define MONTERAYMANAGEDMEMORY_HH_

#include <MonteRayMemory.hh>

namespace MonteRay {

class ManagedMemoryBase {
public:
	static void* operator new(size_t len) {

#ifndef __CUDA_ARCH__
		if( debug ) {
			std::cout << "Debug: AllocBase:new -- Custom new operator, size=" << len << "\n";
		}
#endif
		return MonteRayHostAlloc(len);
	}

	static void* operator new[](size_t len) {

#ifndef __CUDA_ARCH__
		if( debug ) {
			std::cout << "Debug: AllocBase:new[] -- Custom new[] operator, size=" << len << "\n";
		}
#endif
		return MonteRayHostAlloc(len);
	}

	void copyToGPU(cudaStream_t stream = NULL, MonteRayGPUProps device = MonteRayGPUProps() ) {
#ifdef __CUDACC__
		cudaMemAdvise(this, sizeof(*this), cudaMemAdviseSetReadMostly, device.deviceID);
		if( device.deviceProps.concurrentManagedAccess ) {
			cudaMemPrefetchAsync(this, sizeof(*this), device.deviceID, NULL );
		}
#else
		throw std::runtime_error( "AllocBase::copyToGPU -- not valid without CUDA.");
#endif
	}

	void copyToCPU(cudaStream_t stream = NULL) {
#ifdef __CUDACC__
		cudaMemPrefetchAsync(this, sizeof(*this), cudaCpuDeviceId, NULL );
#else
		throw std::runtime_error( "AllocBase::copyToCPU -- not valid without CUDA.");
#endif
	}

	static void operator delete(void* ptr) {

#ifndef __CUDA_ARCH__
		if( debug ) {
			std::cout << "Debug: Custom delete operator.\n";
		}
#endif
		MonteRayHostFree(ptr, true);
	}

	static const bool debug = false;
};

} // end namespace



#endif /* MONTERAYMANAGEDMEMORY_HH_ */
