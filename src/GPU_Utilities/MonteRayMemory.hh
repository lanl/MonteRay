#ifndef MONTERAYMEMORY_HH_
#define MONTERAYMEMORY_HH_

#include <memory>
#include <stdexcept>
#include <cstdlib>
#include <typeinfo>
#include <iostream>

#ifdef __CUDACC__
#include "cuda_runtime_api.h"
#include "MonteRayDefinitions.hh"
#include "GPUErrorCheck.hh"
#endif

namespace MonteRay {

/// MonteRay GPU Properties class
class MonteRayGPUProps {
public:
	MonteRayGPUProps() :
		deviceID( -1 ),
		numDevices( -1 )
	{
#ifdef __CUDACC__
		deviceProps = cudaDevicePropDontCare;
		cudaGetDeviceCount(&numDevices);

		if( numDevices <= 0 ) {
			throw std::runtime_error("MonteRayGPUProps::MonteRayGPUProps() -- No GPU found.");
		}

		// TODO: setup for multiple devices
		cudaGetDevice(&deviceID);

		cudaGetDeviceProperties(&deviceProps , deviceID);

		if( ! deviceProps.managedMemory ) {
			std::cout << "MONTERAY WARNING: GPU does not support managed memory.\n";
		}
		if( ! deviceProps.concurrentManagedAccess ) {
			std::cout << "MONTERAY WARNING: GPU does not support concurrent managed memory access.\n";
		}
	#else
		throw std::runtime_error("MonteRayGPUProps::MonteRayGPUProps() -- CUDA not enabled.");
	#endif
	}

	MonteRayGPUProps( const MonteRayGPUProps& rhs ) {
		deviceID = rhs.deviceID;
		numDevices = rhs.numDevices;
#ifdef __CUDACC__
		deviceProps = rhs.deviceProps;
#endif
	}

	~MonteRayGPUProps(){}

	int deviceID; ///< GPU ID Number
	int numDevices;

#ifdef __CUDACC__
	cudaDeviceProp deviceProps;
#endif
};

inline void* MonteRayHostAlloc(size_t len, bool managed = true) {
#ifdef __CUDACC__
	const bool debug = false;

	if( managed ) {
		if( debug ){
			std::cout << "Debug: MonteRayHostAlloc: allocating " << len << " bytes with cuda managed memory\n";
		}
		void *ptr;
		cudaMallocManaged(&ptr, len);
		return ptr;
	} else {
		if( debug ){
			std::cout << "Debug: MonteRayHostAlloc: allocating " << len << " bytes with malloc\n";
		}
		return std::malloc( len );
	}

#else
	return std::malloc(len);
#endif
}

inline void* MonteRayDeviceAlloc(size_t len ){
	const bool debug = false;

	void *ptr;
#ifdef __CUDACC__
	if( debug ){
		std::cout << "Debug: MonteRayHostAlloc: allocating " << len << " bytes with cudaMalloc\n";
	}
	cudaMalloc(&ptr, len );
#else
	throw std::runtime_error( "MonteRayDeviceAlloc -- can NOT allocate device memory without CUDA." );
#endif
	return ptr;
}

inline void MonteRayHostFree(void* ptr, bool managed ) noexcept {
#ifdef __CUDACC__
	if( managed ) {
		cudaFree(ptr);
	} else {
		std::free(ptr);
	}
#else
	std::free(ptr);
#endif
}

inline void MonteRayDeviceFree(void* ptr) noexcept {
#ifdef __CUDACC__
	cudaFree(ptr);
#else
	throw std::runtime_error( "MonteRayDeviceFree -- can NOT free device memory without CUDA." );
#endif
}

inline void MonteRayMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind){
#ifdef __CUDACC__
	CUDA_CHECK_RETURN( cudaMemcpy(dst, src, count, kind) );
#else
	throw std::runtime_error( "MonteRayMemcpy -- can NOT use cudaMemcpy without CUDA." );
#endif
}
} // end namespace



#endif /* MONTERAYMEMORY_HH_ */
