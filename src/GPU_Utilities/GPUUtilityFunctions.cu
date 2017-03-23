/*
 * GPUUtilityFunctions.cc
 *
 *  Created on: Mar 20, 2017
 *      Author: jsweezy, jsweezy@lanl.gov
 */

#include <cstdio>

#include "GPUErrorCheck.hh"
#include "GPUUtilityFunctions.hh"
#include "MonteRayDefinitions.hh"

namespace MonteRay{

void cudaReset(void) {
#ifdef CUDA
	CUDA_CHECK_RETURN( cudaDeviceReset() );
#endif
}

void gpuCheck() {
#ifdef CUDA
	int deviceCount;

	CUresult result_error = cuInit(0);
	if( result_error != CUDA_SUCCESS ) {
		printf("CUDA call: cuInit failed!\n");
		exit(1);
	}

	result_error = cuDeviceGetCount(&deviceCount);
	if( result_error != CUDA_SUCCESS ) {
		printf("CUDA call: cuDeviceGetCount failed!\n");
		exit(1);
	}
#endif
}

void gpuInfo() {
#ifdef CUDA
	int deviceCount;

	CUresult result_error = cuDeviceGetCount(&deviceCount);
	if( result_error != CUDA_SUCCESS ) {
		printf("CUDA call: cuDeviceGetCount failed!\n");
		exit(1);
	}

	printf("Number of CUDA devices=%d\n",deviceCount);

	for (int i = 0; i < deviceCount; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Compute capability %d,%d\n", prop.major, prop.minor);
		printf("  Memory Clock Rate (KHz): %d\n",
				prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
				prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
				2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
	}
#endif
}

int getNumberOfGPUS(void) {
	int count = 0;

#ifdef CUDA
	cudaGetDeviceCount( &count ) ;
	count = 1;
#endif
	return count;
}

void setCudaPrintBufferSize( size_t size) {
#ifdef CUDA
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, size );
#endif
}

} /* namespace MonteRay */
