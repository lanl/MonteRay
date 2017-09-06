#include <cuda.h>

#include <iostream>

#include "MonteRayDefinitions.hh"
#include "GPUErrorCheck.hh"

#include "ManagedMemory_test_helper.hh"


#ifdef __CUDACC__
__global__ void kernelSumVectors(testClass* A, testClass* B, testClass* C) {
    for( unsigned i=0; i<A->N; ++i) {
    	gpuFloatType_t elementA = A->elements[i] * A->multiple;
    	gpuFloatType_t elementB = B->elements[i] * B->multiple;
    	gpuFloatType_t elementC = elementA + elementB;
    	C->elements[i] = elementC;
    }
    C->N = A->N;
    C->multiple = 1.0;
    return;
}
#endif

void
ManagedMemoryTestHelper::launchSumVectors( testClass* A, testClass* B, testClass* C){
#ifdef __CUDACC__
	cudaEvent_t sync;
	cudaEventCreate(&sync);
	setupTimers();
	kernelSumVectors<<<1,1>>>( A, B, C);
	stopTimers();
	gpuErrchk( cudaPeekAtLastError() );
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);
#endif
}

ManagedMemoryTestHelper::ManagedMemoryTestHelper(){
}

ManagedMemoryTestHelper::~ManagedMemoryTestHelper(){
}

void ManagedMemoryTestHelper::setupTimers(){
#ifdef __CUDACC__
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif
}

void ManagedMemoryTestHelper::stopTimers(){
#ifdef __CUDACC__
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime;

	cudaEventElapsedTime(&elapsedTime, start, stop );

	std::cout << "Elapsed time in CUDA kernel=" << elapsedTime << " msec" << std::endl;
#endif
}
