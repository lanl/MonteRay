
#include <iostream>

#include "MonteRayDefinitions.hh"
#include "GPUErrorCheck.hh"

#include "ManagedMemory_test_helper.hh"


CUDA_CALLABLE_KERNEL  kernelSumVectors(testClass* A, testClass* B, testClass* C) {
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
#else
	kernelSumVectors( A, B, C);
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
