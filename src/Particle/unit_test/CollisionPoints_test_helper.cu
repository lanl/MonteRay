#include <cuda.h>

#include <iostream>

#include "MonteRayDefinitions.hh"
#include "GPUErrorCheck.hh"
#include "CollisionPoints.h"

#include "CollisionPoints_test_helper.hh"

using namespace MonteRay;

#ifdef CUDA
__global__ void testGetCapacity(CollisionPoints* pXS, CollisionPointsHost::CollisionPointsSize_t* results){
	results[0] = pXS->capacity();
	return;
}
#endif

CollisionPointsHost::CollisionPointsSize_t
CollisionPointsTester::launchGetCapacity( unsigned nBlocks, unsigned nThreads, CollisionPointsHost& CPs) {
	CollisionPointsHost::CollisionPointsSize_t* result_device;
	CollisionPointsHost::CollisionPointsSize_t* result;
	size_t allocSize = sizeof( CollisionPointsHost::CollisionPointsSize_t) * 1;
	CUDA_CHECK_RETURN( cudaMalloc( &result_device, allocSize ));
	result = (CollisionPointsHost::CollisionPointsSize_t*) malloc( allocSize );

	cudaEvent_t sync;
	cudaEventCreate(&sync);
    testGetCapacity<<<nBlocks,nThreads>>>(CPs.ptrPoints_device, result_device);
    gpuErrchk( cudaPeekAtLastError() );
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);

	CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(CollisionPointsHost::CollisionPointsSize_t)*1, cudaMemcpyDeviceToHost));

	cudaFree( result_device );
	CollisionPointsHost::CollisionPointsSize_t value = *result;
	free(result);
	return value;
}

#ifdef CUDA
__global__ void testSumEnergy(CollisionPoints* pXS, gpuFloatType_t* results){
	gpuFloatType_t total = 0.0f;
	for(unsigned i=0; i< pXS->size(); ++i ) {
		total += pXS->getEnergy(i);
	}
	results[0] = total;
	return;
}
#endif

gpuFloatType_t
CollisionPointsTester::launchTestSumEnergy( unsigned nBlocks, unsigned nThreads, CollisionPointsHost& CPs) {
	gpuFloatType_t* result_device;
	gpuFloatType_t result[1];
	CUDA_CHECK_RETURN( cudaMalloc( &result_device, sizeof( gpuFloatType_t) * 1 ));

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	testSumEnergy<<<nBlocks,nThreads>>>(CPs.ptrPoints_device, result_device);
	gpuErrchk( cudaPeekAtLastError() );
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);

	CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(gpuFloatType_t)*1, cudaMemcpyDeviceToHost));

	cudaFree( result_device );
	return result[0];
}

CollisionPointsTester::CollisionPointsTester(){
}

CollisionPointsTester::~CollisionPointsTester(){
//		cudaDeviceReset();
}

void CollisionPointsTester::setupTimers(){
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
}

void CollisionPointsTester::stopTimers(){
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime;

	cudaEventElapsedTime(&elapsedTime, start, stop );

	std::cout << "Elapsed time in CUDA kernel=" << elapsedTime << " msec" << std::endl;

}


