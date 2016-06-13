#include <cuda.h>
#include "global.h"
#include "CollisionPoints.h"

#include "CollisionPoints_test_helper.hh"


#ifdef CUDA
__global__ void testGetCapacity(CollisionPoints* pXS, CollisionPointsSize_t* results){
	results[0] = capacity(pXS);
	return;
}
#endif

CollisionPointsSize_t
CollisionPointsTester::launchGetCapacity( unsigned nBlocks, unsigned nThreads, CollisionPointsHost& CPs) {
	CollisionPointsSize_t* result_device;
	CollisionPointsSize_t result[1];
	CUDA_CHECK_RETURN( cudaMalloc( &result_device, sizeof( CollisionPointsSize_t) * 1 ));
	gpuErrchk( cudaPeekAtLastError() );

	cudaEvent_t sync;
	cudaEventCreate(&sync);
    testGetCapacity<<<nBlocks,nThreads>>>(CPs.ptrPoints_device, result_device);
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);

    gpuErrchk( cudaPeekAtLastError() );

	CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(CollisionPointsSize_t)*1, cudaMemcpyDeviceToHost));
	gpuErrchk( cudaPeekAtLastError() );

	cudaFree( result_device );
	CollisionPointsSize_t value = *result;
	return value;
}

#ifdef CUDA
__global__ void testSumEnergy(CollisionPoints* pXS, gpuFloatType_t* results){
	gpuFloatType_t total = 0.0f;
	for(unsigned i=0; i<size(pXS); ++i ) {
		total += getEnergy(pXS, i);
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
	gpuErrchk( cudaPeekAtLastError() );

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	testSumEnergy<<<nBlocks,nThreads>>>(CPs.ptrPoints_device, result_device);
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);
	gpuErrchk( cudaPeekAtLastError() );

	CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(gpuFloatType_t)*1, cudaMemcpyDeviceToHost));
	gpuErrchk( cudaPeekAtLastError() );

	cudaFree( result_device );
	return result[0];
}

CollisionPointsTester::CollisionPointsTester(){
	int deviceCount;

	cuInit(0);
	cuDeviceGetCount(&deviceCount);
	if (deviceCount == 0) {
		printf("No CUDA-compatible devices found\n");
		exit(1);
	}
	printf("Number of CUDA devices=%d\n",deviceCount);
	gpuErrchk( cudaPeekAtLastError() );
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
	gpuErrchk( cudaPeekAtLastError() );

	cudaEventElapsedTime(&elapsedTime, start, stop );

	std::cout << "Elapsed time in CUDA kernel=" << elapsedTime << " msec" << std::endl;

	gpuErrchk( cudaPeekAtLastError() );

}


