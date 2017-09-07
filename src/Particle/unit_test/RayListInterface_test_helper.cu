#include <cuda.h>

#include <iostream>

#include "MonteRayDefinitions.hh"
#include "GPUErrorCheck.hh"
#include "RayListInterface.hh"

#include "RayListInterface_test_helper.hh"

using namespace MonteRay;

#ifdef CUDA
__global__ void testGetCapacity(ParticleRayList* pRayList, RayListInterface::RayListSize_t* results){
	results[0] = pRayList->capacity();
	return;
}
#endif

RayListInterface::RayListSize_t
RayListInterfaceTester::launchGetCapacity( unsigned nBlocks, unsigned nThreads, RayListInterface& CPs) {
	RayListInterface::RayListSize_t* result_device;
	RayListInterface::RayListSize_t* result;
	size_t allocSize = sizeof( RayListInterface::RayListSize_t) * 1;
	CUDA_CHECK_RETURN( cudaMalloc( &result_device, allocSize ));
	result = (RayListInterface::RayListSize_t*) malloc( allocSize );

	cudaEvent_t sync;
	cudaEventCreate(&sync);
    testGetCapacity<<<nBlocks,nThreads>>>(CPs.getPtrPoints()->devicePtr, result_device);
    gpuErrchk( cudaPeekAtLastError() );
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);

	CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(RayListInterface::RayListSize_t)*1, cudaMemcpyDeviceToHost));

	cudaFree( result_device );
	RayListInterface::RayListSize_t value = *result;
	free(result);
	return value;
}

#ifdef CUDA
__global__ void testSumEnergy(ParticleRayList* ParticleRayList, gpuFloatType_t* results){
	gpuFloatType_t total = 0.0f;
	for(unsigned i=0; i< ParticleRayList->size(); ++i ) {
		total += ParticleRayList->getEnergy(i);
	}
	results[0] = total;
	return;
}
#endif

gpuFloatType_t
RayListInterfaceTester::launchTestSumEnergy( unsigned nBlocks, unsigned nThreads, RayListInterface& CPs) {
	gpuFloatType_t* result_device;
	gpuFloatType_t result[1];
	CUDA_CHECK_RETURN( cudaMalloc( &result_device, sizeof( gpuFloatType_t) * 1 ));

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	testSumEnergy<<<nBlocks,nThreads>>>(CPs.getPtrPoints()->devicePtr, result_device);
	gpuErrchk( cudaPeekAtLastError() );
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);

	CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(gpuFloatType_t)*1, cudaMemcpyDeviceToHost));

	cudaFree( result_device );
	return result[0];
}

RayListInterfaceTester::RayListInterfaceTester(){
}

RayListInterfaceTester::~RayListInterfaceTester(){
//		cudaDeviceReset();
}

void RayListInterfaceTester::setupTimers(){
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
}

void RayListInterfaceTester::stopTimers(){
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime;

	cudaEventElapsedTime(&elapsedTime, start, stop );

	std::cout << "Elapsed time in CUDA kernel=" << elapsedTime << " msec" << std::endl;

}


