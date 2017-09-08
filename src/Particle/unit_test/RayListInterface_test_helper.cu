#include <cuda.h>

#include <iostream>

#include "MonteRayDefinitions.hh"
#include "GPUErrorCheck.hh"
#include "RayListInterface.hh"

#include "RayListInterface_test_helper.hh"

namespace MonteRay {

#ifdef CUDA
template< unsigned N> __global__ void
testGetCapacity(RayList_t<N>* pRayList, MonteRay::RayListSize_t* results){
	results[0] = pRayList->capacity();
	return;
}
#endif

template< unsigned N>
MonteRay::RayListSize_t
RayListInterfaceTester<N>::launchGetCapacity( unsigned nBlocks, unsigned nThreads, RayListInterface<N>& CPs) {
	MonteRay::RayListSize_t* result_device;
	MonteRay::RayListSize_t* result;
	size_t allocSize = sizeof( MonteRay::RayListSize_t ) * 1;
	CUDA_CHECK_RETURN( cudaMalloc( &result_device, allocSize ));
	result = (MonteRay::RayListSize_t*) malloc( allocSize );

	cudaEvent_t sync;
	cudaEventCreate(&sync);
    testGetCapacity<<<nBlocks,nThreads>>>(CPs.getPtrPoints()->devicePtr, result_device);
    gpuErrchk( cudaPeekAtLastError() );
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);

	CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(MonteRay::RayListSize_t)*1, cudaMemcpyDeviceToHost));

	cudaFree( result_device );
	MonteRay::RayListSize_t value = *result;
	free(result);
	return value;
}

#ifdef CUDA
template< unsigned N>
__global__ void testSumEnergy(MonteRay::RayList_t<N>* ParticleRayList, gpuFloatType_t* results){
	gpuFloatType_t total = 0.0f;
	for(unsigned i=0; i< ParticleRayList->size(); ++i ) {
		total += ParticleRayList->getEnergy(i);
	}
	results[0] = total;
	return;
}
#endif

template< unsigned N>
gpuFloatType_t
RayListInterfaceTester<N>::launchTestSumEnergy( unsigned nBlocks, unsigned nThreads, RayListInterface<N>& CPs) {
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

template< unsigned N>
RayListInterfaceTester<N>::RayListInterfaceTester(){
}

template< unsigned N>
RayListInterfaceTester<N>::~RayListInterfaceTester(){
//		cudaDeviceReset();
}

template< unsigned N>
void RayListInterfaceTester<N>::setupTimers(){
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
}

template< unsigned N>
void RayListInterfaceTester<N>::stopTimers(){
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime;

	cudaEventElapsedTime(&elapsedTime, start, stop );

	std::cout << "Elapsed time in CUDA kernel=" << elapsedTime << " msec" << std::endl;

}

} //end namespace

template class MonteRay::RayListInterfaceTester<1>;
template class MonteRay::RayListInterfaceTester<3>;
