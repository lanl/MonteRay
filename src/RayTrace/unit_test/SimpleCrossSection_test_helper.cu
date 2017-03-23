#include <cuda.h>
#include "MonteRayDefinitions.hh"

#include "SimpleCrossSection_test_helper.hh"


//SimpleCrossSectionTestHelper::float_t
//SimpleCrossSectionTestHelper::launchGetTotalXS( SimpleCrossSectionHost* pXS, float_t energy){
//	float_t* result_device;
//	float_t result[1];
//	CUDA_CHECK_RETURN( cudaMalloc( &result_device, sizeof( float_t) * 1 ));
//
//	cudaEvent_t sync;
//	cudaEventCreate(&sync);
//	kernelGetTotalXS<<<1,1>>>( pXS->xs_device, energy, result_device);
//  gpuErrchk( cudaPeekAtLastError() );
//	cudaEventRecord(sync, 0);
//	cudaEventSynchronize(sync);
//
//	CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(float_t)*1, cudaMemcpyDeviceToHost));
//
//	cudaFree( result_device );
//	return result[0];
//}

SimpleCrossSectionTestHelper::SimpleCrossSectionTestHelper(){
}

SimpleCrossSectionTestHelper::~SimpleCrossSectionTestHelper(){

//	std::cout << "Debug: starting ~SimpleCrossSectionTestHelper()" << std::endl;
//	std::cout << "Debug: exitting ~SimpleCrossSectionTestHelper()" << std::endl;
}

void SimpleCrossSectionTestHelper::setupTimers(){
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
}

void SimpleCrossSectionTestHelper::stopTimers(){
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime;

	cudaEventElapsedTime(&elapsedTime, start, stop );

	std::cout << "Elapsed time in CUDA kernel=" << elapsedTime << " msec" << std::endl;
}


