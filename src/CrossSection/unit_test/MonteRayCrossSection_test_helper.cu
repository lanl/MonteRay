#include "MonteRayDefinitions.hh"

#include "MonteRayCrossSection_test_helper.hh"


//MonteRayCrossSectionTestHelper::float_t
//MonteRayCrossSectionTestHelper::launchGetTotalXS( MonteRayCrossSectionHost* pXS, float_t energy){
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

MonteRayCrossSectionTestHelper::MonteRayCrossSectionTestHelper(){
}

MonteRayCrossSectionTestHelper::~MonteRayCrossSectionTestHelper(){

//	std::cout << "Debug: starting ~MonteRayCrossSectionTestHelper()" << std::endl;
//	std::cout << "Debug: exitting ~MonteRayCrossSectionTestHelper()" << std::endl;
}

void MonteRayCrossSectionTestHelper::setupTimers(){
#ifdef __CUDACC__
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#else
	timer.start();
#endif
}

void MonteRayCrossSectionTestHelper::stopTimers(){
	float elapsedTime;

#ifdef __CUDACC__
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop );
	std::cout << "Elapsed time in CUDA kernel=" << elapsedTime << " msec" << std::endl;
#else
	timer.stop();
	std::cout << "Elapsed time in non-CUDA kernel=" << timer.getTime()*1000.0 << " msec" << std::endl;
#endif


}


