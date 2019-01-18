#include <iostream>

#include "MonteRayDefinitions.hh"
#include "GPUErrorCheck.hh"

#include "HashLookup_test_helper.hh"

CUDA_CALLABLE_KERNEL  kernelGetLowerBoundbyIndex(const HashLookup* pHash, unsigned isotope, unsigned bin, unsigned* result){
    result[0] = getLowerBoundbyIndex( pHash, isotope, bin);
    return;
}

unsigned
HashLookupTestHelper::launchGetLowerBoundbyIndex( const HashLookupHost* pHash, unsigned isotope, unsigned bin){
	unsigned result[1];

#ifdef __CUDACC__
	unsigned* result_device;
	CUDA_CHECK_RETURN( cudaMalloc( &result_device, sizeof( unsigned) * 1 ));

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	kernelGetLowerBoundbyIndex<<<1,1>>>( pHash->ptr_device, isotope, bin, result_device);
	gpuErrchk( cudaPeekAtLastError() );
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);

	CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(unsigned)*1, cudaMemcpyDeviceToHost));

	cudaFree( result_device );
#else
	kernelGetLowerBoundbyIndex( pHash->getPtr(), isotope, bin, result);
#endif

	return result[0];
}

HashLookupTestHelper::HashLookupTestHelper(){
}

HashLookupTestHelper::~HashLookupTestHelper(){

//	std::cout << "Debug: starting ~MonteRayCrossSectionTestHelper()" << std::endl;
//	std::cout << "Debug: exitting ~MonteRayCrossSectionTestHelper()" << std::endl;
}

void HashLookupTestHelper::setupTimers(){
#ifdef __CUDACC__
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#else
	timer.start();
#endif
}

void HashLookupTestHelper::stopTimers(){
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


