#include <cuda.h>

#include <iostream>

#include "MonteRayDefinitions.hh"
#include "GPUErrorCheck.hh"

#include "HashLookup_test_helper.hh"


#ifdef CUDA
__global__ void kernelGetLowerBoundbyIndex(HashLookup* pHash, unsigned isotope, unsigned bin, unsigned* result){
    result[0] = getLowerBoundbyIndex( pHash, isotope, bin);
    return;
}
#endif

unsigned
HashLookupTestHelper::launchGetLowerBoundbyIndex( HashLookupHost* pHash, unsigned isotope, unsigned bin){
	unsigned* result_device;
	unsigned result[1];
	CUDA_CHECK_RETURN( cudaMalloc( &result_device, sizeof( unsigned) * 1 ));

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	kernelGetLowerBoundbyIndex<<<1,1>>>( pHash->ptr_device, isotope, bin, result_device);
	gpuErrchk( cudaPeekAtLastError() );
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);

	CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(unsigned)*1, cudaMemcpyDeviceToHost));

	cudaFree( result_device );
	return result[0];
}

HashLookupTestHelper::HashLookupTestHelper(){
}

HashLookupTestHelper::~HashLookupTestHelper(){

//	std::cout << "Debug: starting ~MonteRayCrossSectionTestHelper()" << std::endl;
//	std::cout << "Debug: exitting ~MonteRayCrossSectionTestHelper()" << std::endl;
}

void HashLookupTestHelper::setupTimers(){
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
}

void HashLookupTestHelper::stopTimers(){
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime;

	cudaEventElapsedTime(&elapsedTime, start, stop );

	std::cout << "Elapsed time in CUDA kernel=" << elapsedTime << " msec" << std::endl;
}


