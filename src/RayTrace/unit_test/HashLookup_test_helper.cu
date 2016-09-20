#include <cuda.h>
#include "global.h"
#include "gpuGlobal.h"

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
	gpuErrchk( cudaPeekAtLastError() );

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	kernelGetLowerBoundbyIndex<<<1,1>>>( pHash->ptr_device, isotope, bin, result_device);
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);

    gpuErrchk( cudaPeekAtLastError() );

	CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(unsigned)*1, cudaMemcpyDeviceToHost));
	gpuErrchk( cudaPeekAtLastError() );

	cudaFree( result_device );
	return result[0];
}

HashLookupTestHelper::HashLookupTestHelper(){
}

HashLookupTestHelper::~HashLookupTestHelper(){

//	std::cout << "Debug: starting ~SimpleCrossSectionTestHelper()" << std::endl;
//	std::cout << "Debug: exitting ~SimpleCrossSectionTestHelper()" << std::endl;
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
	gpuErrchk( cudaPeekAtLastError() );

	cudaEventElapsedTime(&elapsedTime, start, stop );

	std::cout << "Elapsed time in CUDA kernel=" << elapsedTime << " msec" << std::endl;

	gpuErrchk( cudaPeekAtLastError() );

}


