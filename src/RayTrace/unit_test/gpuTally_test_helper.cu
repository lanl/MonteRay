#include "gpuTally_test_helper.hh"

GPUTallyTestHelper::GPUTallyTestHelper(){
}

GPUTallyTestHelper::~GPUTallyTestHelper(){
}

__global__ void kernelAddTally(struct MonteRay::gpuTally* pTally, unsigned i, float_t a, float_t b){
    pTally->tally[i] =  a + b;
    return;
}

void GPUTallyTestHelper::launchAddTally( MonteRay::gpuTallyHost* tally, unsigned i, float_t a, float_t b ){

	tally->copyToGPU();

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	kernelAddTally<<<1,1>>>( tally->ptr_device, i, a, b);
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);

    gpuErrchk( cudaPeekAtLastError() );

    tally->copyToCPU();
}
