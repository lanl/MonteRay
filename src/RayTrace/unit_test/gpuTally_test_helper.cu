#include "gpuTally_test_helper.hh"
#include "GPUErrorCheck.hh"

namespace MonteRay{

GPUTallyTestHelper::GPUTallyTestHelper(){
}

GPUTallyTestHelper::~GPUTallyTestHelper(){
}

CUDA_CALLABLE_KERNEL void kernelAddTally(struct MonteRay::gpuTally* pTally, unsigned i, float_t a, float_t b){
    pTally->tally[i] =  a + b;
    return;
}

void GPUTallyTestHelper::launchAddTally( MonteRay::gpuTallyHost* tally, unsigned i, float_t a, float_t b ){

	tally->copyToGPU();

#ifdef __CUDACC__
	cudaEvent_t sync;
	cudaEventCreate(&sync);
	kernelAddTally<<<1,1>>>( tally->ptr_device, i, a, b);

	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);
	MONTERAY_PEAKATLASTERROR(true);
#else
	kernelAddTally( tally->getPtr(), i, a, b);
#endif

    tally->copyToCPU();
}

} // end namespace
