#include <cuda.h>

#include "MonteRayDefinitions.hh"
#include "GPUErrorCheck.hh"
#include "GPUAtomicAdd.hh"
#include "ExpectedPathLength.h"

#include "RayTraceNightly_genericGPU_test_helper.hh"

template<unsigned N>
FIGenericGPUTestHelper<N>::FIGenericGPUTestHelper(unsigned num){
}

template<unsigned N>
FIGenericGPUTestHelper<N>::~FIGenericGPUTestHelper(){
}

template<unsigned N>
void FIGenericGPUTestHelper<N>::setupTimers(){
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
}

template<unsigned N>
void FIGenericGPUTestHelper<N>::stopTimers(){
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime;
	gpuErrchk( cudaPeekAtLastError() );

	cudaEventElapsedTime(&elapsedTime, start, stop );

	std::cout << "Elapsed time in CUDA kernel=" << elapsedTime << " msec" << std::endl;

	gpuErrchk( cudaPeekAtLastError() );
}

template class FIGenericGPUTestHelper<1>;
template class FIGenericGPUTestHelper<3>;

