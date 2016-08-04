#include <cuda.h>
#include "global.h"
#include "gpuGlobal.h"

#include "genericGPU_test_helper.hh"

GenericGPUTestHelper::GenericGPUTestHelper(){
	int deviceCount;
}

GenericGPUTestHelper::~GenericGPUTestHelper(){
}

void GenericGPUTestHelper::setupTimers(){
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
}

void GenericGPUTestHelper::stopTimers(){
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime;
	gpuErrchk( cudaPeekAtLastError() );

	cudaEventElapsedTime(&elapsedTime, start, stop );

	std::cout << "Elapsed time in CUDA kernel=" << elapsedTime << " msec" << std::endl;

	gpuErrchk( cudaPeekAtLastError() );
}


