#include <cuda.h>

#include <iostream>

#include "MonteRayDefinitions.hh"

#include "genericGPU_test_helper.hh"


GenericGPUTestHelper::GenericGPUTestHelper(){}

GenericGPUTestHelper::~GenericGPUTestHelper(){}

void GenericGPUTestHelper::setupTimers(){
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
}

void GenericGPUTestHelper::stopTimers(){
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime;

	cudaEventElapsedTime(&elapsedTime, start, stop );

	std::cout << "Elapsed time in CUDA kernel=" << elapsedTime << " msec" << std::endl;
}


