#include "genericGPU_test_helper.hh"

#include <iostream>

#include "MonteRayDefinitions.hh"

GenericGPUTestHelper::GenericGPUTestHelper(){}

GenericGPUTestHelper::~GenericGPUTestHelper(){}

void GenericGPUTestHelper::setupTimers(){
#ifdef __CUDACC__
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif
}

void GenericGPUTestHelper::stopTimers(){
	float elapsedTime;

#ifdef __CUDACC__
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop );
#endif

	std::cout << "Elapsed time in CUDA kernel=" << elapsedTime << " msec" << std::endl;
}


