#include <iostream>

#include "MonteRayDefinitions.hh"

#include "genericGPU_test_helper.hh"

GenericGPUTestHelper::GenericGPUTestHelper(){}

GenericGPUTestHelper::~GenericGPUTestHelper(){}

void GenericGPUTestHelper::setupTimers(){
#ifdef __CUDACC__
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#else
	timer.start();
#endif
}

void GenericGPUTestHelper::stopTimers(){
#ifdef __CUDACC__
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime;

	cudaEventElapsedTime(&elapsedTime, start, stop );
	std::cout << "Elapsed time in CUDA kernel=" << elapsedTime << " msec" << std::endl;
#else
	timer.stop();
	std::cout << "Elapsed time in non-CUDA kernel=" << timer.getTime()*1000 << " msec" << std::endl;
#endif
}


