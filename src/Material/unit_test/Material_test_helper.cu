#include <iostream>

#include "MonteRayDefinitions.hh"

#include "Material_test_helper.hh"


MaterialTestHelper::MaterialTestHelper(){}

MaterialTestHelper::~MaterialTestHelper(){}

void MaterialTestHelper::setupTimers(){
#ifdef __CUDACC__
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif
}

void MaterialTestHelper::stopTimers(){
	float elapsedTime;

#ifdef __CUDACC__
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop );
#endif

	std::cout << "MaterialTestHelper: Elapsed time in CUDA kernel=" << elapsedTime << " msec" << std::endl;
}


