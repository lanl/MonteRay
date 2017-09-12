#include <cuda.h>

#include <iostream>

#include "MonteRayDefinitions.hh"

#include "Material_test_helper.hh"


MaterialTestHelper::MaterialTestHelper(){}

MaterialTestHelper::~MaterialTestHelper(){}

void MaterialTestHelper::setupTimers(){
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
}

void MaterialTestHelper::stopTimers(){
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime;

	cudaEventElapsedTime(&elapsedTime, start, stop );

	std::cout << "MaterialTestHelper: Elapsed time in CUDA kernel=" << elapsedTime << " msec" << std::endl;
}


