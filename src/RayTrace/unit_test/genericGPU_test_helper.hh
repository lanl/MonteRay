#ifndef UNIT_TEST_GENERICGPU_TEST_HELPER_HH_
#define UNIT_TEST_GENERICGPU_TEST_HELPER_HH_

#include "MonteRayConstants.hh"
#include "driver_types.h" // cuda driver types

#ifdef CUDA
#include <cuda.h>
#endif

using namespace MonteRay;

class GenericGPUTestHelper
{
public:

	GenericGPUTestHelper();

	~GenericGPUTestHelper();

	void setupTimers();

	void stopTimers();

private:
	cudaEvent_t start, stop;

};
#endif /* UNIT_TEST_GENERICGPU_TEST_HELPER_HH_ */


