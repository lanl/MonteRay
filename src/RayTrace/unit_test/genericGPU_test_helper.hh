#ifndef UNIT_TEST_GENERICGPU_TEST_HELPER_HH_
#define UNIT_TEST_GENERICGPU_TEST_HELPER_HH_

#include "global.h"
#include "/projects/opt/centos7/cuda/7.5/include/driver_types.h"

#ifdef CUDA
#include <cuda.h>
#endif

class GenericGPUTestHelper
{
public:
	typedef global::float_t float_t;

	GenericGPUTestHelper();

	~GenericGPUTestHelper();

	void setupTimers();

	void stopTimers();

private:
	cudaEvent_t start, stop;

};
#endif /* UNIT_TEST_GENERICGPU_TEST_HELPER_HH_ */


