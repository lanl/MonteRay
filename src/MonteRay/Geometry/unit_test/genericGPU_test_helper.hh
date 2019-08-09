#ifndef UNIT_TEST_GENERICGPU_TEST_HELPER_HH_
#define UNIT_TEST_GENERICGPU_TEST_HELPER_HH_

#include "MonteRayDefinitions.hh"
#include "MonteRay_timer.hh"

using namespace MonteRay;

class GenericGPUTestHelper
{
public:

	GenericGPUTestHelper();

	~GenericGPUTestHelper();

	void setupTimers();

	void stopTimers();

private:
#ifdef __CUDACC__
	cudaEvent_t start, stop;
#else
	cpuTimer timer;
#endif

};
#endif /* UNIT_TEST_GENERICGPU_TEST_HELPER_HH_ */


