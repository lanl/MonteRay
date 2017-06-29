#ifndef UNIT_TEST_GENERICGPU_TEST_HELPER_HH_
#define UNIT_TEST_GENERICGPU_TEST_HELPER_HH_

#include "MonteRayDefinitions.hh"

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


