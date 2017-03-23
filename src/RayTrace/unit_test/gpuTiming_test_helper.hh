#ifndef UNIT_TEST_GPUTIMING_TEST_HELPER_HH_
#define UNIT_TEST_GPUTIMING_TEST_HELPER_HH_

#include "MonteRayDefinitions.hh"

namespace MonteRay{
class gpuTimingHost;
}

class GPUTimingTestHelper
{
public:

	GPUTimingTestHelper();

	~GPUTimingTestHelper();

	void launchGPUSleep( MonteRay::clock64_t nCycles, MonteRay::gpuTimingHost* );
	double launchGPUStreamSleep(unsigned nBlocks, unsigned nThreads, MonteRay::clock64_t nCycles, unsigned milliseconds);

};

#endif /* UNIT_TEST_GPUTIMING_TEST_HELPER_HH_ */


