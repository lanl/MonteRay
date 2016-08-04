#ifndef UNIT_TEST_GPUTIMING_TEST_HELPER_HH_
#define UNIT_TEST_GPUTIMING_TEST_HELPER_HH_

#include "gpuTiming.h"

using namespace MonteRay;

class GPUTimingTestHelper
{
public:
	GPUTimingTestHelper();

	~GPUTimingTestHelper();

	void launchGPUSleep( clock64_t nCycles, gpuTimingHost* );
	double launchGPUStreamSleep(unsigned nBlocks, unsigned nThreads, clock64_t nCycles, unsigned milliseconds);

};

#endif /* UNIT_TEST_GPUTIMING_TEST_HELPER_HH_ */


