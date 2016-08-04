#ifndef UNIT_TEST_GPUTIMING_TEST_HELPER_HH_
#define UNIT_TEST_GPUTIMING_TEST_HELPER_HH_

#include "gpuTally.h"

class GPUTallyTestHelper
{
public:

	GPUTallyTestHelper();

	~GPUTallyTestHelper();

	void launchAddTally( MonteRay::gpuTallyHost* tally, unsigned i, float_t a, float_t b );

};
#endif /* UNIT_TEST_GPUTIMING_TEST_HELPER_HH_ */


