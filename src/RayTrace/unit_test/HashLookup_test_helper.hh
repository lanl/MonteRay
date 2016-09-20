#ifndef GPUDISTANCECALCULATOR_TEST_HELPER_HH_
#define GPUDISTANCECALCULATOR_TEST_HELPER_HH_

#include "global.h"
#include "/projects/opt/centos7/cuda/7.5/include/driver_types.h"

#include "HashLookup.h"

#ifdef CUDA
#include <cuda.h>
#endif

using namespace MonteRay;

class HashLookupTestHelper
{
public:
	typedef global::float_t float_t;

	HashLookupTestHelper();

	~HashLookupTestHelper();

	void setupTimers();

	void stopTimers();

	unsigned launchGetLowerBoundbyIndex( HashLookupHost* pHash, unsigned isotope, unsigned bin);

private:
	cudaEvent_t start, stop;

};

#endif /* GPUDISTANCECALCULATOR_TEST_HELPER_HH_ */


