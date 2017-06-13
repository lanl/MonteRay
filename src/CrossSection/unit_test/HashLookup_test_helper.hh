#ifndef GPUDISTANCECALCULATOR_TEST_HELPER_HH_
#define GPUDISTANCECALCULATOR_TEST_HELPER_HH_

#include "MonteRayConstants.hh"

#include "HashLookup.h"

#ifdef CUDA
#include <cuda.h>
#include "driver_types.h" // cuda driver types
#endif

using namespace MonteRay;

class HashLookupTestHelper
{
public:

	HashLookupTestHelper();

	~HashLookupTestHelper();

	void setupTimers();

	void stopTimers();

	unsigned launchGetLowerBoundbyIndex( HashLookupHost* pHash, unsigned isotope, unsigned bin);

private:
	cudaEvent_t start, stop;

};

#endif /* GPUDISTANCECALCULATOR_TEST_HELPER_HH_ */


