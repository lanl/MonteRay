#ifndef GPUDISTANCECALCULATOR_TEST_HELPER_HH_
#define GPUDISTANCECALCULATOR_TEST_HELPER_HH_

#include "MonteRayConstants.hh"
#include "MonteRayDefinitions.hh"

#include "HashLookup.hh"

#ifndef __CUDACC__
#include "MonteRay_timer.hh"
#endif

using namespace MonteRay;

class HashLookupTestHelper
{
public:

	HashLookupTestHelper();

	~HashLookupTestHelper();

	void setupTimers();

	void stopTimers();

	unsigned launchGetLowerBoundbyIndex( const HashLookupHost* pHash, unsigned isotope, unsigned bin);

private:
#ifdef __CUDACC__
	cudaEvent_t start, stop;
#else
	cpuTimer timer;
#endif

};

#endif /* GPUDISTANCECALCULATOR_TEST_HELPER_HH_ */


