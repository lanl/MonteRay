#ifndef GPUDISTANCECALCULATOR_TEST_HELPER_HH_
#define GPUDISTANCECALCULATOR_TEST_HELPER_HH_

#include "MonteRayDefinitions.hh"
#include "MonteRayConstants.hh"

#include "MonteRayCrossSection.hh"

#ifndef __CUDACC__
#include "MonteRay_timer.hh"
#endif

using namespace MonteRay;

class MonteRayCrossSectionTestHelper
{
public:

	MonteRayCrossSectionTestHelper();

	~MonteRayCrossSectionTestHelper();

	void setupTimers();

	void stopTimers();

	//float_t launchGetTotalXS( MonteRayCrossSectionHost* pXS, float_t energy);

private:
#ifdef __CUDACC__
	cudaEvent_t start, stop;
#else
	cpuTimer timer;
#endif

};

#endif /* GPUDISTANCECALCULATOR_TEST_HELPER_HH_ */


