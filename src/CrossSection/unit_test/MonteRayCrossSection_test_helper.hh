#ifndef GPUDISTANCECALCULATOR_TEST_HELPER_HH_
#define GPUDISTANCECALCULATOR_TEST_HELPER_HH_

#include "MonteRayConstants.hh"

#include "MonteRayCrossSection.hh"

#ifdef CUDA
#include <cuda.h>
#include "driver_types.h" // cuda driver types
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
	cudaEvent_t start, stop;

};

#endif /* GPUDISTANCECALCULATOR_TEST_HELPER_HH_ */


