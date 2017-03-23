#ifndef GPUDISTANCECALCULATOR_TEST_HELPER_HH_
#define GPUDISTANCECALCULATOR_TEST_HELPER_HH_

#include "MonteRayConstants.hh"

#include "SimpleCrossSection.h"

#ifdef CUDA
#include <cuda.h>
#include "driver_types.h" // cuda driver types
#endif

using namespace MonteRay;

class SimpleCrossSectionTestHelper
{
public:

	SimpleCrossSectionTestHelper();

	~SimpleCrossSectionTestHelper();

	void setupTimers();

	void stopTimers();

	//float_t launchGetTotalXS( SimpleCrossSectionHost* pXS, float_t energy);

private:
	cudaEvent_t start, stop;

};

#endif /* GPUDISTANCECALCULATOR_TEST_HELPER_HH_ */


