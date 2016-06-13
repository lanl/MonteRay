#ifndef GPUDISTANCECALCULATOR_TEST_HELPER_HH_
#define GPUDISTANCECALCULATOR_TEST_HELPER_HH_

#include "global.h"
#include "/projects/opt/centos7/cuda/7.5/include/driver_types.h"

#include "SimpleCrossSection.h"

#ifdef CUDA
#include <cuda.h>
#endif

class SimpleCrossSectionTestHelper
{
public:
	typedef global::float_t float_t;

	SimpleCrossSectionTestHelper();

	~SimpleCrossSectionTestHelper();

	void setupTimers();

	void stopTimers();

	float_t launchGetTotalXS( SimpleCrossSectionHost* pXS, float_t energy);

private:
	cudaEvent_t start, stop;

};

#endif /* GPUDISTANCECALCULATOR_TEST_HELPER_HH_ */


