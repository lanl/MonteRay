#ifndef MATERIAL_UNITTEST_HELPER_HH_
#define MATERIAL_UNITTEST_HELPER_HH_

#include "MonteRayConstants.hh"
#include "driver_types.h" // cuda driver types

#ifdef CUDA
#include <cuda.h>
#endif

using namespace MonteRay;

class MaterialTestHelper
{
public:

	MaterialTestHelper();

	~MaterialTestHelper();

	void setupTimers();

	void stopTimers();

private:
	cudaEvent_t start, stop;

};
#endif /* MATERIAL_UNITTEST_HELPER_HH_ */


