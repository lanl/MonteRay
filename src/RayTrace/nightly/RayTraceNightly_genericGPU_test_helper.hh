#ifndef FI_TEST_GENERICGPU_TEST_HELPER_HH_
#define FI_TEST_GENERICGPU_TEST_HELPER_HH_

#include "MonteRayConstants.hh"

#ifdef __CUDACC__
#include <cuda.h>
#include "driver_types.h" // cuda driver types
#endif

#include "RayListInterface.hh"
#include "MonteRayCrossSection.hh"
#include "MonteRayMaterialList.hh"
#include "MonteRay_MaterialProperties.hh"
#include "GridBins.h"

using namespace MonteRay;

template<unsigned N>
class FIGenericGPUTestHelper
{
public:

	FIGenericGPUTestHelper(unsigned nCells);

	~FIGenericGPUTestHelper();

	void setupTimers();

	void stopTimers();

private:
	cudaEvent_t start, stop;

};

#endif /* FI_TEST_GENERICGPU_TEST_HELPER_HH_ */

