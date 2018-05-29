#ifndef FI_TEST_GENERICGPU_TEST_HELPER_HH_
#define FI_TEST_GENERICGPU_TEST_HELPER_HH_

#include "MonteRayDefinitions.hh"
#include "MonteRayConstants.hh"

#include "RayListInterface.hh"
#include "MonteRayCrossSection.hh"
#include "MonteRayMaterialList.hh"
#include "MonteRay_MaterialProperties.hh"
#include "GridBins.hh"

#ifndef __CUDACC__
#include "MonteRay_timer.hh"
#endif

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
#ifdef __CUDACC__
	cudaEvent_t start, stop;
#else
	cpuTimer timer;
#endif

};

#endif /* FI_TEST_GENERICGPU_TEST_HELPER_HH_ */

