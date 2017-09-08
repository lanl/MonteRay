#ifndef FI_TEST_GENERICGPU_TEST_HELPER_HH_
#define FI_TEST_GENERICGPU_TEST_HELPER_HH_

#include "MonteRayConstants.hh"

#ifdef __CUDACC__
#include <cuda.h>
#include "driver_types.h" // cuda driver types
#endif

#include "RayListInterface.hh"
#include "MonteRayCrossSection.hh"
#include "SimpleMaterialList.h"
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

	void copyGridtoGPU( GridBins* );

	void launchTallyCrossSection(unsigned nBlocks, unsigned nThreads, RayListInterface<N>* pCP, MonteRayCrossSectionHost* pXS );
	void launchTallyCrossSection(unsigned nBlocks, unsigned nThreads, RayListInterface<N>* pCP, SimpleMaterialListHost* pMatList, unsigned matIndex, gpuFloatType_t density );
	void launchTallyCrossSectionAtCollision(unsigned nBlocks, unsigned nThreads, RayListInterface<N>* pCP, SimpleMaterialListHost* pMatList, MonteRay_MaterialProperties* pMatProps );

	gpuFloatType_t getTotalXSByMatProp(MonteRay_MaterialProperties* matProps, SimpleMaterialList* pMatList, HashLookup* pHash, unsigned HashBin, unsigned cell, gpuFloatType_t E);
	gpuFloatType_t getTotalXSByMatProp(MonteRay_MaterialProperties* matProps, SimpleMaterialList* pMatList, unsigned cell, gpuFloatType_t E);

	void launchSumCrossSectionAtCollisionLocation(unsigned nBlocks, unsigned nThreads, RayListInterface<N>* pCP, SimpleMaterialListHost* pMatList, MonteRay_MaterialProperties* pMatProps );
	void launchRayTraceTally(unsigned nBlocks, unsigned nThreads, RayListInterface<N>* pCP, SimpleMaterialListHost* pMatList, MonteRay_MaterialProperties* pMatProps );

	gpuFloatType_t getTally(unsigned i) const { return tally[i]; }

private:
	cudaEvent_t start, stop;
	unsigned nCells;

	gpuTallyType_t* tally;
	GridBins* grid_device;

};

#endif /* FI_TEST_GENERICGPU_TEST_HELPER_HH_ */

