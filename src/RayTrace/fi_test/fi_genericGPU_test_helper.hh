#ifndef FI_TEST_GENERICGPU_TEST_HELPER_HH_
#define FI_TEST_GENERICGPU_TEST_HELPER_HH_

#include "global.h"
#include "/projects/opt/centos7/cuda/7.5/include/driver_types.h"

#ifdef CUDA
#include <cuda.h>
#endif

#include "CollisionPoints.h"
#include "SimpleCrossSection.h"
#include "SimpleMaterialList.h"
#include "SimpleMaterialProperties.h"
#include "GridBins.h"

using namespace MonteRay;

class FIGenericGPUTestHelper
{
public:
	typedef global::float_t float_t;

	FIGenericGPUTestHelper(unsigned nCells);

	~FIGenericGPUTestHelper();

	void setupTimers();

	void stopTimers();

	void copyGridtoGPU( GridBins* );

	void launchTallyCrossSection(unsigned nBlocks, unsigned nThreads, CollisionPointsHost* pCP, SimpleCrossSectionHost* pXS );
	void launchTallyCrossSection(unsigned nBlocks, unsigned nThreads, CollisionPointsHost* pCP, SimpleMaterialListHost* pMatList, unsigned matIndex, gpuFloatType_t density );
	void launchTallyCrossSectionAtCollision(unsigned nBlocks, unsigned nThreads, CollisionPointsHost* pCP, SimpleMaterialListHost* pMatList, SimpleMaterialPropertiesHost* pMatProps );
	gpuFloatType_t getTotalXSByMatProp(SimpleMaterialProperties* matProps, SimpleMaterialList* pMatList, unsigned cell, gpuFloatType_t E);
	void launchSumCrossSectionAtCollisionLocation(unsigned nBlocks, unsigned nThreads, CollisionPointsHost* pCP, SimpleMaterialListHost* pMatList, SimpleMaterialPropertiesHost* pMatProps );
	void launchRayTraceTally(unsigned nBlocks, unsigned nThreads, CollisionPointsHost* pCP, SimpleMaterialListHost* pMatList, SimpleMaterialPropertiesHost* pMatProps );

	gpuFloatType_t getTally(unsigned i) const { return tally[i]; }

private:
	cudaEvent_t start, stop;
	unsigned nCells;

	gpuTallyType_t* tally;
	GridBins* grid_device;

};

#endif /* FI_TEST_GENERICGPU_TEST_HELPER_HH_ */

