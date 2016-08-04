#ifndef EXPECTEDPATHLENGTH_H_
#define EXPECTEDPATHLENGTH_H_

#include "CollisionPoints.h"
#include "SimpleMaterialList.h"
#include "SimpleMaterialProperties.h"
#include "gpuRayTrace.h"
#include "gpuTally.h"

#include <functional>
#include <cpuTimer.h>

namespace MonteRay{

class gpuTimingHost;

#ifdef CUDA
__device__
void
tallyCollision(GridBins* pGrid, SimpleMaterialList* pMatList, SimpleMaterialProperties* pMatProps, gpuParticle_t* p, gpuTallyType_t* tally);

__device__
void
tallyCollision(GridBins* pGrid, SimpleMaterialList* pMatList, SimpleMaterialProperties* pMatProps, gpuParticle_t* p, gpuTally* tally, unsigned tid);

__device__
gpuFloatType_t
tallyCellSegment(SimpleMaterialList* pMatList, SimpleMaterialProperties* pMatProps, gpuTallyType_t* tally, unsigned cell, gpuFloatType_t distance, gpuFloatType_t energy, gpuFloatType_t enteringFraction);

__device__
double
tallyCellSegment(SimpleMaterialList* pMatList, SimpleMaterialProperties* pMatProps, gpuTally* tally, unsigned cell, gpuFloatType_t distance, gpuFloatType_t energy, double enteringFraction);


__device__
gpuFloatType_t
tallyAttenuation(GridBins* pGrid, SimpleMaterialList* pMatList, SimpleMaterialProperties* pMatProps, gpuParticle_t* p);

__device__
gpuFloatType_t
attenuateRayTraceOnly(SimpleMaterialList* pMatList, SimpleMaterialProperties* pMatProps, unsigned cell, gpuFloatType_t distance, gpuFloatType_t energy, gpuFloatType_t enteringFraction );

__global__
void
rayTraceTally(GridBins* pGrid, CollisionPoints* pCP, SimpleMaterialList* pMatList, SimpleMaterialProperties* pMatProps, gpuTallyType_t* tally);

__global__
void
rayTraceTally(GridBins* pGrid, CollisionPoints* pCP, SimpleMaterialList* pMatList, SimpleMaterialProperties* pMatProps, gpuTally* tally);

#endif


MonteRay::tripleTime launchRayTraceTally(
		                 std::function<void (void)> cpuWork,
                         unsigned nBlocks,
		                 unsigned nThreads,
		                 GridBinsHost* grid,
		                 CollisionPointsHost* pCP,
		                 SimpleMaterialListHost* pMatList,
		                 SimpleMaterialPropertiesHost* pMatProps,
		                 gpuTallyHost* pTally
		                );

}

#endif /* EXPECTEDPATHLENGTH_H_ */
