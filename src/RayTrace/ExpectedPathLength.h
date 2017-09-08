#ifndef EXPECTEDPATHLENGTH_H_
#define EXPECTEDPATHLENGTH_H_

#include "RayListInterface.hh"
#include "SimpleMaterialList.h"
#include "MonteRay_MaterialProperties.hh"
#include "HashLookup.h"
#include "gpuRayTrace.h"
#include "gpuTally.h"

#include <functional>
#include <MonteRay_timer.hh>

#define MAXNUMMATERIALS 10

namespace MonteRay{

class gpuTimingHost;

#ifdef __CUDACC__
template<unsigned N>
__device__
void
tallyCollision(GridBins* pGrid, SimpleMaterialList* pMatList, MonteRay_MaterialProperties_Data* pMatProps, HashLookup* pHash, Ray_t<N>* p, gpuTallyType_t* tally);

template<unsigned N>
__device__
void
tallyCollision(GridBins* pGrid, SimpleMaterialList* pMatList, MonteRay_MaterialProperties_Data* pMatProps, HashLookup* pHash, Ray_t<N>* p, gpuTally* tally, unsigned tid);

__device__
gpuTallyType_t
tallyCellSegment(SimpleMaterialList* pMatList, MonteRay_MaterialProperties_Data* pMatProps, gpuFloatType_t* materialXS, gpuTallyType_t* tally, unsigned cell, gpuFloatType_t distance, gpuFloatType_t energy, gpuFloatType_t weight, gpuTallyType_t opticalPathLength);

__device__
gpuTallyType_t
tallyCellSegment(SimpleMaterialList* pMatList, MonteRay_MaterialProperties_Data* pMatProps, gpuFloatType_t* materialXS, struct gpuTally* pTally, unsigned cell, gpuFloatType_t distance, gpuFloatType_t energy, gpuFloatType_t weight, gpuTallyType_t opticalPathLength);

template<unsigned N>
__device__
gpuTallyType_t
tallyAttenuation(GridBins* pGrid, SimpleMaterialList* pMatList, MonteRay_MaterialProperties_Data* pMatProps, HashLookup* pHash, Ray_t<N>* p);


__device__
gpuTallyType_t
attenuateRayTraceOnly(SimpleMaterialList* pMatList, MonteRay_MaterialProperties_Data* pMatProps, HashLookup* pHash, unsigned HashBin, unsigned cell, gpuFloatType_t distance, gpuFloatType_t energy, gpuTallyType_t enteringFraction );

template<unsigned N>__global__ void
rayTraceTally(GridBins* pGrid, RayList_t<N>* pCP, SimpleMaterialList* pMatList, MonteRay_MaterialProperties_Data* pMatProps, HashLookup* pHash, gpuTallyType_t* tally);

template<unsigned N> __global__ void
rayTraceTally(GridBins* pGrid, RayList_t<N>* pCP, SimpleMaterialList* pMatList, MonteRay_MaterialProperties_Data* pMatProps, HashLookup* pHash, gpuTally* tally);

#endif

template<unsigned N>
MonteRay::tripleTime launchRayTraceTally(
		                 std::function<void (void)> cpuWork,
                         unsigned nBlocks,
		                 unsigned nThreads,
		                 GridBinsHost* grid,
		                 RayListInterface<N>* pCP,
		                 SimpleMaterialListHost* pMatList,
		                 MonteRay_MaterialProperties* pMatProps,
		                 gpuTallyHost* pTally
		                );

}

#endif /* EXPECTEDPATHLENGTH_H_ */
