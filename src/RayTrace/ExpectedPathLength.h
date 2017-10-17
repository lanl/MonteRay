#ifndef EXPECTEDPATHLENGTH_H_
#define EXPECTEDPATHLENGTH_H_

#include "RayListInterface.hh"
#include "MonteRayMaterialList.hh"
#include "MonteRay_MaterialProperties.hh"
#include "HashLookup.h"
#include "gpuRayTrace.h"
#include "gpuTally.h"

#include <functional>
#include <MonteRay_timer.hh>

#define MAXNUMMATERIALS 10

namespace MonteRay{

class gpuTimingHost;

template<unsigned N>
CUDA_CALLABLE_MEMBER void
tallyCollision(GridBins* pGrid, MonteRayMaterialList* pMatList,
		       MonteRay_MaterialProperties_Data* pMatProps, const HashLookup* pHash, Ray_t<N>* p,
		       gpuTallyType_t* pTally, unsigned tid = 0);

CUDA_CALLABLE_MEMBER
gpuTallyType_t
tallyCellSegment(MonteRayMaterialList* pMatList, MonteRay_MaterialProperties_Data* pMatProps,
		gpuFloatType_t* materialXS, gpuTallyType_t* tally, unsigned cell, gpuFloatType_t distance,
		gpuFloatType_t energy, gpuFloatType_t weight, gpuTallyType_t opticalPathLength);

template<unsigned N>
CUDA_CALLABLE_MEMBER
gpuTallyType_t
tallyAttenuation(GridBins* pGrid,
			     MonteRayMaterialList* pMatList,
			     MonteRay_MaterialProperties_Data* pMatProps,
			     const HashLookup* pHash,
			     Ray_t<N>* p);

CUDA_CALLABLE_MEMBER
gpuTallyType_t
attenuateRayTraceOnly(const MonteRayMaterialList* pMatList,
					  const MonteRay_MaterialProperties_Data* pMatProps,
					  const HashLookup* pHash,
					  unsigned HashBin,
					  unsigned cell,
					  gpuFloatType_t distance,
					  gpuFloatType_t energy,
					  gpuTallyType_t enteringFraction,
					  ParticleType_t particleType );

template<unsigned N>
CUDA_CALLABLE_KERNEL
void
rayTraceTally(GridBins* pGrid,
			  RayList_t<N>* pCP,
			  MonteRayMaterialList* pMatList,
			  MonteRay_MaterialProperties_Data* pMatProps,
			  const HashLookup* pHash,
			  gpuTallyType_t* tally);

template<unsigned N>
MonteRay::tripleTime launchRayTraceTally(
		                 std::function<void (void)> cpuWork,
                         unsigned nBlocks,
		                 unsigned nThreads,
		                 GridBinsHost* grid,
		                 RayListInterface<N>* pCP,
		                 MonteRayMaterialListHost* pMatList,
		                 MonteRay_MaterialProperties* pMatProps,
		                 gpuTallyHost* pTally
		                );

}

#endif /* EXPECTEDPATHLENGTH_H_ */
