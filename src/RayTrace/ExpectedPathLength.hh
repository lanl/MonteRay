#ifndef EXPECTEDPATHLENGTH_H_
#define EXPECTEDPATHLENGTH_H_

#include "RayListInterface.hh"
#include "MonteRayMaterialList.hh"
#include "MonteRay_MaterialProperties.hh"
#include "HashLookup.h"
#include "gpuRayTrace.hh"
#include "gpuTally.hh"

#include <functional>
#include <MonteRay_timer.hh>

#define MAXNUMMATERIALS 10

namespace MonteRay{

class gpuTimingHost;

template<unsigned N>
CUDA_CALLABLE_MEMBER void
tallyCollision(const GridBins* pGrid,
		       const MonteRayMaterialList* pMatList,
		       const MonteRay_MaterialProperties_Data* pMatProps,
		       const HashLookup* pHash,
		       const Ray_t<N>* p,
		       gpuTallyType_t* pTally,
		       unsigned tid = 0);

CUDA_CALLABLE_MEMBER
gpuTallyType_t
tallyCellSegment(const MonteRayMaterialList* pMatList,
		         const MonteRay_MaterialProperties_Data* pMatProps,
		         const gpuFloatType_t* materialXS,
		               gpuTallyType_t* tally,
		               unsigned cell,
		               gpuRayFloat_t distance,
	 	               gpuFloatType_t energy,
	 	               gpuFloatType_t weight,
	 	               gpuTallyType_t opticalPathLength);

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
rayTraceTally(const GridBins* pGrid,
			  const RayList_t<N>* pCP,
			  const MonteRayMaterialList* pMatList,
			  const MonteRay_MaterialProperties_Data* pMatProps,
			  const HashLookup* pHash,
			  gpuTallyType_t* tally);

template<unsigned N>
MonteRay::tripleTime launchRayTraceTally(
		                 std::function<void (void)> cpuWork,
                         unsigned nBlocks,
		                 unsigned nThreads,
		                 const GridBinsHost* grid,
		                 const RayListInterface<N>* pCP,
		                 const MonteRayMaterialListHost* pMatList,
		                 const MonteRay_MaterialProperties* pMatProps,
		                 gpuTallyHost* pTally
		                );

}

#endif /* EXPECTEDPATHLENGTH_H_ */
