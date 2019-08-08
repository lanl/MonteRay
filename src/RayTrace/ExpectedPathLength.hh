#ifndef MR_EXPECTEDPATHLENGTH_HH_
#define MR_EXPECTEDPATHLENGTH_HH_

#include <limits>

#include <functional>
#include <memory>

#include "RayListInterface.hh"

namespace MonteRay{

class gpuTimingHost;
class MonteRay_MaterialProperties;
class MonteRay_MaterialProperties_Data;
class MonteRayMaterialListHost;
class HashLookup;
class gpuTallyHost;
class tripleTime;
class RayWorkInfo;

template <typename MaterialList>
CUDA_CALLABLE_MEMBER
gpuTallyType_t
tallyCellSegment(const MaterialList* pMatList,
        const MonteRay_MaterialProperties_Data* pMatProps,
        const gpuFloatType_t* materialXS,
        gpuTallyType_t* tally,
        unsigned cell,
        gpuRayFloat_t distance,
        gpuFloatType_t energy,
        gpuFloatType_t weight,
        gpuTallyType_t opticalPathLength);

template<unsigned N, typename GRIDTYPE, typename MaterialList>
CUDA_CALLABLE_MEMBER void
tallyCollision(
        unsigned particleID,
        const GRIDTYPE* pGrid,
        const MaterialList* pMatList,
        const MonteRay_MaterialProperties_Data* pMatProps,
        const HashLookup* pHash,
        const Ray_t<N>* p,
        RayWorkInfo* pRayInfo,
        gpuTallyType_t* pTally
);

template<unsigned N, typename GRIDTYPE, typename MaterialList>
CUDA_CALLABLE_KERNEL 
rayTraceTally(const GRIDTYPE* pGrid,
        const RayList_t<N>* pCP,
        const MaterialList* pMatList,
        const MonteRay_MaterialProperties_Data* pMatProps,
        const HashLookup* pHash,
        RayWorkInfo* pRayInfo,
        gpuTallyType_t* tally);

template<unsigned N, typename GRIDTYPE, typename MaterialList>
MonteRay::tripleTime launchRayTraceTally(
        std::function<void (void)> cpuWork,
        int nBlocks,
        int nThreads,
        const GRIDTYPE* pGrid,
        const RayListInterface<N>* pCP,
        const MaterialList* pMatList,
        const MonteRay_MaterialProperties* pMatProps,
        gpuTallyHost* pTally
);

} /* end namespace */

#endif /* MR_EXPECTEDPATHLENGTH_HH_ */
