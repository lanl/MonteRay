#ifndef MR_EXPECTEDPATHLENGTH_HH_
#define MR_EXPECTEDPATHLENGTH_HH_

#include <limits>

#include <functional>
#include <memory>

#include "MaterialProperties.hh"
#include "RayListInterface.hh"
#include "BasicTally.hh"

namespace MonteRay{

class gpuTimingHost;
class tripleTime;
class RayWorkInfo;

template <typename MaterialList>
CUDA_CALLABLE_MEMBER
gpuTallyType_t
tallyCellSegment(const MaterialList* pMatList,
        const MaterialProperties* pMatProps,
        const gpuFloatType_t* materialXS,
        gpuTallyType_t* tally,
        unsigned cell,
        gpuRayFloat_t distance,
        gpuFloatType_t energy,
        gpuFloatType_t weight,
        gpuTallyType_t opticalPathLength);

template<unsigned N, typename Geometry, typename MaterialList>
CUDA_CALLABLE_MEMBER void
tallyCollision(
        unsigned particleID,
        const Geometry* pGeometry,
        const MaterialList* pMatList,
        const MaterialProperties* pMatProps,
        const Ray_t<N>* p,
        RayWorkInfo* pRayInfo,
        gpuTallyType_t* pTally
);

template<unsigned N, typename Geometry, typename MaterialList>
CUDA_CALLABLE_KERNEL 
rayTraceTally(const Geometry* pGeometry,
        const RayList_t<N>* pCP,
        const MaterialList* pMatList,
        const MaterialProperties* pMatProps,
        RayWorkInfo* pRayInfo,
        gpuTallyType_t* tally);

template<unsigned N, typename Geometry, typename MaterialList>
MonteRay::tripleTime launchRayTraceTally(
        std::function<void (void)> cpuWork,
        int nBlocks,
        int nThreads,
        const Geometry* pGeometry,
        const RayListInterface<N>* pCP,
        const MaterialList* pMatList,
        const MaterialProperties* pMatProps,
        BasicTally* const pTally
);

} /* end namespace */

#endif /* MR_EXPECTEDPATHLENGTH_HH_ */
