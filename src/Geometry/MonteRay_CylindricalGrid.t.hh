#ifndef MONTERAY_CYLINDRICALGRID_T_HH_
#define MONTERAY_CYLINDRICALGRID_T_HH_

#include "MonteRay_CylindricalGrid.hh"
#include "MonteRay_GridSystemInterface.t.hh"

namespace MonteRay {

template<bool OUTWARD>
CUDA_CALLABLE_MEMBER
void
MonteRay_CylindricalGrid::radialCrossingDistancesSingleDirection( singleDimRayTraceMap_t& rayTraceMap, const Position_t& pos, const Direction_t& dir, gpuRayFloat_t distance) const {
    // helper function to wrap generalized radialCrossingDistancesSingleDirection
    gpuRayFloat_t particleRSq = calcParticleRSq( pos );
    unsigned rIndex = pRVertices->getRadialIndexFromRSq(particleRSq);

    gpuRayFloat_t A = calcQuadraticA( dir );
    gpuRayFloat_t B = calcQuadraticB( pos, dir);

    // ray-trace
    radialCrossingDistanceSingleDirection<OUTWARD>(rayTraceMap, *pRVertices, particleRSq, A, B, distance, rIndex);
}

}

#endif /* MONTERAY_CYLINDRICALGRID_T_HH_ */
