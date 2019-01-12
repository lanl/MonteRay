#ifndef MONTERAY_SPHERICALGRID_T_HH_
#define MONTERAY_SPHERICALGRID_T_HH_

#include "MonteRay_SphericalGrid.hh"
#include "MonteRay_GridSystemInterface.t.hh"
#include "RayWorkInfo.hh"

namespace MonteRay {

template<bool OUTWARD>
CUDA_CALLABLE_MEMBER
void
MonteRay_SphericalGrid::radialCrossingDistancesSingleDirection(
        const unsigned dim,
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        const Position_t& pos,
        const Direction_t& dir,
        const gpuRayFloat_t distance) const {
    // helper function to wrap generalized radialCrossingDistancesSingleDirection
    gpuRayFloat_t particleRSq = calcParticleRSq( pos );
    unsigned rIndex = pRVertices->getRadialIndexFromRSq(particleRSq);

    gpuRayFloat_t A = calcQuadraticA( dir );
    gpuRayFloat_t B = calcQuadraticB( pos, dir);

    // ray-trace
    radialCrossingDistanceSingleDirection<OUTWARD>(
            dim,
            threadID,
            rayInfo,
            *pRVertices,
            particleRSq,
            A,
            B,
            distance,
            rIndex);
}

}

#endif /* MONTERAY_SPHERICALGRID_T_HH_ */
