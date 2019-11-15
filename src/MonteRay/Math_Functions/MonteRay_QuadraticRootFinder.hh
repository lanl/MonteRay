#ifndef MONTERAYQUADRATICROOTFINDER_HH_
#define MONTERAYQUADRATICROOTFINDER_HH_

#include <limits>

#include "MonteRayTypes.hh"
#include "ThirdParty/Math.hh"

namespace MonteRay {

using Float_t = gpuRayFloat_t;

struct Roots {
    static constexpr gpuRayFloat_t inf = std::numeric_limits<gpuRayFloat_t>::infinity();

    gpuRayFloat_t R1 = inf;
    gpuRayFloat_t R2 = inf;

    CUDA_CALLABLE_MEMBER constexpr gpuRayFloat_t min(){ return Math::min(R1, R2); }
    CUDA_CALLABLE_MEMBER constexpr gpuRayFloat_t areInf(){ 
      return (R1 == inf) && (R2 == inf);
    }
};

CUDA_CALLABLE_MEMBER
gpuRayFloat_t FindMinPositiveRoot(gpuRayFloat_t, gpuRayFloat_t, gpuRayFloat_t);

CUDA_CALLABLE_MEMBER
Roots FindPositiveRoots(gpuRayFloat_t, gpuRayFloat_t, gpuRayFloat_t);


} /* namespace MonteRay */

#endif /* MONTERAYQUADRATICROOTFINDER_HH_ */
