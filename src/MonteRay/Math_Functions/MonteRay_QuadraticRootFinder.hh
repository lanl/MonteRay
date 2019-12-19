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
    CUDA_CALLABLE_MEMBER constexpr gpuRayFloat_t max(){ return Math::max(R1, R2); }
    CUDA_CALLABLE_MEMBER constexpr gpuRayFloat_t areInf(){ 
      return (R1 == inf) && (R2 == inf);
    }
};

///\brief Quadratic equation solver.
///
///\details We used Numerical Recipes formula from the "Quadratic and Cubic Equations" Section to \n
/// solve this quadratic.
CUDA_CALLABLE_MEMBER 
constexpr Roots FindRoots(gpuRayFloat_t A, gpuRayFloat_t B, gpuRayFloat_t C) {
    gpuRayFloat_t Discriminant = B*B - 4.0 * A * C;

    // The roots (e.g. intersection distances for ray tracing) are separated by the
    // square root of the discriminant.  If this separation is very small, it is
    // best to ignore the intersection since the enter/exit points are so close that
    // they become numerically difficult to differentiate, AND any tally contribution
    // would be negligible.

    if( Discriminant < 0.0 ){
        return Roots{};
    }

    // TPB: this used to be just sqrt, which was double precision on the GPU probably
    gpuRayFloat_t temp = (B < 0.0) ? 
        -0.5*( B - Math::sqrt(Discriminant) ) : 
        -0.5*( B + Math::sqrt(Discriminant) );
    return {temp/A, C/temp};
}

CUDA_CALLABLE_MEMBER 
constexpr auto FindMaxValidRoot(gpuRayFloat_t A, gpuRayFloat_t B, gpuRayFloat_t C) {
  auto roots = FindRoots(A, B, C);
  if (roots.R1 <= 0.0){
    roots.R1 = -Roots::inf;
  }
  if (roots.R2 <= 0.0){
    roots.R2 = -Roots::inf;
  }
  return Math::max(static_cast<gpuRayFloat_t>(0.0), roots.max());
}

CUDA_CALLABLE_MEMBER 
constexpr Roots FindPositiveRoots(gpuRayFloat_t A, gpuRayFloat_t B, gpuRayFloat_t C) {
  auto roots = FindRoots(A, B, C);
  if (roots.R1 <= 0.0){
    roots.R1 = Roots::inf;
  }
  if (roots.R2 <= 0.0){
    roots.R2 = Roots::inf;
  }
  return roots;
}

} /* namespace MonteRay */

#endif /* MONTERAYQUADRATICROOTFINDER_HH_ */
