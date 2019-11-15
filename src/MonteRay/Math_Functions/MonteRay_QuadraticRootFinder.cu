#include "MonteRay_QuadraticRootFinder.hh"

#include <cmath>

#ifdef __CUDACC__
#include <float.h>
#include <math_constants.h>
#endif

#include "MonteRayDefinitions.hh"

namespace MonteRay {

///\brief Quadratic equation solver.
///
///\details We used Numerical Recipes formula from the "Quadratic and Cubic Equations" Section to \n
/// solve this quadratic.
CUDA_CALLABLE_MEMBER
gpuRayFloat_t
FindMinPositiveRoot(gpuRayFloat_t A, gpuRayFloat_t B, gpuRayFloat_t C) {

    /// For doubles, NearEpsilon ~= 1.0e-14
    const gpuRayFloat_t NearEpsilon = 100.0 * std::numeric_limits<gpuRayFloat_t>::epsilon();
    const gpuRayFloat_t minposRoot = std::numeric_limits<gpuRayFloat_t>::infinity();

    gpuRayFloat_t Discriminant = B*B - 4.0 * A * C;

    // The roots (e.g. intersection distances for ray tracing) are separated by the
    // square root of the discriminant.  If this separation is very small, it is
    // best to ignore the intersection since the enter/exit points are so close that
    // they become numerically difficult to differentiate, AND any tally contribution
    // would be negligible.
    if (Discriminant < NearEpsilon )
        return minposRoot;

    gpuRayFloat_t temp = (B < 0.0) ? -0.5*( B - sqrt(Discriminant) ) :
                               -0.5*( B + sqrt(Discriminant) );
    gpuRayFloat_t root1 = temp/A;
    gpuRayFloat_t root2 = C/temp;

    if( root1 > NearEpsilon )
        minposRoot = root1;

    if( root2 > NearEpsilon && root2 < minposRoot )
        minposRoot = root2;

    return minposRoot;
}

///\brief Quadratic equation solver.
///
///\details We used Numerical Recipes formula from the "Quadratic and Cubic Equations" Section to \n
/// solve this quadratic.
CUDA_CALLABLE_MEMBER
Roots FindPositiveRoots(gpuRayFloat_t A, gpuRayFloat_t B, gpuRayFloat_t C) {
    gpuRayFloat_t Discriminant = B*B - 4.0 * A * C;

    // The roots (e.g. intersection distances for ray tracing) are separated by the
    // square root of the discriminant.  If this separation is very small, it is
    // best to ignore the intersection since the enter/exit points are so close that
    // they become numerically difficult to differentiate, AND any tally contribution
    // would be negligible.

    if( Discriminant < 0.0 ){
        return Roots{};
    }

    gpuRayFloat_t temp = (B < 0.0) ? 
        -0.5*( B - sqrt(Discriminant) ) :
        -0.5*( B + sqrt(Discriminant) );
    gpuRayFloat_t root1 = temp/A;
    gpuRayFloat_t root2 = C/temp;

    if (root1 <= 0.0){
      root1 = Roots::inf;
    }
    if (root2 <= 0.0){
      root2 = Roots::inf;
    }
    return Roots{root1, root2};
}


} /* namespace MonteRay */
