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
Float_t
FindMinPositiveRoot(Float_t A, Float_t B, Float_t C) {

    /// For doubles, NearEpsilon ~= 1.0e-14
#ifndef __CUDA_ARCH__
    const Float_t NearEpsilon = 100.0 * std::numeric_limits<Float_t>::epsilon();
    Float_t minposRoot = std::numeric_limits<Float_t>::infinity();
#else
#if RAY_DOUBLEPRECISION < 1
    const Float_t NearEpsilon = 100.0 * FLT_EPSILON;
    Float_t minposRoot = CUDART_INF_F;
#else
    const Float_t NearEpsilon = 100.0 * DBL_EPSILON;
    Float_t minposRoot = CUDART_INF;
#endif
#endif

    Float_t Discriminant = B*B - 4.0 * A * C;

    // The roots (e.g. intersection distances for ray tracing) are separated by the
    // square root of the discriminant.  If this separation is very small, it is
    // best to ignore the intersection since the enter/exit points are so close that
    // they become numerically difficult to differentiate, AND any tally contribution
    // would be negligible.
    if (Discriminant < NearEpsilon )
        return minposRoot;

    Float_t temp = (B < 0.0) ? -0.5*( B - sqrt(Discriminant) ) :
                               -0.5*( B + sqrt(Discriminant) );
    Float_t root1 = temp/A;
    Float_t root2 = C/temp;

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
Roots
FindPositiveRoots(Float_t A, Float_t B, Float_t C) {
    //constexpr Float_t inf = std::numeric_limits<double>::infinity();
    /// For doubles, NearEpsilon ~= 1.0e-14
#ifndef __CUDA_ARCH__
    //const Float_t NearEpsilon = 100.0 * std::numeric_limits<Float_t>::epsilon();
#else
#if RAY_DOUBLEPRECISION < 1
    //const Float_t NearEpsilon = 100.0 * FLT_EPSILON;
    Float_t minposRoot = CUDART_INF_F;
#else
    //const Float_t NearEpsilon = 100.0 * DBL_EPSILON;
    Float_t minposRoot = CUDART_INF;
#endif
#endif


    //	std::vector<double> roots(2,inf);
    Roots roots;


    Float_t Discriminant = B*B - 4.0 * A * C;

    // The roots (e.g. intersection distances for ray tracing) are separated by the
    // square root of the discriminant.  If this separation is very small, it is
    // best to ignore the intersection since the enter/exit points are so close that
    // they become numerically difficult to differentiate, AND any tally contribution
    // would be negligible.

    //if( Discriminant < NearEpsilon ){
    if( Discriminant < 0.0 ){
        return roots;
    }

    Float_t temp = (B < 0.0) ? -0.5*( B - sqrt(Discriminant) ) :
            -0.5*( B + sqrt(Discriminant) );
    Float_t root1 = temp/A;
    Float_t root2 = C/temp;

    //if( root1 > NearEpsilon )
    if( root1 > 0.0 )
        roots.R1 = root1;

    //if( root2 > NearEpsilon )
    if( root2 > 0.0 )
        roots.R2 = root2;

    return roots;
}


} /* namespace MonteRay */
