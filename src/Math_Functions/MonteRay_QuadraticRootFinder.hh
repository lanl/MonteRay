/*
 * MonteRayQuadraticRootFinder.hh
 *
 *  Created on: Feb 2, 2018
 *      Author: jsweezy
 */

#ifndef MONTERAYQUADRATICROOTFINDER_HH_
#define MONTERAYQUADRATICROOTFINDER_HH_

#include <limits>

#include "MonteRayDefinitions.hh"

namespace MonteRay {

typedef gpuRayFloat_t Float_t;

class Roots {
public:
//#ifndef __CUDA_ARCH__
//    const Float_t NearEpsilon = 100.0 * std::numeric_limits<Float_t>::infinity();
//#else
//    const Float_t NearEpsilon = 100.0 * FLT_MAX;
//#endif
	static constexpr Float_t inf = std::numeric_limits<Float_t>::infinity();
	CUDA_CALLABLE_MEMBER Roots(){}
	CUDA_CALLABLE_MEMBER ~Roots(){}

	Float_t R1 = inf;
	Float_t R2 = inf;

	CUDA_CALLABLE_MEMBER Float_t min(){ if( R1 < R2 ) return R1; return R2; }
};

CUDA_CALLABLE_MEMBER
Float_t FindMinPositiveRoot(Float_t, Float_t, Float_t);

CUDA_CALLABLE_MEMBER
Roots FindPositiveRoots(Float_t, Float_t, Float_t);


} /* namespace MonteRay */

#endif /* MONTERAYQUADRATICROOTFINDER_HH_ */
