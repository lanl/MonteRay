/*
 * MonteRayQuadraticRootFinder.hh
 *
 *  Created on: Feb 2, 2018
 *      Author: jsweezy
 */

#ifndef MONTERAYQUADRATICROOTFINDER_HH_
#define MONTERAYQUADRATICROOTFINDER_HH_

#include "MonteRayDefinitions.hh"

#include <limits>

namespace MonteRay {

typedef gpuFloatType_t Float_t;

class Roots {
public:
	static constexpr Float_t inf = std::numeric_limits<double>::infinity();
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
