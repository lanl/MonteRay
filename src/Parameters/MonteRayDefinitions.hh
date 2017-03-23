/*
 * MonteRayDefinitions.hh
 *
 *  Created on: Mar 20, 2017
 *      Author: jsweezy  - jsweezy@lanl.gov
 */

#ifndef MONTERAYDEFINITIONS_HH_
#define MONTERAYDEFINITIONS_HH_

#ifdef CUDA
#include <cuda.h>
#endif

#include "MonteRayPreprocessorDefinitions.hh"

#define TALLY_DOUBLEPRECISION 1 // turn on (1) and off (0) for double precision tally array and compute

namespace MonteRay{

// typedefs
typedef float float_t;
typedef float gpuFloatType_t;

#if TALLY_DOUBLEPRECISION < 1
typedef float gpuTallyType_t;
#else
typedef double gpuTallyType_t;
#endif

typedef long long clock64_t;

}

#endif /* MONTERAYDEFINITIONS_HH_ */
