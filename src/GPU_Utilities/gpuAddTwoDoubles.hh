#ifndef GPUADDTWODOUBLES_HH_
#define GPUADDTWODOUBLES_HH_

#include "MonteRayTypes.hh"

namespace MonteRay {

CUDA_CALLABLE_KERNEL void add_double(unsigned N, int *a, int *b, int *c );
CUDA_CALLABLE_KERNEL void add_single(unsigned N, int *a, int *b, int *c );

double gpuAddTwoDoubles( double, double );
float  gpuAddTwoFloats(  float, float );

} /* namespace MonteRay */

#endif /* GPUADDTWODOUBLES_HH_ */
