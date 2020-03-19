#ifndef GPUADDTWODOUBLES_HH_
#define GPUADDTWODOUBLES_HH_

#include "MonteRayTypes.hh"

namespace MonteRay {

CUDA_CALLABLE_KERNEL  add_double(unsigned N, double *a, double *b, double *c );
CUDA_CALLABLE_KERNEL  add_single(unsigned N, float *a, float *b, float *c );

double gpuAddTwoDoubles( double, double );
float  gpuAddTwoFloats(  float, float );

} /* namespace MonteRay */

#endif /* GPUADDTWODOUBLES_HH_ */
