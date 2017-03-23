#ifndef GPUADDTWODOUBLES_HH_
#define GPUADDTWODOUBLES_HH_

namespace MonteRay {

#ifdef CUDA
__global__ void add_double(unsigned N, int *a, int *b, int *c );
__global__ void add_single(unsigned N, int *a, int *b, int *c );
#endif


double gpuAddTwoDoubles( double, double );
float  gpuAddTwoFloats(  float, float );

} /* namespace MonteRay */

#endif /* GPUADDTWODOUBLES_HH_ */
