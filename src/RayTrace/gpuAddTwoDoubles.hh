#ifndef GPUADDTWODOUBLES_HH_
#define GPUADDTWODOUBLES_HH_

#include <vector>

namespace MonteRay {

#ifdef CUDA
__global__ void add(unsigned N, int *a, int *b, int *c );
#endif

typedef double value_t;
value_t gpuAddTwoDoubles( value_t, value_t );

} /* namespace MonteRay */

#endif /* GPUADDTWODOUBLES_HH_ */
