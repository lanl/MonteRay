#include "gpuDeviceProperties.hh"

namespace MonteRay {

int getNumberOfGPUS(void) {
	int count = 0;

#ifdef CUDA
	cudaGetDeviceCount( &count ) ;
	count = 1;
#endif
	return count;
}

}




