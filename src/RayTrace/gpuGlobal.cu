#include "gpuGlobal.h"

void cudaReset(void) {
#ifdef CUDA
	cudaDeviceReset();
	gpuErrchk( cudaPeekAtLastError() );
#endif
}
