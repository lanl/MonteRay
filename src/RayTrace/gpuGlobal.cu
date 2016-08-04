#include "gpuGlobal.h"

namespace MonteRay{

void cudaReset(void) {
#ifdef CUDA
	cudaDeviceReset();
	gpuErrchk( cudaPeekAtLastError() );
#endif
}

void gpuCheck() {
	int deviceCount;

	cuInit(0);
	cuDeviceGetCount(&deviceCount);
	if (deviceCount == 0) {
		printf("No CUDA-compatible devices found\n");
		exit(1);
	}
	printf("Number of CUDA devices=%d\n",deviceCount);
	gpuErrchk( cudaPeekAtLastError() );
}

gpuSync::gpuSync(){
	cudaEventCreate(&sync_event);
}

gpuSync::~gpuSync(){
	cudaEventDestroy(sync_event);
}

void gpuSync::sync(){
	cudaEventRecord(sync_event, 0);
	cudaEventSynchronize(sync_event);
}

void setCudaPrintBufferSize( size_t size) {
#ifdef CUDA
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, size );
#endif
}

}
