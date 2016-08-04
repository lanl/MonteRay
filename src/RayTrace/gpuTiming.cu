#include "gpuTiming.h"
#include <stdexcept>
#include <climits>

namespace MonteRay{

void ctor(struct gpuTiming* pOrig) {
	pOrig->start = 0;
	pOrig->stop = 0;
}

void dtor(struct gpuTiming* ptr){
}

gpuTimingHost::gpuTimingHost() {
	ptr = new gpuTiming;
    ctor(ptr);
    cudaCopyMade = false;
    ptr_device = NULL;

    rate = 0;
#ifdef CUDA
    setRate( gpuTimingHost::getCyclesPerSecond() );
    copyToGPU();
#endif

}

gpuTimingHost::~gpuTimingHost() {
    if( ptr != 0 ) {
        dtor( ptr );
        delete ptr;
        ptr = 0;
    }

    if( cudaCopyMade ) {
#ifdef CUDA
    	cudaFree( ptr_device );
#endif
    }
}

void gpuTimingHost::copyToGPU(void) {
#ifdef CUDA
	if(cudaCopyMade != true ) {
		cudaCopyMade = true;

		// allocate target struct
		CUDA_CHECK_RETURN( cudaMalloc(&ptr_device, sizeof( gpuTiming) ));
		gpuErrchk( cudaPeekAtLastError() );
	}

	// copy data
	CUDA_CHECK_RETURN( cudaMemcpy(ptr_device, ptr, sizeof( gpuTiming ), cudaMemcpyHostToDevice));
	gpuErrchk( cudaPeekAtLastError() );
#endif
}

void gpuTimingHost::copyToCPU(void) {
#ifdef CUDA
	cudaCopyMade = true;

	// copy data
	CUDA_CHECK_RETURN( cudaMemcpy(ptr, ptr_device, sizeof( gpuTiming ), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaPeekAtLastError() );
#endif
}

void copy(struct gpuTiming* pCopy, struct gpuTiming* pOrig) {
	pCopy->start = pOrig->start;
	pCopy->stop = pOrig->stop;
}

// Returns number of cycles required for requested seconds
clock64_t
gpuTimingHost::getCyclesPerSecond(void)
{
	clock64_t Hz = 0;
    // Get device frequency in Hz
#ifdef CUDA
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    gpuErrchk( cudaPeekAtLastError() );
    Hz = clock64_t(prop.clockRate) * 1000;
#endif
    return Hz;
}

double gpuTimingHost::getGPUTime(void) {
	if( rate == 0 ) {
		throw std::runtime_error( "GPU rate not set." );
	}
	if( !cudaCopyMade ){
		throw std::runtime_error( "gpuTiming not sent to GPU." );
	}
	copyToCPU();

	clock64_t deltaT;
	clock64_t start = ptr->start;
	clock64_t stop = ptr->stop;
	deltaT = stop - start;

	return double(deltaT) / rate;
}

}

