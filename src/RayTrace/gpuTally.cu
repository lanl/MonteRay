#include "gpuTally.h"
#include <stdexcept>

namespace MonteRay{

void ctor(struct gpuTally* pOrig, unsigned num) {
    if( num <=0 ) { num = 1; }

    pOrig->size = num;

    unsigned allocSize = sizeof(gpuTallyType_t)*num;
    pOrig->tally = (gpuTallyType_t*) malloc( allocSize);
    if (pOrig->tally == 0) abort ();

    for( unsigned i=0; i<num; ++i ){
    	pOrig->tally[i] =  0.0;
    }
}

void dtor(struct gpuTally* ptr){
    if( ptr->tally != 0 ) {
        free(ptr->tally);
        ptr->tally = 0;
    }
}

#ifdef CUDA
void cudaCtor(gpuTally* ptr, unsigned num) {
     gpuErrchk( cudaPeekAtLastError() );

     ptr->size = num;
     unsigned allocSize = sizeof( gpuTallyType_t ) * num;

     CUDA_CHECK_RETURN( cudaMalloc(&ptr->tally, allocSize ));
     gpuErrchk( cudaPeekAtLastError() );
}

void cudaCtor(gpuTally* pCopy, gpuTally* pOrig) {
	unsigned num = pOrig->size;
	cudaCtor( pCopy, num);

	unsigned allocSize = sizeof( gpuTallyType_t ) * num;

    CUDA_CHECK_RETURN( cudaMemcpy(pCopy->tally, pOrig->tally, allocSize, cudaMemcpyHostToDevice));
    gpuErrchk( cudaPeekAtLastError() );
}

void cudaDtor(gpuTally* ptr) {
    cudaFree( ptr->tally );
}
#endif


void copy(struct gpuTally* pCopy, struct gpuTally* pOrig) {
    unsigned num = pOrig->size;
    if( num <=0 ) { num = 1; }

    ctor( pCopy, num);
    for( unsigned i=0; i<num; ++i ){
        pCopy->tally[i] = pOrig->tally[i];
    }
}

#ifdef CUDA
__device__
#endif
void score(struct gpuTally* ptr, unsigned cell, gpuTallyType_t value ) {
	atomicAdd( &(ptr->tally[cell]), value);
	//atomicAddDouble( &(ptr->tally[cell]), value );
}

gpuTallyHost::gpuTallyHost(unsigned num) {
	ptr = new gpuTally;
    ctor(ptr, num);
    cudaCopyMade = false;
    ptr_device = NULL;
    temp = NULL;
}

gpuTallyHost::~gpuTallyHost() {
    if( ptr != 0 ) {
        dtor( ptr );
        delete ptr;
        ptr = 0;
    }

    if( cudaCopyMade ) {
    	cudaDtor( temp );
    	delete temp;
#ifdef CUDA
    	cudaFree( ptr_device );
#endif
    }
}

void gpuTallyHost::clear(void) {
	for( unsigned i=0; i< size(); ++i ) {
		ptr->tally[i] = 0.0;
	}
#ifdef CUDA
	unsigned num = ptr->size;
	unsigned allocSize = sizeof( gpuTallyType_t ) * num;

	// copy data
    CUDA_CHECK_RETURN( cudaMemcpy(temp->tally, ptr->tally, allocSize, cudaMemcpyHostToDevice));
    gpuErrchk( cudaPeekAtLastError() );
#endif
}

void gpuTallyHost::copyToGPU(void) {
#ifdef CUDA
    CUDA_CHECK_RETURN( cudaMalloc(&ptr_device, sizeof( gpuTally) ));
    gpuErrchk( cudaPeekAtLastError() );

	cudaCopyMade = true;

	temp = new gpuTally;
	cudaCtor(temp, ptr );

	// copy data
	CUDA_CHECK_RETURN( cudaMemcpy(ptr_device, temp, sizeof( gpuTally ), cudaMemcpyHostToDevice));
	gpuErrchk( cudaPeekAtLastError() );
#endif
}

void gpuTallyHost::copyToCPU(void) {
	if( ! cudaCopyMade ) {
		throw std::runtime_error( "Error: gpuTallyHost::copyToCPU -- no copy to GPU made so can not copy from GPU" );
	}

#ifdef CUDA
	unsigned num = ptr->size;
	unsigned allocSize = sizeof( gpuTallyType_t ) * num;

	// copy data
    CUDA_CHECK_RETURN( cudaMemcpy(ptr->tally, temp->tally, allocSize, cudaMemcpyDeviceToHost));
    gpuErrchk( cudaPeekAtLastError() );
#endif
}

}



