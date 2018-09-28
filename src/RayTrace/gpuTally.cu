#include "gpuTally.hh"

#include <stdexcept>
#include <fstream>

#include "GPUErrorCheck.hh"
#include "GPUAtomicAdd.hh"
#include "MonteRay_binaryIO.hh"

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


void cudaCtor(gpuTally* ptr, unsigned num) {
#ifdef __CUDACC__
    ptr->size = num;
    unsigned allocSize = sizeof( gpuTallyType_t ) * num;

    CUDA_CHECK_RETURN( cudaMalloc(&ptr->tally, allocSize ));
#endif
}

void cudaCtor(gpuTally* pCopy, gpuTally* pOrig) {
#ifdef __CUDACC__
    unsigned num = pOrig->size;
    cudaCtor( pCopy, num);

    unsigned allocSize = sizeof( gpuTallyType_t ) * num;

    CUDA_CHECK_RETURN( cudaMemcpy(pCopy->tally, pOrig->tally, allocSize, cudaMemcpyHostToDevice));
#endif
}

void cudaDtor(gpuTally* ptr) {
#ifdef __CUDACC__
    cudaFree( ptr->tally );
#endif
}


void copy(struct gpuTally* pCopy, struct gpuTally* pOrig) {
    unsigned num = pOrig->size;
    if( num <=0 ) { num = 1; }

    ctor( pCopy, num);
    for( unsigned i=0; i<num; ++i ){
        pCopy->tally[i] = pOrig->tally[i];
    }
}

CUDA_CALLABLE_MEMBER
void score(struct gpuTally* ptr, unsigned cell, gpuTallyType_t value ) {
    gpu_atomicAdd( &(ptr->tally[cell]), value);
}

gpuTallyHost::gpuTallyHost(unsigned num) {
    ctor(num);
}

void gpuTallyHost::ctor( unsigned num) {
    ptr = new gpuTally;
    MonteRay::ctor(ptr, num);
    cudaCopyMade = false;
    ptr_device = NULL;
    temp = NULL;
}

void gpuTallyHost::dtor() {
    if( ptr != 0 ) {
        MonteRay::dtor( ptr );
        delete ptr;
        ptr = 0;
    }

    if( cudaCopyMade ) {
        cudaDtor( temp );
        delete temp;
#ifdef __CUDACC__
        cudaFree( ptr_device );
#endif
    }
}


gpuTallyHost::~gpuTallyHost() {
    dtor();
}

void gpuTallyHost::clear(void) {
    for( unsigned i=0; i< size(); ++i ) {
        ptr->tally[i] = 0.0;
    }
#ifdef __CUDACC__
    unsigned num = ptr->size;
    unsigned allocSize = sizeof( gpuTallyType_t ) * num;

    // copy data
    CUDA_CHECK_RETURN( cudaMemcpy(temp->tally, ptr->tally, allocSize, cudaMemcpyHostToDevice));
#endif
}

void gpuTallyHost::copyToGPU(void) {
#ifdef __CUDACC__
    CUDA_CHECK_RETURN( cudaMalloc(&ptr_device, sizeof( gpuTally) ));

    cudaCopyMade = true;

    temp = new gpuTally;
    cudaCtor(temp, ptr );

    // copy ptr data
    CUDA_CHECK_RETURN( cudaMemcpy(ptr_device, temp, sizeof( gpuTally ), cudaMemcpyHostToDevice));

    // copy data
    CUDA_CHECK_RETURN( cudaMemcpy(temp->tally, ptr->tally, sizeof( gpuTallyType_t ) * ptr->size, cudaMemcpyHostToDevice));
#endif
}

void gpuTallyHost::copyToCPU(void) {
#ifdef __CUDACC__
    if( ! cudaCopyMade ) {
        throw std::runtime_error( "Error: gpuTallyHost::copyToCPU -- no copy to GPU made so can not copy from GPU" );
    }

    unsigned num = ptr->size;
    unsigned allocSize = sizeof( gpuTallyType_t ) * num;

    // copy data
    CUDA_CHECK_RETURN( cudaMemcpy(ptr->tally, temp->tally, allocSize, cudaMemcpyDeviceToHost));
#endif
}

void gpuTallyHost::write( std::string filename ) const {
    std::ofstream outfile;
    outfile.open( filename.c_str(), std::ios::binary | std::ios::out);
    if( ! outfile.is_open() ) {
        fprintf(stderr, "gpuTallyHost::write -- Failure to open file to write gpuTally info,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
        exit(1);
    }
    assert( outfile.good() );
    outfile.exceptions(std::ios_base::failbit | std::ios_base::badbit );

    outfile.seekp(0, std::ios::beg); // reposition to start of file

    unsigned version = 0;
    binaryIO::write(outfile,version);
    binaryIO::write(outfile,size());

    for( unsigned i = 0; i< size(); ++i ){
        binaryIO::write(outfile, getTally(i) );
    }
    outfile.close();
}

void gpuTallyHost::read( std::string filename ) {
    std::ifstream infile;
    infile.open( filename.c_str(), std::ios::binary | std::ios::in);
    if( ! infile.is_open() ) {
        fprintf(stderr, "gpuTallyHost::write -- Failure to open file to read gpuTally info,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
        exit(1);
    }
    assert( infile.good() );
    infile.exceptions(std::ios_base::failbit | std::ios_base::badbit );

    infile.seekg(0, std::ios::beg); // reposition to start of file

    unsigned version;
    binaryIO::read(infile,version);

    unsigned size;
    binaryIO::read(infile,size);

    dtor();
    ctor(size);

    for( unsigned i = 0; i < size; ++i ){
        gpuTallyType_t value;
        binaryIO::read(infile, value);
        setTally(i,value);
    }
    infile.close();
}

}



