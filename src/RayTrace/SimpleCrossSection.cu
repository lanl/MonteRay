#include "SimpleCrossSection.h"

#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "binaryIO.h"

namespace MonteRay{

void ctor(struct SimpleCrossSection* pXS, unsigned num) {
    if( num <=0 ) { num = 1; }

    pXS->id = -1;
    pXS->numPoints = num;
    pXS->AWR = 0.0;

    unsigned allocSize = sizeof(gpuFloatType_t)*num;
    pXS->energies  = (gpuFloatType_t*) malloc( allocSize);
    if (pXS->energies == 0) abort ();

    pXS->totalXS   = (gpuFloatType_t*) malloc( allocSize );
    if (pXS->totalXS == 0) abort ();

    for( unsigned i=0; i<num; ++i ){
        pXS->energies[i] = -1.0;
        pXS->totalXS[i] =  0.0;
    }
}

void dtor(struct SimpleCrossSection* pXS) {
    if( pXS->energies != 0 ) {
        free(pXS->energies);
        pXS->energies = 0;
    }
    if( pXS->totalXS != 0 ) {
        free(pXS->totalXS);
        pXS->totalXS = 0;
    }
}

#ifdef CUDA
void cudaCtor(SimpleCrossSection* ptr, unsigned num) {
     gpuErrchk( cudaPeekAtLastError() );

	 ptr->numPoints = num;
     unsigned allocSize = sizeof( gpuFloatType_t ) * num;

     CUDA_CHECK_RETURN( cudaMalloc(&ptr->energies, allocSize ));
     gpuErrchk( cudaPeekAtLastError() );

     CUDA_CHECK_RETURN( cudaMalloc(&ptr->totalXS, allocSize ));
     gpuErrchk( cudaPeekAtLastError() );
}

void cudaCtor(SimpleCrossSection* pCopy, SimpleCrossSection* pOrig) {
	unsigned num = pOrig->numPoints;
	cudaCtor( pCopy, num);

	pCopy->id = pOrig->id;
	pCopy->AWR = pOrig->AWR;

	unsigned allocSize = sizeof( gpuFloatType_t ) * pOrig->numPoints;

    CUDA_CHECK_RETURN( cudaMemcpy(pCopy->energies, pOrig->energies, allocSize, cudaMemcpyHostToDevice));
    gpuErrchk( cudaPeekAtLastError() );

    CUDA_CHECK_RETURN( cudaMemcpy(pCopy->totalXS, pOrig->totalXS, allocSize, cudaMemcpyHostToDevice));
    gpuErrchk( cudaPeekAtLastError() );
}

void cudaDtor(SimpleCrossSection* ptr) {
    cudaFree( ptr->energies );
    cudaFree( ptr->totalXS );
}
#endif

void copy(struct SimpleCrossSection* pCopy, struct SimpleCrossSection* pOrig ) {
    unsigned num = pOrig->numPoints;
    if( num <=0 ) { num = 1; }

    ctor( pCopy, num);
    pCopy->id = pOrig->id;
    pCopy->AWR = pOrig->AWR;
    for( unsigned i=0; i<num; ++i ){
        pCopy->energies[i] = pOrig->energies[i];
        pCopy->totalXS[i] =  pOrig->totalXS[i];
    }
}

#ifdef CUDA
__device__ __host__
#endif
int getID(struct SimpleCrossSection* pXS) {
	return pXS->id;
}

#ifdef CUDA
__device__ __host__
#endif
void setID(struct SimpleCrossSection* pXS, unsigned i) {
	if( pXS->id < 0 ) {
		pXS->id = i;
	}
}

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getEnergy(struct SimpleCrossSection* pXS, unsigned i ) {
    return pXS->energies[i];
}


#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getTotalXSByIndex(struct SimpleCrossSection* pXS, unsigned i ){
    return pXS->totalXS[i];
}

#ifdef CUDA
__device__ __host__
#endif
unsigned getIndex(struct SimpleCrossSection* pXS, gpuFloatType_t value ){
    // modified from http://en.cppreference.com/w/cpp/algorithm/upper_bound
	return getIndexBinary( pXS, 0, pXS->numPoints-1, value);
}

#ifdef CUDA
__device__ __host__
#endif
unsigned getIndexBinary(struct SimpleCrossSection* pXS, unsigned lower, unsigned upper, gpuFloatType_t value ){
    // modified from http://en.cppreference.com/w/cpp/algorithm/upper_bound
    unsigned it, step;
    unsigned first = lower;
    unsigned count = upper-lower+1;

    while (count > 0U) {
        it = first;
        step = count / 2;
        it += step;
        if(!(value < pXS->energies[it])) {
            first = ++it;
            count -= step + 1;
        } else {
            count = step;
        }
    }
    if( first > 0 ) { --first; }
    return first;
}

#ifdef CUDA
__device__ __host__
#endif
unsigned getIndexLinear(struct SimpleCrossSection* pXS, unsigned lower, unsigned upper, gpuFloatType_t value ){

    for( unsigned i=lower+1; i < upper+1; ++i ){
    	if( value < pXS->energies[ i ] ) {
    		return i-1;
    	}
    }
    if( value < pXS->energies[ lower ] ) { return lower; }
    return upper;
}

#ifdef CUDA
__device__ __host__
#endif
unsigned getIndex(struct SimpleCrossSection* pXS, struct HashLookup* pHash, unsigned hashBin, gpuFloatType_t E ){
	unsigned isotope = MonteRay::getID(pXS);
	unsigned lowerBin = MonteRay::getLowerBoundbyIndex(pHash, isotope, hashBin);
	unsigned upperBin = MonteRay::getUpperBoundbyIndex(pHash, isotope, hashBin);

	if( upperBin-lowerBin+1 <= 8 ){
		return getIndexLinear( pXS, lowerBin, upperBin, E);
	} else {
		return getIndexBinary( pXS, lowerBin, upperBin, E);
	}

}


#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getAWR(struct SimpleCrossSection* pXS) {
    return pXS->AWR;
}

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getTotalXSByIndex(struct SimpleCrossSection* pXS, unsigned i, gpuFloatType_t E ) {

    gpuFloatType_t lower =  pXS->totalXS[i];
    gpuFloatType_t upper =  pXS->totalXS[i+1];
    gpuFloatType_t deltaE = pXS->energies[i+1] - pXS->energies[i];

    gpuFloatType_t value = lower + (upper-lower) * (E - pXS->energies[i])/deltaE;
    return value;
}

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getTotalXS(struct SimpleCrossSection* pXS, gpuFloatType_t E ) {

    if( E > pXS->energies[ pXS->numPoints-1] ) {
        return pXS->totalXS[ pXS->numPoints-1];
    }

    if( E < pXS->energies[ 0 ] ) {
        return pXS->totalXS[ 0 ];
    }

    unsigned i = getIndex(pXS, E);
    return getTotalXSByIndex( pXS, i, E);
}

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getTotalXS(struct SimpleCrossSection* pXS, struct HashLookup* pHash, unsigned hashBin, gpuFloatType_t E ) {

    if( E > pXS->energies[ pXS->numPoints-1] ) {
        return pXS->totalXS[ pXS->numPoints-1];
    }

    if( E < pXS->energies[ 0 ] ) {
        return pXS->totalXS[ 0 ];
    }

    unsigned i = getIndex(pXS, pHash, hashBin, E);
    return getTotalXSByIndex( pXS, i, E);
}

#ifdef CUDA
__global__ void kernelGetTotalXS(struct SimpleCrossSection* pXS, HashLookup* pHash, unsigned HashBin, gpuFloatType_t E, gpuFloatType_t* results){
    results[0] = getTotalXS(pXS, pHash, HashBin, E);
    return;
}
#endif

#ifdef CUDA
__global__ void kernelGetTotalXS(struct SimpleCrossSection* pXS,  gpuFloatType_t E, gpuFloatType_t* results){
    results[0] = getTotalXS(pXS, E);
    return;
}
#endif

gpuFloatType_t
launchGetTotalXS( SimpleCrossSectionHost* pXS, gpuFloatType_t energy){
#ifdef CUDA
	gpuFloatType_t* result_device;
	gpuFloatType_t result[1];
	CUDA_CHECK_RETURN( cudaMalloc( &result_device, sizeof( gpuFloatType_t) * 1 ));
	gpuErrchk( cudaPeekAtLastError() );

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	kernelGetTotalXS<<<1,1>>>( pXS->xs_device, energy, result_device);
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);

    gpuErrchk( cudaPeekAtLastError() );

	CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(gpuFloatType_t)*1, cudaMemcpyDeviceToHost));
	gpuErrchk( cudaPeekAtLastError() );

	cudaFree( result_device );
	return result[0];
#else
	return -100.0;
#endif
}


#if !defined( CUDA )
#include "ContinuousNeutron.hh"
void SimpleCrossSectionHost::load( const ContinuousNeutron& cn){
    unsigned num = cn.getEnergyGrid().GridSize();
    dtor( xs );
    ctor( xs, num );

    gpuFloatType_t ratio = cn.getAWR();
    setAWR( ratio );

    for( unsigned i=0; i<num; ++i ){
        gpuFloatType_t energy = (cn.getEnergyGrid())[i];
        gpuFloatType_t totalXS = cn.TotalXsec( energy, -1.0, i);
        xs->energies[i] = energy;
        xs->totalXS[i] = totalXS;
    }

}
#endif

SimpleCrossSectionHost::SimpleCrossSectionHost(unsigned num){
    xs = (struct SimpleCrossSection*) malloc( sizeof(struct SimpleCrossSection) );
    ctor(xs,num);

    cudaCopyMade = false;
    temp = NULL;

#ifdef CUDA
    CUDA_CHECK_RETURN( cudaMalloc(&xs_device, sizeof( SimpleCrossSection) ));
    gpuErrchk( cudaPeekAtLastError() );
#endif
}

SimpleCrossSectionHost::~SimpleCrossSectionHost(){
    dtor(xs);

    if( xs != 0 ) {
        free(xs);
        xs = 0;
    }

    if( cudaCopyMade ) {
        cudaDtor( temp );
        delete temp;
    }
#ifdef CUDA
    cudaFree( xs_device );
#endif
}

gpuFloatType_t SimpleCrossSectionHost::getTotalXS( struct HashLookup* pHash, unsigned hashBin, gpuFloatType_t E ) const {
	return MonteRay::getTotalXS(xs, pHash, hashBin, E);
}

gpuFloatType_t SimpleCrossSectionHost::getTotalXSByHashIndex(struct HashLookup* pHash, unsigned i, gpuFloatType_t E) const {
	return MonteRay::getTotalXS(xs, pHash, i, E);
}

unsigned SimpleCrossSectionHost::getIndex( HashLookupHost* pHost, unsigned hashBin, gpuFloatType_t e ) const {
	return MonteRay::getIndex( xs, pHost->getPtr(), hashBin, e);
}


void SimpleCrossSectionHost::copyToGPU(void) {
#ifdef CUDA
	gpuErrchk( cudaPeekAtLastError() );
    cudaCopyMade = true;
    temp = new SimpleCrossSection;
    cudaCtor(temp, xs );
    CUDA_CHECK_RETURN( cudaMemcpy(xs_device, temp, sizeof( SimpleCrossSection ), cudaMemcpyHostToDevice));
    gpuErrchk( cudaPeekAtLastError() );
#endif
}

void SimpleCrossSectionHost::load(struct SimpleCrossSection* ptrXS ) {
    unsigned num = ptrXS->numPoints;
    dtor( xs );
    ctor( xs, num );

    setAWR( ptrXS->AWR );

    for( unsigned i=0; i<num; ++i ){
        gpuFloatType_t energy = ptrXS->energies[i];
        gpuFloatType_t totalXS = ptrXS->totalXS[i];
        xs->energies[i] = energy;
        xs->totalXS[i] = totalXS;
    }
}

void SimpleCrossSectionHost::write(std::ostream& outf) const{
    binaryIO::write(outf, xs->numPoints );
    binaryIO::write(outf, xs->AWR );
    for( unsigned i=0; i<xs->numPoints; ++i ){
        binaryIO::write(outf, xs->energies[i] );
    }
    for( unsigned i=0; i<xs->numPoints; ++i ){
        binaryIO::write(outf, xs->totalXS[i] );
    }
}

void SimpleCrossSectionHost::read(std::istream& infile) {
    unsigned num;
    binaryIO::read(infile, num);
    dtor( xs );
    ctor( xs, num );

    binaryIO::read(infile, xs->AWR );
    for( unsigned i=0; i<num; ++i ){
        binaryIO::read(infile, xs->energies[i] );
    }

    for( unsigned i=0; i<num; ++i ){
        binaryIO::read(infile, xs->totalXS[i] );
    }
}


void SimpleCrossSectionHost::write( const std::string& filename ) {
    std::ofstream outfile;

    outfile.open( filename.c_str(), std::ios::binary | std::ios::out);
    if( ! outfile.is_open() ) {
        fprintf(stderr, "SimpleCrossSectionHost::write -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
        exit(1);
    }
    assert( outfile.good() );
    outfile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
    write( outfile );
    outfile.close();
}

void SimpleCrossSectionHost::read( const std::string& filename ) {
    std::ifstream infile;
    if( infile.is_open() ) {
        infile.close();
    }
    infile.open( filename.c_str(), std::ios::binary | std::ios::in);

    if( ! infile.is_open() ) {
        fprintf(stderr, "SimpleCrossSectionHost::read -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
        exit(1);
    }
    assert( infile.good() );
    infile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
    read(infile);
    infile.close();
}

}
