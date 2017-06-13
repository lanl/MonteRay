#include "MonteRayCrossSection.hh"

#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "GPUErrorCheck.hh"
#include "MonteRay_binaryIO.hh"

namespace MonteRay{

void ctor(struct MonteRayCrossSection* pXS, unsigned num) {
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

void dtor(struct MonteRayCrossSection* pXS) {
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
void cudaCtor(MonteRayCrossSection* ptr, unsigned num) {

	 ptr->numPoints = num;
     unsigned allocSize = sizeof( gpuFloatType_t ) * num;

     CUDA_CHECK_RETURN( cudaMalloc(&ptr->energies, allocSize ));

     CUDA_CHECK_RETURN( cudaMalloc(&ptr->totalXS, allocSize ));
}

void cudaCtor(MonteRayCrossSection* pCopy, MonteRayCrossSection* pOrig) {
	unsigned num = pOrig->numPoints;
	cudaCtor( pCopy, num);

	pCopy->id = pOrig->id;
	pCopy->AWR = pOrig->AWR;

	unsigned allocSize = sizeof( gpuFloatType_t ) * pOrig->numPoints;

    CUDA_CHECK_RETURN( cudaMemcpy(pCopy->energies, pOrig->energies, allocSize, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN( cudaMemcpy(pCopy->totalXS, pOrig->totalXS, allocSize, cudaMemcpyHostToDevice));
}

void cudaDtor(MonteRayCrossSection* ptr) {
    cudaFree( ptr->energies );
    cudaFree( ptr->totalXS );
}
#endif

void copy(struct MonteRayCrossSection* pCopy, struct MonteRayCrossSection* pOrig ) {
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
int getID(struct MonteRayCrossSection* pXS) {
	return pXS->id;
}

#ifdef CUDA
__device__ __host__
#endif
void setID(struct MonteRayCrossSection* pXS, unsigned i) {
	if( pXS->id < 0 ) {
		pXS->id = i;
	}
}

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getEnergy(struct MonteRayCrossSection* pXS, unsigned i ) {
    return pXS->energies[i];
}


#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getTotalXSByIndex(struct MonteRayCrossSection* pXS, unsigned i ){
    return pXS->totalXS[i];
}

#ifdef CUDA
__device__ __host__
#endif
unsigned getIndex(struct MonteRayCrossSection* pXS, gpuFloatType_t value ){
    // modified from http://en.cppreference.com/w/cpp/algorithm/upper_bound
	return getIndexBinary( pXS, 0, pXS->numPoints-1, value);
}

#ifdef CUDA
__device__ __host__
#endif
unsigned getIndexBinary(struct MonteRayCrossSection* pXS, unsigned lower, unsigned upper, gpuFloatType_t value ){
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
unsigned getIndexLinear(struct MonteRayCrossSection* pXS, unsigned lower, unsigned upper, gpuFloatType_t value ){

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
unsigned getIndex(struct MonteRayCrossSection* pXS, struct HashLookup* pHash, unsigned hashBin, gpuFloatType_t E ){
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
gpuFloatType_t getAWR(struct MonteRayCrossSection* pXS) {
    return pXS->AWR;
}

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getTotalXSByIndex(struct MonteRayCrossSection* pXS, unsigned i, gpuFloatType_t E ) {

    gpuFloatType_t lower =  pXS->totalXS[i];
    gpuFloatType_t upper =  pXS->totalXS[i+1];
    gpuFloatType_t deltaE = pXS->energies[i+1] - pXS->energies[i];

    gpuFloatType_t value = lower + (upper-lower) * (E - pXS->energies[i])/deltaE;
    return value;
}

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getTotalXS(struct MonteRayCrossSection* pXS, gpuFloatType_t E ) {

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
gpuFloatType_t getTotalXS(struct MonteRayCrossSection* pXS, struct HashLookup* pHash, unsigned hashBin, gpuFloatType_t E ) {

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
__global__ void kernelGetTotalXS(struct MonteRayCrossSection* pXS, HashLookup* pHash, unsigned HashBin, gpuFloatType_t E, gpuFloatType_t* results){
    results[0] = getTotalXS(pXS, pHash, HashBin, E);
    return;
}
#endif

#ifdef CUDA
__global__ void kernelGetTotalXS(struct MonteRayCrossSection* pXS,  gpuFloatType_t E, gpuFloatType_t* results){
    results[0] = getTotalXS(pXS, E);
    return;
}
#endif

gpuFloatType_t
launchGetTotalXS( MonteRayCrossSectionHost* pXS, gpuFloatType_t energy){
#ifdef CUDA
	gpuFloatType_t* result_device;
	gpuFloatType_t result[1];
	CUDA_CHECK_RETURN( cudaMalloc( &result_device, sizeof( gpuFloatType_t) * 1 ));

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	kernelGetTotalXS<<<1,1>>>( pXS->xs_device, energy, result_device);
	gpuErrchk( cudaPeekAtLastError() );
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);

	CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(gpuFloatType_t)*1, cudaMemcpyDeviceToHost));

	cudaFree( result_device );
	return result[0];
#else
	return -100.0;
#endif
}


#if !defined( CUDA )
#include "ContinuousNeutron.hh"
void MonteRayCrossSectionHost::load( const ContinuousNeutron& cn){
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

MonteRayCrossSectionHost::MonteRayCrossSectionHost(unsigned num){
    xs = (struct MonteRayCrossSection*) malloc( sizeof(struct MonteRayCrossSection) );
    ctor(xs,num);

    cudaCopyMade = false;
    temp = NULL;

#ifdef CUDA
    CUDA_CHECK_RETURN( cudaMalloc(&xs_device, sizeof( MonteRayCrossSection) ));
#endif
}

MonteRayCrossSectionHost::~MonteRayCrossSectionHost(){
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

gpuFloatType_t MonteRayCrossSectionHost::getTotalXS( struct HashLookup* pHash, unsigned hashBin, gpuFloatType_t E ) const {
	return MonteRay::getTotalXS(xs, pHash, hashBin, E);
}

gpuFloatType_t MonteRayCrossSectionHost::getTotalXSByHashIndex(struct HashLookup* pHash, unsigned i, gpuFloatType_t E) const {
	return MonteRay::getTotalXS(xs, pHash, i, E);
}

unsigned MonteRayCrossSectionHost::getIndex( HashLookupHost* pHost, unsigned hashBin, gpuFloatType_t e ) const {
	return MonteRay::getIndex( xs, pHost->getPtr(), hashBin, e);
}


void MonteRayCrossSectionHost::copyToGPU(void) {
#ifdef CUDA
    cudaCopyMade = true;
    temp = new MonteRayCrossSection;
    cudaCtor(temp, xs );
    CUDA_CHECK_RETURN( cudaMemcpy(xs_device, temp, sizeof( MonteRayCrossSection ), cudaMemcpyHostToDevice));
#endif
}

void MonteRayCrossSectionHost::load(struct MonteRayCrossSection* ptrXS ) {
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

void MonteRayCrossSectionHost::write(std::ostream& outf) const{
    binaryIO::write(outf, xs->numPoints );
    binaryIO::write(outf, xs->AWR );
    for( unsigned i=0; i<xs->numPoints; ++i ){
        binaryIO::write(outf, xs->energies[i] );
    }
    for( unsigned i=0; i<xs->numPoints; ++i ){
        binaryIO::write(outf, xs->totalXS[i] );
    }
}

void MonteRayCrossSectionHost::read(std::istream& infile) {
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


void MonteRayCrossSectionHost::write( const std::string& filename ) {
    std::ofstream outfile;

    outfile.open( filename.c_str(), std::ios::binary | std::ios::out);
    if( ! outfile.is_open() ) {
        fprintf(stderr, "MonteRayCrossSectionHost::write -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
        exit(1);
    }
    assert( outfile.good() );
    outfile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
    write( outfile );
    outfile.close();
}

void MonteRayCrossSectionHost::read( const std::string& filename ) {
    std::ifstream infile;
    if( infile.is_open() ) {
        infile.close();
    }
    infile.open( filename.c_str(), std::ios::binary | std::ios::in);

    if( ! infile.is_open() ) {
        fprintf(stderr, "MonteRayCrossSectionHost::read -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
        exit(1);
    }
    assert( infile.good() );
    infile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
    read(infile);
    infile.close();
}

}
