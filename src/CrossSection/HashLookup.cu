#include "HashLookup.h"

#include <math.h>

#include "GPUErrorCheck.hh"
#include "MonteRayDefinitions.hh"
#include "MonteRayConstants.hh"

#include "MonteRay_binaryIO.hh"
#include "MonteRayCrossSection.hh"

namespace MonteRay{

void ctor(HashLookup* ptr, unsigned num, unsigned nBins ) {
    if( num <=0 ) { num = 1; }
    if( nBins <=2 ) { nBins = 3; }

    ptr->maxNumIsotopes = num;
    ptr->numIsotopes = 0;
    ptr->eMax = 0.0f;
    ptr->eMin = MonteRay::inf;
    ptr->N = nBins;

    unsigned allocSize = sizeof(unsigned)*nBins*num;

    ptr->binBounds  = (unsigned*) malloc( allocSize);
    if(ptr->binBounds == 0) abort ();

    for( unsigned i=0; i<num*nBins; ++i ){
    	ptr->binBounds[i] = 0U;
    }
}


void cudaCtor(HashLookup* pCopy, unsigned num, unsigned nBins) {
#ifdef __CUDACC__
	pCopy->maxNumIsotopes = num;
	pCopy->N = nBins;

	// binBounds
	unsigned allocSize = sizeof(unsigned)*num*nBins;
	CUDA_CHECK_RETURN( cudaMalloc(&pCopy->binBounds, allocSize ));
#endif
}

void cudaCtor(struct HashLookup* pCopy, struct HashLookup* pOrig){
#ifdef __CUDACC__
	cudaCtor( pCopy, pOrig->maxNumIsotopes, pOrig->N);
	pCopy->maxNumIsotopes = pOrig->maxNumIsotopes;
	pCopy->numIsotopes = pOrig->numIsotopes;
	pCopy->eMin = pOrig->eMin;
	pCopy->eMax = pOrig->eMax;
	pCopy->delta = pOrig->delta;
#endif
}

void dtor(HashLookup* ptr) {
    if( ptr->binBounds != 0 ) {
        free( ptr->binBounds );
        ptr->binBounds = 0;
    }
}

void cudaDtor(HashLookup* ptr) {
#ifdef __CUDACC__
	cudaFree( ptr->binBounds );
#endif
}

HashLookupHost::HashLookupHost(unsigned num, unsigned nBins) {
     ptr = new HashLookup;
     ctor( ptr, num, nBins);
     temp = NULL;
     ptr_device = NULL;
     cudaCopyMade = false;
}


HashLookupHost::~HashLookupHost() {
     dtor( ptr );
     delete ptr;

#ifdef __CUDACC__
     if( cudaCopyMade ) {
       	cudaDtor( temp );
       	delete temp;
     	cudaFree( ptr_device );
     }
#endif
}


void HashLookupHost::copyToGPU(void) {
#ifdef __CUDACC__
	cudaCopyMade = true;
    temp = new HashLookup;
//    copy(temp, ptr);

	unsigned num = ptr->maxNumIsotopes;

	// allocate target struct
	CUDA_CHECK_RETURN( cudaMalloc(&ptr_device, sizeof( HashLookup ) ));

	// allocate target dynamic memory
	cudaCtor( temp, ptr);

	unsigned allocSize = sizeof(unsigned)*num*ptr->N;
	CUDA_CHECK_RETURN( cudaMemcpy(temp->binBounds, ptr->binBounds, allocSize, cudaMemcpyHostToDevice));

	// copy data
	CUDA_CHECK_RETURN( cudaMemcpy(ptr_device, temp, sizeof( HashLookup ), cudaMemcpyHostToDevice));
#endif
}

void copy(HashLookup* pCopy, const HashLookup* const pOrig ) {
    unsigned num = pOrig->maxNumIsotopes;
    if( num <=0 ) { num = 1; }

    ctor( pCopy, num, pOrig->N);

	pCopy->numIsotopes = pOrig->numIsotopes;
	pCopy->eMin = pOrig->eMin;
	pCopy->eMax = pOrig->eMax;
	pCopy->delta = pOrig->delta;

    for( unsigned i=0; i<num*pOrig->N; ++i ){
        pCopy->binBounds[i] = pOrig->binBounds[i];
    }
}

CUDA_CALLABLE_MEMBER
unsigned getMaxNumIsotopes(const HashLookup* ptr ) {
    return ptr->maxNumIsotopes;
}

CUDA_CALLABLE_MEMBER
unsigned getNumIsotopes(const HashLookup* ptr ) {
    return ptr->numIsotopes;
}

CUDA_CALLABLE_MEMBER
gpuFloatType_t getMaxEnergy(const HashLookup* ptr ) {
	return std::exp(ptr->eMax);
}

CUDA_CALLABLE_MEMBER
gpuFloatType_t getMinEnergy(const HashLookup* ptr ) {
	return std::exp(ptr->eMin);
}

CUDA_CALLABLE_MEMBER
unsigned getNBins(const HashLookup* ptr ) {
	return ptr->N;
}

unsigned HashLookupHost::getNBins(void) {
	return MonteRay::getNBins( ptr );
}

CUDA_CALLABLE_MEMBER
bool setHashMinMax(HashLookup* ptr, MonteRayCrossSection* xs ) {
	setID(xs, ptr->numIsotopes );

	ptr->numIsotopes++;
	unsigned numIsotopes = ptr->numIsotopes;
	unsigned numBins = ptr->N;
	if( numIsotopes > ptr->maxNumIsotopes ) {
		printf("Error: HasLookup::addIsotope -- exceeded max number of isotopes. %s %d", __FILE__, __LINE__);
		return true;
	}
	if(xs->energies[0] <= 0.0  ) {
		printf("Error: HasLookup::addIsotope -- minimum cross-section can not be zero or less. %s %d", __FILE__, __LINE__);
		return true;
	}

	if( logf(xs->energies[0]) < ptr->eMin ) { ptr->eMin = logf(xs->energies[0]); }
	if( logf(xs->energies[ xs->numPoints -1]) > ptr->eMax ) { ptr->eMax = logf(xs->energies[xs->numPoints-1]); }

	if( ptr->eMin > ptr->eMax ) {
		printf("Error: HasLookup::addIsotope -- min energy > max energy %s %d", __FILE__, __LINE__);
		return true;
	}
	ptr->delta = (ptr->eMax - ptr->eMin)/numBins;
	return false;
}

CUDA_CALLABLE_MEMBER
void setHashBinBounds(HashLookup* ptr, MonteRayCrossSection* xs, unsigned j ) {
	for( unsigned i = 0; i < ptr->N; ++i ){
		unsigned index = getBinBoundIndex(ptr, j, i);
		gpuFloatType_t hashEnergy = std::exp( ptr->eMin + i*ptr->delta);
		ptr->binBounds[index] = getIndex(xs, hashEnergy);
	}
}

CUDA_CALLABLE_MEMBER
unsigned getBinBoundIndex(const HashLookup* ptr, unsigned isotope, unsigned index ){
	if( isotope > ptr->numIsotopes) {
		printf("Error: HasLookup::getBinBoundIndex -- isotope ( = %d )  > numIsotopes (= %d), %s %d\n", isotope, ptr->numIsotopes, __FILE__, __LINE__);
		ABORT( "HashLookup.cu -- getBinBoundIndex" );
	}
	if( index > ptr->N) {
		printf("Error: HasLookup::getBinBoundIndex -- index ( = %d )  > numBins (= %d), %s %d\n", index, ptr->N, __FILE__, __LINE__);
		ABORT( "HashLookup.cu -- getBinBoundIndex" );
	}
	unsigned i = isotope + index*ptr->maxNumIsotopes;
	if( i >= ptr->maxNumIsotopes*ptr->N ){
		printf("Error: HasLookup::getBinBoundIndex -- index outside of range. isotope = %d, index=%d, %s %d\n", isotope, index, __FILE__, __LINE__);
		printf("Error: HasLookup::getBinBoundIndex -- index outside of range. i = %d, N*maxNumIsotopes=%d,\n", i, ptr->maxNumIsotopes*ptr->N  );
		printf("Error: HasLookup::getBinBoundIndex -- index outside of range. N = %d, maxNumIsotopes=%d,\n", ptr->N, ptr->maxNumIsotopes  );
		ABORT( "HashLookup.cu -- getBinBoundIndex" );
	}
	return i;
}


void HashLookupHost::addIsotope( MonteRayCrossSectionHost* xs ) {
	addIsotope( xs->getXSPtr() );
}

void HashLookupHost::addIsotope( MonteRayCrossSection* xs ) {
	xsList.push_back(xs);
	if( xs->id < 0 ) {
		bool err = MonteRay::setHashMinMax(ptr, xs );
		if( err ) {
			throw std::runtime_error( "Error:  HashLookupHost::addIsotope -- setHashMinMax failure.\n");
		}
	}
	for( unsigned i=0; i<xsList.size(); ++i) {
		MonteRay::setHashBinBounds( ptr, xsList.at(i), i);
	}
}

CUDA_CALLABLE_MEMBER
unsigned getHashBin(const HashLookup* ptr, gpuFloatType_t energy ) {
	gpuFloatType_t logE = logf(energy);
	if( logE <= ptr->eMin) { return 0; }
	if( logE >= ptr->eMax ) { return ptr->N-1; }
	return (logE-ptr->eMin)/ptr->delta;
}

CUDA_CALLABLE_MEMBER
unsigned getLowerBoundbyIndex(const HashLookup* ptr, unsigned isotope, unsigned index ){
	return ptr->binBounds[getBinBoundIndex( ptr, isotope, index )];
}

CUDA_CALLABLE_MEMBER
unsigned getUpperBoundbyIndex(const HashLookup* ptr, unsigned isotope, unsigned index ){
	if( index < ptr->N - 1 ) {
		return getLowerBoundbyIndex( ptr, isotope, index+1) + 1;
	}
	return index;
}

unsigned HashLookupHost::getLowerBoundbyIndex( unsigned isotope, unsigned index) const {
	return MonteRay::getLowerBoundbyIndex(ptr, isotope, index);
}

unsigned HashLookupHost::getUpperBoundbyIndex( unsigned isotope, unsigned index) const {
	return MonteRay::getUpperBoundbyIndex(ptr, isotope, index);
}

}
