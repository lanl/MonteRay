#include "SimpleMaterial.h"

#include "binaryIO.h"

namespace MonteRay{

void ctor(struct SimpleMaterial* ptr, unsigned num) {
    if( num <=0 ) { num = 1; }
    ptr->numIsotopes = num;
    ptr->AtomicWeight = 0.0;

    unsigned allocSize = sizeof(gpuFloatType_t)*num;

    ptr->fraction  = (gpuFloatType_t*) malloc( allocSize);
    if(ptr->fraction == 0) abort ();

    allocSize = sizeof(struct SimpleCrossSection*)*num;
    ptr->xs = (struct SimpleCrossSection**) malloc( allocSize );
    if(ptr->xs == 0) abort ();

    for( unsigned i=0; i<num; ++i ){
        ptr->fraction[i] = 0.0;

        ptr->xs[i] = 0; // initialize to null
    }
}

#ifdef CUDA
void cudaCtor(SimpleMaterial* pCopy, unsigned num) {

	pCopy->numIsotopes = num;
	pCopy->AtomicWeight = 0.0;

	// fractions
	unsigned allocSize = sizeof(gpuFloatType_t)*num;
	CUDA_CHECK_RETURN( cudaMalloc(&pCopy->fraction, allocSize ));
	gpuErrchk( cudaPeekAtLastError() );

    // SimpleCrossSections
	allocSize = sizeof(SimpleCrossSection*)*num;
	CUDA_CHECK_RETURN( cudaMalloc(&pCopy->xs, allocSize ));
	gpuErrchk( cudaPeekAtLastError() );
}

void cudaCtor(struct SimpleMaterial* pCopy, struct SimpleMaterial* pOrig){
	unsigned num = pOrig->numIsotopes;
	cudaCtor( pCopy, num);

	pCopy->AtomicWeight = pOrig->AtomicWeight;
}
#endif

void dtor(struct SimpleMaterial* ptr){
    if( ptr->fraction != 0 ) {
        free(ptr->fraction);
        ptr->fraction = 0;
    }
    if( ptr->xs != 0 ) {
        free(ptr->xs);
        ptr->xs = 0;
    }
}

#ifdef CUDA
void cudaDtor(SimpleMaterial* ptr) {
	cudaFree( ptr->fraction );
	cudaFree( ptr->xs );
}
#endif

SimpleMaterialHost::SimpleMaterialHost(unsigned numIsotopes) {
    pMat = new SimpleMaterial;
    ctor(pMat, numIsotopes );
    cudaCopyMade = false;
    ptr_device = NULL;
    temp = NULL;

    isotope_device_ptr_list = (SimpleCrossSection**) malloc( sizeof(SimpleCrossSection* )*numIsotopes );
    for( unsigned i=0; i< numIsotopes; ++i ){
    	isotope_device_ptr_list[i] = 0;
    }
}

SimpleMaterialHost::~SimpleMaterialHost() {
    if( pMat != 0 ) {
        dtor( pMat );
        delete pMat;
        pMat = 0;
    }

    if( cudaCopyMade ) {
#ifdef CUDA
    	cudaFree( ptr_device );
    	cudaDtor( temp );
    	delete temp;
#endif
    }
    free( isotope_device_ptr_list );
}

void SimpleMaterialHost::copyToGPU(void) {
#ifdef CUDA
	cudaCopyMade = true;
	temp = new SimpleMaterial;

    copy(temp, pMat);

	unsigned num = pMat->numIsotopes;

	temp->numIsotopes = pMat->numIsotopes;
	temp->AtomicWeight = pMat->AtomicWeight;

	// allocate target struct
	CUDA_CHECK_RETURN( cudaMalloc(&ptr_device, sizeof( SimpleMaterial) ));
	gpuErrchk( cudaPeekAtLastError() );

	// allocate target dynamic memory
	cudaCtor( temp, pMat);

	unsigned allocSize = sizeof(gpuFloatType_t)*num;
	CUDA_CHECK_RETURN( cudaMemcpy(temp->fraction, pMat->fraction, allocSize, cudaMemcpyHostToDevice));
	gpuErrchk( cudaPeekAtLastError() );

	allocSize = sizeof(SimpleCrossSection*)*num;
	CUDA_CHECK_RETURN( cudaMemcpy(temp->xs, isotope_device_ptr_list, allocSize, cudaMemcpyHostToDevice));
	gpuErrchk( cudaPeekAtLastError() );

	// copy data
	CUDA_CHECK_RETURN( cudaMemcpy(ptr_device, temp, sizeof( SimpleMaterial ), cudaMemcpyHostToDevice));
	gpuErrchk( cudaPeekAtLastError() );
#endif
}

void copy(struct SimpleMaterial* pCopy, struct SimpleMaterial* pOrig) {
    unsigned num = pOrig->numIsotopes;
    if( num <=0 ) { num = 1; }

    ctor( pCopy, num);
    pCopy->AtomicWeight = pOrig->AtomicWeight;

    for( unsigned i=0; i<num; ++i ){
        pCopy->fraction[i] = pOrig->fraction[i];
        if( pOrig->xs[i] != 0 ) {
            if( pCopy->xs[i] == 0  ) {
                pCopy->xs[i] = new SimpleCrossSection;  // TODO: memory leak in this case -- need shared ptr
            }
            copy( pCopy->xs[i], pOrig->xs[i] );
        }
    }
}

#ifdef CUDA
__device__ __host__
#endif
unsigned getNumIsotopes(struct SimpleMaterial* ptr ) { return ptr->numIsotopes; }

#ifdef CUDA
__global__ void kernelGetNumIsotopes(SimpleMaterial* pMat, unsigned* results){
	results[0] = getNumIsotopes(pMat);
	return;
}
#endif

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getFraction(struct SimpleMaterial* ptr, unsigned i) {
    return ptr->fraction[i];
}

#ifdef CUDA
__device__ __host__
#endif
void normalizeFractions(struct SimpleMaterial* ptr ){
    gpuFloatType_t total = 0.0f;
    for( unsigned i=0; i<ptr->numIsotopes; ++i){
        total += ptr->fraction[i];
    }
    for( unsigned i=0; i<ptr->numIsotopes; ++i){
        ptr->fraction[i] =  ptr->fraction[i]/total;
    }
    calcAtomicWeight( ptr );
}

#ifdef CUDA
__device__ __host__
#endif
void calcAtomicWeight(struct SimpleMaterial* ptr ){
    gpuFloatType_t total = 0.0f;
    for( unsigned i=0; i<ptr->numIsotopes; ++i){
        if( ptr->xs[i] != 0 ) {
            total += ptr->fraction[i] * ptr->xs[i]->AWR;
        }
    }
    ptr->AtomicWeight = total * gpu_neutron_molar_mass;
}

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getAtomicWeight(struct SimpleMaterial* ptr ) {
    return ptr->AtomicWeight;
}

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getMicroTotalXS(struct SimpleMaterial* ptr, HashLookup* pHash, unsigned HashBin, gpuFloatType_t E){
    gpuFloatType_t total = 0.0f;
    for( unsigned i=0; i<ptr->numIsotopes; ++i){
        if( ptr->xs[i] != 0 ) {
            total += getTotalXS( ptr->xs[i], pHash, HashBin, E) * ptr->fraction[i];
        }
    }
    return total;
}

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getMicroTotalXS(struct SimpleMaterial* ptr, gpuFloatType_t E){
    gpuFloatType_t total = 0.0f;
    for( unsigned i=0; i<ptr->numIsotopes; ++i){
        if( ptr->xs[i] != 0 ) {
            total += getTotalXS( ptr->xs[i], E) * ptr->fraction[i];
        }
    }
    return total;
}

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getTotalXS(struct SimpleMaterial* ptr, HashLookup* pHash, unsigned HashBin, gpuFloatType_t E, gpuFloatType_t density){
    return getMicroTotalXS(ptr, pHash, HashBin, E ) * density * gpu_AvogadroBarn / ptr->AtomicWeight;
}

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getTotalXS(struct SimpleMaterial* ptr, gpuFloatType_t E, gpuFloatType_t density){
    return getMicroTotalXS(ptr, E ) * density * gpu_AvogadroBarn / ptr->AtomicWeight;
}

#ifdef CUDA
__device__ __host__
#endif
void cudaAdd(struct SimpleMaterial* ptr, struct SimpleCrossSection* xs, unsigned index ) {
	ptr->xs[ index ] = xs;
}

#ifdef CUDA
__device__ __host__
#endif
void setID(struct SimpleMaterial* ptr, unsigned index, unsigned id ) {
	MonteRay::setID(ptr->xs[index], id );
}

#ifdef CUDA
__device__ __host__
#endif
int getID(struct SimpleMaterial* ptr, unsigned index ) {
	return MonteRay::getID( ptr->xs[index] );
}

void SimpleMaterialHost::setID( unsigned index, unsigned id) {
	MonteRay::setID(pMat, index, id );
}

int SimpleMaterialHost::getID( unsigned index ) {
	return MonteRay::getID(pMat, index );
}

void SimpleMaterialHost::add(unsigned index,struct SimpleCrossSectionHost& xs, gpuFloatType_t frac ) {
    if( index > getNumIsotopes() ) {
        fprintf(stderr, "SimpleMaterialHost::add -- index > number of allocated isotopes.  %s %d\n", __FILE__, __LINE__);
        exit(1);
    }

    pMat->fraction[index] = frac;
    pMat->xs[index] = xs.getXSPtr();

    calcAWR();

#ifdef CUDA
    isotope_device_ptr_list[index] = xs.xs_device;
#endif
}

unsigned SimpleMaterialHost::launchGetNumIsotopes(void) {
	typedef unsigned type_t;

	type_t* result_device;
	type_t result[1];
	CUDA_CHECK_RETURN( cudaMalloc( &result_device, sizeof( type_t) * 1 ));
	gpuErrchk( cudaPeekAtLastError() );

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	kernelGetNumIsotopes<<<1,1>>>(ptr_device, result_device);
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);

    gpuErrchk( cudaPeekAtLastError() );

	CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(type_t)*1, cudaMemcpyDeviceToHost));
	gpuErrchk( cudaPeekAtLastError() );

	cudaFree( result_device );
	return result[0];
}


#ifndef CUDA
void SimpleMaterialHost::add(unsigned index,struct SimpleCrossSection* xs, gpuFloatType_t frac ) {
    if( index > getNumIsotopes() ) {
        fprintf(stderr, "SimpleMaterialHost::add -- index > number of allocated isotopes.  %s %d\n", __FILE__, __LINE__);
        exit(1);
    }

    pMat->fraction[index] = frac;
    pMat->xs[index] = xs;

    calcAWR();
}
#endif

void SimpleMaterialHost::write(std::ostream& outf) const{
    unsigned realNumIsotopes = 0;
    for( unsigned i=0; i<getNumIsotopes(); ++i ){
        if( pMat->xs[i] != 0 ) {
            ++realNumIsotopes;
        }
    }

    binaryIO::write(outf, realNumIsotopes );

    binaryIO::write(outf, getAtomicWeight() );

    for( unsigned i=0; i<getNumIsotopes(); ++i ){
        if( pMat->xs[i] != 0 ) {
            binaryIO::write(outf, getFraction(i) );
        }
    }
    for( unsigned i=0; i<getNumIsotopes(); ++i ){
        if( pMat->xs[i] != 0 ) {
            SimpleCrossSectionHost xs(1);
            xs.load( pMat->xs[i] );
            xs.write(outf);
        }
    }
}

void SimpleMaterialHost::read(std::istream& infile) {
    unsigned num;
    binaryIO::read(infile, num);
    dtor( pMat );
    ctor( pMat, num );

    binaryIO::read(infile, pMat->AtomicWeight );
    for( unsigned i=0; i<num; ++i ){
        gpuFloatType_t fraction;
        binaryIO::read(infile, fraction );
        pMat->fraction[i] = fraction;
    }

    for( unsigned i=0; i<num; ++i ){
        SimpleCrossSectionHost* xs = new SimpleCrossSectionHost(1); // TODO: need shared ptr here
        xs->read( infile );
        add(i, *xs, pMat->fraction[i]);
    }
}

void SimpleMaterialHost::load(struct SimpleMaterial* ptrMat ) {
    copy( pMat, ptrMat);
}

}
