#include "MonteRayMaterial.hh"

#include "GPUErrorCheck.hh"
#include "MonteRayConstants.hh"
#include "MonteRay_binaryIO.hh"

namespace MonteRay{

void ctor(struct MonteRayMaterial* ptr, unsigned num) {
    if( num <=0 ) { num = 1; }
    ptr->numIsotopes = num;
    ptr->AtomicWeight = 0.0;

    unsigned allocSize = sizeof(gpuFloatType_t)*num;

    ptr->fraction  = (gpuFloatType_t*) malloc( allocSize);
    if(ptr->fraction == 0) abort ();

    allocSize = sizeof(struct MonteRayCrossSection*)*num;
    ptr->xs = (struct MonteRayCrossSection**) malloc( allocSize );
    if(ptr->xs == 0) abort ();

    for( unsigned i=0; i<num; ++i ){
        ptr->fraction[i] = 0.0;

        ptr->xs[i] = 0; // initialize to null
    }
}


void cudaCtor(MonteRayMaterial* pCopy, unsigned num) {
#ifdef __CUDACC__
	pCopy->numIsotopes = num;
	pCopy->AtomicWeight = 0.0;

	// fractions
	unsigned allocSize = sizeof(gpuFloatType_t)*num;
	CUDA_CHECK_RETURN( cudaMalloc(&pCopy->fraction, allocSize ));

    // MonteRayCrossSections
	allocSize = sizeof(MonteRayCrossSection*)*num;
	CUDA_CHECK_RETURN( cudaMalloc(&pCopy->xs, allocSize ));
#endif
}

void cudaCtor(struct MonteRayMaterial* pCopy, struct MonteRayMaterial* pOrig){
	unsigned num = pOrig->numIsotopes;
	cudaCtor( pCopy, num);

	pCopy->AtomicWeight = pOrig->AtomicWeight;
}


void dtor(struct MonteRayMaterial* ptr){
    if( ptr->fraction != 0 ) {
        free(ptr->fraction);
        ptr->fraction = 0;
    }
    if( ptr->xs != 0 ) {
        free(ptr->xs);
        ptr->xs = 0;
    }
}

void cudaDtor(MonteRayMaterial* ptr) {
#ifdef __CUDACC__
	cudaFree( ptr->fraction );
	cudaFree( ptr->xs );
#endif
}

MonteRayMaterialHost::MonteRayMaterialHost(unsigned numIsotopes) {
    pMat = new MonteRayMaterial;
    ctor(pMat, numIsotopes );
    cudaCopyMade = false;
    ptr_device = NULL;
    temp = NULL;

    isotope_device_ptr_list = (MonteRayCrossSection**) malloc( sizeof(MonteRayCrossSection* )*numIsotopes );
    for( unsigned i=0; i< numIsotopes; ++i ){
    	isotope_device_ptr_list[i] = 0;
    }
}

MonteRayMaterialHost::~MonteRayMaterialHost() {
    if( pMat != 0 ) {
        dtor( pMat );
        delete pMat;
        pMat = 0;
    }

    if( cudaCopyMade ) {
#ifdef __CUDACC__
    	cudaFree( ptr_device );
    	cudaDtor( temp );
    	delete temp;
#endif
    }
    free( isotope_device_ptr_list );
}

void MonteRayMaterialHost::copyToGPU(void) {
#ifdef __CUDACC__
	cudaCopyMade = true;
	temp = new MonteRayMaterial;

    copy(temp, pMat);

	unsigned num = pMat->numIsotopes;

	temp->numIsotopes = pMat->numIsotopes;
	temp->AtomicWeight = pMat->AtomicWeight;

	// allocate target struct
	CUDA_CHECK_RETURN( cudaMalloc(&ptr_device, sizeof( MonteRayMaterial) ));

	// allocate target dynamic memory
	cudaCtor( temp, pMat);

	unsigned allocSize = sizeof(gpuFloatType_t)*num;
	CUDA_CHECK_RETURN( cudaMemcpy(temp->fraction, pMat->fraction, allocSize, cudaMemcpyHostToDevice));

	allocSize = sizeof(MonteRayCrossSection*)*num;
	CUDA_CHECK_RETURN( cudaMemcpy(temp->xs, isotope_device_ptr_list, allocSize, cudaMemcpyHostToDevice));

	// copy data
	CUDA_CHECK_RETURN( cudaMemcpy(ptr_device, temp, sizeof( MonteRayMaterial ), cudaMemcpyHostToDevice));
#endif
}

void copy(struct MonteRayMaterial* pCopy, struct MonteRayMaterial* pOrig) {
    unsigned num = pOrig->numIsotopes;
    if( num <=0 ) { num = 1; }

    ctor( pCopy, num);
    pCopy->AtomicWeight = pOrig->AtomicWeight;

    for( unsigned i=0; i<num; ++i ){
        pCopy->fraction[i] = pOrig->fraction[i];
        if( pOrig->xs[i] != 0 ) {
            if( pCopy->xs[i] == 0  ) {
                pCopy->xs[i] = new MonteRayCrossSection;  // TODO: memory leak in this case -- need shared ptr
            }
            copy( pCopy->xs[i], pOrig->xs[i] );
        }
    }
}

CUDA_CALLABLE_MEMBER
unsigned getNumIsotopes(struct MonteRayMaterial* ptr ) { return ptr->numIsotopes; }

CUDA_CALLABLE_KERNEL void kernelGetNumIsotopes(MonteRayMaterial* pMat, unsigned* results){
	results[0] = getNumIsotopes(pMat);
	return;
}

CUDA_CALLABLE_MEMBER
gpuFloatType_t getFraction(struct MonteRayMaterial* ptr, unsigned i) {
    return ptr->fraction[i];
}

CUDA_CALLABLE_MEMBER
void normalizeFractions(struct MonteRayMaterial* ptr ){
    gpuFloatType_t total = 0.0f;
    for( unsigned i=0; i<ptr->numIsotopes; ++i){
        total += ptr->fraction[i];
    }
    for( unsigned i=0; i<ptr->numIsotopes; ++i){
        ptr->fraction[i] =  ptr->fraction[i]/total;
    }
    calcAtomicWeight( ptr );
}

CUDA_CALLABLE_MEMBER
void calcAtomicWeight(struct MonteRayMaterial* ptr ){
    gpuFloatType_t total = 0.0f;
    for( unsigned i=0; i<ptr->numIsotopes; ++i){
        if( ptr->xs[i] != 0 ) {
            total += ptr->fraction[i] * ptr->xs[i]->AWR;
        }
    }
    ptr->AtomicWeight = total * gpu_neutron_molar_mass;
}

CUDA_CALLABLE_MEMBER
gpuFloatType_t getAtomicWeight(struct MonteRayMaterial* ptr ) {
    return ptr->AtomicWeight;
}

CUDA_CALLABLE_MEMBER
gpuFloatType_t getMicroTotalXS(const struct MonteRayMaterial* ptr, const HashLookup* pHash, unsigned HashBin, gpuFloatType_t E){
	const bool debug = false;

	if( debug ) printf("Debug: MonteRayMaterials::getMicroTotalXS(const struct MonteRayMaterial* ptr, const HashLookup* pHash, unsigned HashBin, gpuFloatType_t E)\n");

	gpuFloatType_t total = 0.0f;
	if( debug ) printf("Debug: MonteRayMaterials::getMicroTotalXS -- ptr->numIsotopes = %d\n", ptr->numIsotopes);
    for( unsigned i=0; i<ptr->numIsotopes; ++i){
    	if( debug ) printf("Debug: MonteRayMaterials::getMicroTotalXS i = %d \n", i);
        if( ptr->xs[i] != 0 ) {
            total += getTotalXS( ptr->xs[i], pHash, HashBin, E) * ptr->fraction[i];
        }
    }
    if( debug ) printf("Debug: MonteRayMaterials::getMicroTotalXS -- total microscopic xsec = %f \n", total );
    return total;
}

CUDA_CALLABLE_MEMBER
gpuFloatType_t getMicroTotalXS(const struct MonteRayMaterial* ptr, gpuFloatType_t E){
    gpuFloatType_t total = 0.0f;
    for( unsigned i=0; i<ptr->numIsotopes; ++i){
        if( ptr->xs[i] != 0 ) {
            total += getTotalXS( ptr->xs[i], E) * ptr->fraction[i];
        }
    }
    return total;
}

CUDA_CALLABLE_MEMBER
gpuFloatType_t getTotalXS(const struct MonteRayMaterial* ptr, const HashLookup* pHash, unsigned HashBin, gpuFloatType_t E, gpuFloatType_t density){
//	printf("Debug: MonteRayMaterials::getTotalXS(const struct MonteRayMaterial* ptr, const HashLookup* pHash, unsigned HashBin, gpuFloatType_t E, gpuFloatType_t density)\n");
	return getMicroTotalXS(ptr, pHash, HashBin, E ) * density * gpu_AvogadroBarn / ptr->AtomicWeight;
}

CUDA_CALLABLE_MEMBER
gpuFloatType_t getTotalXS(const struct MonteRayMaterial* ptr, gpuFloatType_t E, gpuFloatType_t density){
    return getMicroTotalXS(ptr, E ) * density * gpu_AvogadroBarn / ptr->AtomicWeight;
}

CUDA_CALLABLE_MEMBER
void cudaAdd(struct MonteRayMaterial* ptr, struct MonteRayCrossSection* xs, unsigned index ) {
	ptr->xs[ index ] = xs;
}

CUDA_CALLABLE_MEMBER
void setID(struct MonteRayMaterial* ptr, unsigned index, unsigned id ) {
	MonteRay::setID(ptr->xs[index], id );
}

CUDA_CALLABLE_MEMBER
int getID(struct MonteRayMaterial* ptr, unsigned index ) {
	return MonteRay::getID( ptr->xs[index] );
}

void MonteRayMaterialHost::setID( unsigned index, unsigned id) {
	MonteRay::setID(pMat, index, id );
}

int MonteRayMaterialHost::getID( unsigned index ) {
	return MonteRay::getID(pMat, index );
}

void MonteRayMaterialHost::add(unsigned index,struct MonteRayCrossSectionHost& xs, gpuFloatType_t frac ) {
    if( index > getNumIsotopes() ) {
        fprintf(stderr, "MonteRayMaterialHost::add -- index > number of allocated isotopes.  %s %d\n", __FILE__, __LINE__);
        exit(1);
    }

    pMat->fraction[index] = frac;
    pMat->xs[index] = xs.getXSPtr();

    calcAWR();

#ifdef __CUDACC__
    isotope_device_ptr_list[index] = xs.xs_device;
#endif
}

unsigned MonteRayMaterialHost::launchGetNumIsotopes(void) {
	typedef unsigned type_t;
	type_t result[1];

#ifdef __CUDACC__
	type_t* result_device;

	CUDA_CHECK_RETURN( cudaMalloc( &result_device, sizeof( type_t) * 1 ));

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	kernelGetNumIsotopes<<<1,1>>>(ptr_device, result_device);
    gpuErrchk( cudaPeekAtLastError() );
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);

	CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(type_t)*1, cudaMemcpyDeviceToHost));

	cudaFree( result_device );
#else
	kernelGetNumIsotopes(pMat, result);
#endif

	return result[0];
}


#ifndef __CUDACC__
void MonteRayMaterialHost::add(unsigned index,struct MonteRayCrossSection* xs, gpuFloatType_t frac ) {
    if( index > getNumIsotopes() ) {
        fprintf(stderr, "MonteRayMaterialHost::add -- index > number of allocated isotopes.  %s %d\n", __FILE__, __LINE__);
        exit(1);
    }

    pMat->fraction[index] = frac;
    pMat->xs[index] = xs;

    calcAWR();
}
#endif

void MonteRayMaterialHost::write(std::ostream& outf) const{
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
            MonteRayCrossSectionHost xs(1);
            xs.load( pMat->xs[i] );
            xs.write(outf);
        }
    }
}

void MonteRayMaterialHost::read(std::istream& infile) {
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
        MonteRayCrossSectionHost* xs = new MonteRayCrossSectionHost(1); // TODO: need shared ptr here
        xs->read( infile );
        add(i, *xs, pMat->fraction[i]);
    }
}

void MonteRayMaterialHost::load(struct MonteRayMaterial* ptrMat ) {
    copy( pMat, ptrMat);
}

}
