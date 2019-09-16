#include "MonteRayMaterial.hh"

#include <fstream>

#include "MonteRayDefinitions.hh"
#include "GPUErrorCheck.hh"
#include "MonteRayConstants.hh"
#include "MonteRay_binaryIO.hh"
#include "HashLookup.hh"
#include "MonteRayCrossSection.hh"
#include "MonteRayMemory.hh"
#include "MonteRayParallelAssistant.hh"

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
    pCopy->fraction = (gpuFloatType_t*) MONTERAYDEVICEALLOC( allocSize, std::string("MonteRayMaterial::fraction") );

    // MonteRayCrossSections
    allocSize = sizeof(MonteRayCrossSection*)*num;
    pCopy->xs = (MonteRayCrossSection**) MONTERAYDEVICEALLOC( allocSize, std::string("MonteRayMaterial::xs") );
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
    MonteRayDeviceFree( ptr->fraction );
    MonteRayDeviceFree( ptr->xs );
#endif
}

MonteRayMaterialHost::MonteRayMaterialHost(unsigned numIsotopes) {
    pMat = new MonteRayMaterial;
    ctor(pMat, numIsotopes );

    isotope_device_ptr_list = (MonteRayCrossSection**) malloc( sizeof(MonteRayCrossSection* )*numIsotopes );
    for( unsigned i=0; i< numIsotopes; ++i ){
        isotope_device_ptr_list[i] = 0;
    }
}

MonteRayMaterialHost::~MonteRayMaterialHost() {
    //std::cout << "Debug: MonteRayMaterialHost::~MonteRayMaterialHost()\n" << std::endl;
    if( pMat ) {
        dtor( pMat );
        delete pMat;
        pMat = 0;
    }

    for( auto itr = ownedCrossSections.begin(); itr != ownedCrossSections.end(); ++itr) {
        delete (*itr);
    }
    ownedCrossSections.clear();

    if( ptr_device ) MonteRayDeviceFree( ptr_device );
    ptr_device = nullptr;

    if( temp ) {
#ifdef __CUDACC__
        cudaDtor( temp );
        delete temp;
#endif
    }
    free( isotope_device_ptr_list );
}

void MonteRayMaterialHost::copyToGPU(void) {
#ifdef __CUDACC__
    if( ! MonteRay::isWorkGroupMaster() ) return;
    cudaCopyMade = true;

    if( temp ) {
        cudaDtor( temp );
        delete temp;
    }
    temp = new MonteRayMaterial;

    copy(temp, pMat);

    unsigned num = pMat->numIsotopes;

    temp->numIsotopes = pMat->numIsotopes;
    temp->AtomicWeight = pMat->AtomicWeight;

    // allocate target struct
    if( ptr_device ) {
        MonteRayDeviceFree(ptr_device);
    }
    ptr_device = (MonteRayMaterial*) MONTERAYDEVICEALLOC( sizeof( MonteRayMaterial), std::string("MonteRayMaterialHost::ptr_device") );

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

void MonteRayMaterialHost::copyOwnedCrossSectionsToGPU(void) {
#ifdef __CUDACC__
    for( auto itr = ownedCrossSections.begin(); itr != ownedCrossSections.end(); ++itr) {
        (*itr)->copyToGPU();
    }
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

CUDA_CALLABLE_KERNEL  kernelGetNumIsotopes(MonteRayMaterial* pMat, unsigned* results){
    results[0] = getNumIsotopes(pMat);
    return;
}

CUDA_CALLABLE_KERNEL  kernelGetTotalXS(MonteRayMaterial* pMat, gpuFloatType_t E, gpuFloatType_t density,  gpuFloatType_t* results){
    results[0] = getTotalXS(pMat, E, density);
    return;
}

CUDA_CALLABLE_KERNEL  kernelGetTotalXS(MonteRayMaterial* pMat, const HashLookup* pHash, gpuFloatType_t E, gpuFloatType_t density,  gpuFloatType_t* results){
    unsigned HashBin = getHashBin(pHash,E);
    results[0] = getTotalXS(pMat, pHash, HashBin, E, density);
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
//    printf("Debug: MonteRayMaterial.cu :getTotalXS(const MonteRayMaterialList* ptr, const HashLookup* pHash, unsigned HashBin, gpuFloatType_t E, gpuFloatType_t density)\n");
//    printf("Debug: ptr=%p\n", ptr);
//    printf("Debug: pHash=%p\n", pHash);
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

    result_device = (type_t*) MONTERAYDEVICEALLOC( sizeof( type_t) * 1, std::string("MonteRayMaterialHost::launchGetNumIsotopes::result_device") );

    cudaEvent_t sync;
    cudaEventCreate(&sync);
    kernelGetNumIsotopes<<<1,1>>>(ptr_device, result_device);
    gpuErrchk( cudaPeekAtLastError() );
    cudaEventRecord(sync, 0);
    cudaEventSynchronize(sync);

    CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(type_t)*1, cudaMemcpyDeviceToHost));

    MonteRayDeviceFree( result_device );
#else
    kernelGetNumIsotopes(pMat, result);
#endif

    return result[0];
}

gpuFloatType_t MonteRayMaterialHost::launchGetTotalXS(gpuFloatType_t E, gpuFloatType_t density ) {
    gpuFloatType_t result[1];

#ifdef __CUDACC__
    gpuFloatType_t* result_device;

    result_device = (gpuFloatType_t*) MONTERAYDEVICEALLOC( sizeof( gpuFloatType_t) * 1, std::string("launchGetTotalXS::result_device") );

    cudaEvent_t sync;
    cudaEventCreate(&sync);
    kernelGetTotalXS<<<1,1>>>( ptr_device, E, density, result_device);
    gpuErrchk( cudaPeekAtLastError() );
    cudaEventRecord(sync, 0);
    cudaEventSynchronize(sync);

    CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(gpuFloatType_t)*1, cudaMemcpyDeviceToHost));

    MonteRayDeviceFree( result_device );
#else
    kernelGetTotalXS( pMat, E, density, result);
#endif
    return result[0];
}

gpuFloatType_t MonteRayMaterialHost::launchGetTotalXSViaHash(HashLookupHost& hash, gpuFloatType_t E, gpuFloatType_t density ) {
    gpuFloatType_t result[1];

#ifdef __CUDACC__
    gpuFloatType_t* result_device;

    result_device = (gpuFloatType_t*) MONTERAYDEVICEALLOC( sizeof( gpuFloatType_t) * 1, std::string("launchGetTotalXS::result_device") );

    cudaEvent_t sync;
    cudaEventCreate(&sync);
    hash.copyToGPU();
    kernelGetTotalXS<<<1,1>>>( ptr_device, hash.ptr_device, E, density, result_device);
    gpuErrchk( cudaPeekAtLastError() );
    cudaEventRecord(sync, 0);
    cudaEventSynchronize(sync);

    CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(gpuFloatType_t)*1, cudaMemcpyDeviceToHost));

    MonteRayDeviceFree( result_device );
#else
    kernelGetTotalXS( pMat, hash.getPtr(), E, density, result);
#endif
    return result[0];
}


void MonteRayMaterialHost::add(unsigned index,struct MonteRayCrossSection* xs, gpuFloatType_t frac ) {
    if( index > getNumIsotopes() ) {
        fprintf(stderr, "MonteRayMaterialHost::add -- index > number of allocated isotopes.  %s %d\n", __FILE__, __LINE__);
        exit(1);
    }

    pMat->fraction[index] = frac;
    pMat->xs[index] = xs;

    calcAWR();
}

void MonteRayMaterialHost::readFromFile( const std::string& filename, HashLookupHost* pHash ) {
    std::ifstream infile;
    if( infile.is_open() ) {
        infile.close();
    }
    infile.open( filename.c_str(), std::ios::binary | std::ios::in);

    if( ! infile.is_open() ) {
        fprintf(stderr, "Error:  MonteRayMaterialHost::readFromFile -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
        throw std::runtime_error("MonteRayMaterialHost::readFromFile -- Failure to open file" );
    }
    assert( infile.good() );
    infile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
    read(infile, pHash);
    infile.close();
}

void MonteRayMaterialHost::writeToFile( const std::string& filename ) const {
    std::ofstream outfile;

    outfile.open( filename.c_str(), std::ios::binary | std::ios::out);
    if( ! outfile.is_open() ) {
        fprintf(stderr, "MonteRayMaterialHost::writeToFile -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
        throw std::runtime_error("MonteRayMaterialHost::writeToFile -- Failure to open file" );
    }
    assert( outfile.good() );
    outfile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
    write( outfile );
    outfile.close();
}

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

void MonteRayMaterialHost::read(std::istream& infile, HashLookupHost* pHash) {
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
        ownedCrossSections.push_back( new MonteRayCrossSectionHost(1) ); // TODO: need shared ptr here
        ownedCrossSections.back()->read( infile );
        add(i, *(ownedCrossSections.back()), pMat->fraction[i]);
        if( pHash ) {
            pHash->addIsotope( ownedCrossSections.back() );
        }
    }
}

void MonteRayMaterialHost::load(struct MonteRayMaterial* ptrMat ) {
    copy( pMat, ptrMat);
}

gpuFloatType_t
MonteRayMaterialHost::getMicroTotalXS(HashLookup* pHash, gpuFloatType_t E ){
    unsigned HashBin = getHashBin(pHash, E);
    return MonteRay::getMicroTotalXS(pMat, pHash, HashBin, E);
}

gpuFloatType_t
MonteRayMaterialHost::getTotalXS(HashLookup* pHash, gpuFloatType_t E, gpuFloatType_t density ){
    unsigned HashBin = getHashBin(pHash, E);
    return MonteRay::getTotalXS(pMat, pHash, HashBin, E, density);
}

}
