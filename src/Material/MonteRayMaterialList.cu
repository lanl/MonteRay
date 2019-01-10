#include "MonteRayMaterialList.hh"

#include <fstream>

#include "MonteRayMaterial.hh"
#include "GPUErrorCheck.hh"
#include "MonteRay_binaryIO.hh"
#include "HashLookup.hh"
#include "MonteRayMemory.hh"

namespace MonteRay{

void ctor(MonteRayMaterialList* ptr, unsigned num ) {
    if( num <=0 ) { num = 1; }
    ptr->numMaterials = num;

    unsigned allocSize = sizeof(unsigned)*num;

    if( ptr->materialID ) {
        free(ptr->materialID);
    }
    ptr->materialID  = (unsigned*) malloc( allocSize);
    if(ptr->materialID == NULL) abort ();

    allocSize = sizeof(MonteRayMaterial*)*num;

    if( ptr->materials ) {
        free( ptr->materials );
    }
    ptr->materials = (MonteRayMaterial**) malloc( allocSize );
    if(ptr->materials == NULL) abort ();

    for( unsigned i=0; i<num; ++i ){
        ptr->materialID[i] = 0;

        ptr->materials[i] = 0; // set to null ptr
    }
}


void cudaCtor(MonteRayMaterialList* pCopy, unsigned num) {
#ifdef __CUDACC__
    pCopy->numMaterials = num;

    // materialID
    unsigned allocSize = sizeof(unsigned)*num;
    if( pCopy->materialID ) {
        MonteRayDeviceFree( pCopy->materialID );
    }
    pCopy->materialID = (unsigned*) MONTERAYDEVICEALLOC( allocSize, std::string("MonteRayMaterialList::materialID") );

    // materials
    allocSize = sizeof(MonteRayMaterial*)*num;
    if( pCopy->materials ) {
        MonteRayDeviceFree( pCopy->materials );
    }
    pCopy->materials = (MonteRayMaterial**) MONTERAYDEVICEALLOC( allocSize, std::string("MonteRayMaterialList::materials") );
#endif
}

void cudaCtor(struct MonteRayMaterialList* pCopy, struct MonteRayMaterialList* pOrig){
#ifdef __CUDACC__
    unsigned num = pOrig->numMaterials;
    cudaCtor( pCopy, num);
#endif
}

void dtor(MonteRayMaterialList* ptr) {
    if( ptr->materialID != 0 ) {
        free( ptr->materialID );
        ptr->materialID = 0;
    }

    if( ptr->materials != 0 ) {
        free( ptr->materials );
        ptr->materials = 0;
    }
}


void cudaDtor(MonteRayMaterialList* ptr) {
#ifdef __CUDACC__
    MonteRayDeviceFree( ptr->materialID );
    MonteRayDeviceFree( ptr->materials );
#endif
}

MonteRayMaterialListHost::MonteRayMaterialListHost(unsigned num, unsigned maxNumIsotopes, unsigned nBins) {
    if( num > MAXNUMMATERIALS ) {
        std::stringstream msg;
        msg << "MonteRay -- MonteRayMaterialListHost::MonteRayMaterialListHost.\n" <<
               "The requested number of materials(" << num << "), " <<
               "exceeds the maximum number of materials, MAXNUMMATERIALS = " << MAXNUMMATERIALS <<" !\n"
                << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "\n\n";
        throw std::runtime_error( msg.str() );
    }

    nBinsHash = nBins;
    pHash = new HashLookupHost( maxNumIsotopes, nBins);
    reallocate(num);
}

void
MonteRayMaterialListHost::reallocate(unsigned num) {
    numMats = num;

    if( pMatList ) {
        delete pMatList;
    }
    pMatList = new MonteRayMaterialList;

    ctor( pMatList, num);

    if( temp ) {
        delete temp;
    }
    temp = nullptr;

    if( ptr_device ) {
        MonteRayDeviceFree( ptr_device );
    }
    ptr_device = nullptr;

    cudaCopyMade = false;

#ifdef __CUDACC__
    if( material_device_ptr_list ) {
        free(material_device_ptr_list);
        material_device_ptr_list = nullptr;
    }
    material_device_ptr_list = (MonteRayMaterial**) malloc( sizeof(MonteRayMaterial* )*num );
    for( unsigned i=0; i< num; ++i ){
        material_device_ptr_list[i] = 0;
    }
#endif
}

MonteRayMaterialListHost::~MonteRayMaterialListHost() {
    dtor( pMatList );
    delete pMatList;
    delete pHash;
#ifdef __CUDACC__
    if( cudaCopyMade ) {
        cudaDtor( temp );
        delete temp;
        MonteRayDeviceFree( ptr_device );
    }
    free( material_device_ptr_list );
#endif

    for( auto itr = ownedMaterials.begin(); itr != ownedMaterials.end(); ++itr) {
        delete (*itr);
    }
    ownedMaterials.clear();
}

void MonteRayMaterialListHost::copyToGPU(void) {
#ifdef __CUDACC__
    pHash->copyToGPU();

    for( auto itr = ownedMaterials.begin(); itr != ownedMaterials.end(); ++itr) {
       (*itr)->copyToGPU();
    }

    cudaCopyMade = true;
    temp = new MonteRayMaterialList;
    unsigned num = pMatList->numMaterials;

    // allocate target struct
    ptr_device = (MonteRayMaterialList*) MONTERAYDEVICEALLOC( sizeof( MonteRayMaterialList ), std::string("MonteRayMaterialListHost::ptr_device") );

    // allocate target dynamic memory
    cudaCtor( temp, pMatList);

    unsigned allocSize = sizeof(unsigned)*num;
    CUDA_CHECK_RETURN( cudaMemcpy(temp->materialID, pMatList->materialID, allocSize, cudaMemcpyHostToDevice));

    allocSize = sizeof(MonteRayCrossSection*)*num;
    CUDA_CHECK_RETURN( cudaMemcpy(temp->materials, material_device_ptr_list, allocSize, cudaMemcpyHostToDevice));

    // copy data
    CUDA_CHECK_RETURN( cudaMemcpy(ptr_device, temp, sizeof( MonteRayMaterialList ), cudaMemcpyHostToDevice));

    for( auto itr = ownedMaterials.begin(); itr != ownedMaterials.end(); ++itr) {
          (*itr)->copyOwnedCrossSectionsToGPU();
       }
#endif
}

void copy(MonteRayMaterialList* pCopy, const MonteRayMaterialList* const pOrig ) {
    unsigned num = pOrig->numMaterials;
    if( num <=0 ) { num = 1; }

    ctor( pCopy, num);

    for( unsigned i=0; i<num; ++i ){
        pCopy->materialID[i] = pOrig->materialID[i];
        if( pOrig->materials[i] != 0 ) {
            if( pCopy->materials[i] == 0 ) {
                pCopy->materials[i] = new MonteRayMaterial; // TODO: memory leak -- need shared ptr
            }
            copy( pCopy->materials[i], pOrig->materials[i]);
        }
    }
}

CUDA_CALLABLE_MEMBER
unsigned getNumberMaterials(MonteRayMaterialList* ptr) {
    return ptr->numMaterials;
}

CUDA_CALLABLE_MEMBER
unsigned getMaterialID(MonteRayMaterialList* ptr, unsigned i ) {
    return ptr->materialID[i];
}

CUDA_CALLABLE_MEMBER
MonteRayMaterial* getMaterial(MonteRayMaterialList* ptr, unsigned i ){
    return ptr->materials[i];
}

CUDA_CALLABLE_MEMBER
const MonteRayMaterial* getMaterial(const MonteRayMaterialList* ptr, unsigned i ){
    return ptr->materials[i];
}

CUDA_CALLABLE_MEMBER
gpuFloatType_t getTotalXS(const MonteRayMaterialList* ptr, unsigned i, const HashLookup* pHash, unsigned HashBin, gpuFloatType_t E, gpuFloatType_t density) {
//    printf("Debug: MonteRayMaterials::getTotalXS(const MonteRayMaterialList* ptr, unsigned i, const HashLookup* pHash, unsigned HashBin, gpuFloatType_t E, gpuFloatType_t density)\n");
//    printf("Debug: ptr=%p\n", ptr);
//    printf("Debug: pHash=%p\n", pHash);
//    printf("Debug: i=%u\n", i);
//    printf("Debug: getMaterial(ptr,i)=%p\n", getMaterial(ptr,i));
    return getTotalXS( getMaterial(ptr,i) , pHash, HashBin, E, density );
}

CUDA_CALLABLE_MEMBER
gpuFloatType_t getTotalXS(const MonteRayMaterialList* ptr, unsigned i, gpuFloatType_t E, gpuFloatType_t density) {
    return getTotalXS( getMaterial(ptr,i), E, density );
}

CUDA_CALLABLE_MEMBER
unsigned materialIDtoIndex(MonteRayMaterialList* ptr, unsigned id ) {
    for( unsigned i=0; i < ptr->numMaterials; ++i ){
        if( id == ptr->materialID[i] ) {
            return i;
        }
    }

    printf("Error: materialIDtoIndex -- id=%d not found.  %s %d\n", id, __FILE__, __LINE__);
    ABORT( "materialIDtoIndex" );
    return 0;
}

CUDA_CALLABLE_KERNEL  kernelGetTotalXS(struct MonteRayMaterialList* pMatList, unsigned matIndex, const HashLookup* pHash, unsigned HashBin, gpuFloatType_t E, gpuFloatType_t density, gpuFloatType_t* results){
//    printf("Debug: kernelGetTotalXS \n");
//    printf("Debug: pMatList=%p\n", pMatList);
//    printf("Debug: pHash=%p\n", pHash);
    results[0] = getTotalXS(pMatList, matIndex, pHash, HashBin, E, density);
    return;
}

gpuFloatType_t MonteRayMaterialListHost::launchGetTotalXS(unsigned i, gpuFloatType_t E, gpuFloatType_t density) const {

    typedef gpuFloatType_t type_t;
    type_t result[1];
    unsigned HashBin = getHashBin( pHash->getPtr(), E);

#ifdef __CUDACC__
    type_t* result_device;

    result_device = (type_t*) MONTERAYDEVICEALLOC( sizeof( type_t) * 1, std::string("MonteRayMaterialListHost::launchGetTotalXS::result_device") );

    cudaEvent_t sync;
    cudaEventCreate(&sync);

//    printf("Debug: MonteRayMaterialListHost::launchGetTotalXS \n");
//    printf("Debug: ptr_device=%p\n", ptr_device);
//    printf("Debug: pHash->getPtrDevice()=%p\n", pHash->getPtrDevice());
    kernelGetTotalXS<<<1,1>>>(ptr_device, i, pHash->getPtrDevice(), HashBin, E, density, result_device);
    gpuErrchk( cudaPeekAtLastError() );
    cudaEventRecord(sync, 0);
    cudaEventSynchronize(sync);

    CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(type_t)*1, cudaMemcpyDeviceToHost));

    MonteRayDeviceFree( result_device );

#else
    kernelGetTotalXS(pMatList, i, pHash->getPtr(), HashBin, E, density, result);
#endif
    return result[0];
}


void MonteRayMaterialListHost::add( unsigned index, MonteRayMaterialHost& mat, unsigned id) {
    if( index >= getNumberMaterials() ) {
        fprintf(stderr, "MonteRayMaterialListHost::add -- index > number of allocated materials.  %s %d\n", __FILE__, __LINE__);
        exit(1);
    }

    pMatList->materialID[index] = id;
    pMatList->materials[index] = mat.getPtr();

#ifdef __CUDACC__
    material_device_ptr_list[index] = mat.ptr_device;
#endif
    for( unsigned i = 0; i < mat.getNumIsotopes(); ++i ){
        int currentID = mat.getID(i);
        if( currentID < 0 ) {
            pHash->addIsotope( mat.getPtr()->xs[i] );
        }
    }
}


void MonteRayMaterialListHost::add( unsigned index, MonteRayMaterial* mat, unsigned id) {
    if( index >= getNumberMaterials() ) {
        fprintf(stderr, "MonteRayMaterialListHost::add -- index > number of allocated materials.  %s %d\n", __FILE__, __LINE__);
        exit(1);
    }

    pMatList->materialID[index] = id;
    pMatList->materials[index] = mat;

    for( auto i = 0; i < getNumIsotopes(mat); ++i ){
        int currentID = getID(mat, i);
        if( currentID < 0 ) {
            pHash->addIsotope( mat->xs[i] );
        }
    }
}


void MonteRayMaterialListHost::writeToFile( const std::string& filename) const {
    std::ofstream out;
    out.open( filename.c_str(), std::ios::binary | std::ios::out);
    write( out );
    out.close();
}

void MonteRayMaterialListHost::readFromFile( const std::string& filename) {
    std::ifstream in;
    in.open( filename.c_str(), std::ios::binary | std::ios::in);
    if( ! in.good() ) {
        throw std::runtime_error( "MonteRayMaterialListHost::readFromFile -- can't open file for reading" );
    }
    read( in );
    in.close();
}

void MonteRayMaterialListHost::write(std::ostream& outfile) const {
    unsigned version = 0;
    binaryIO::write(outfile, version );
    binaryIO::write(outfile, pMatList->numMaterials );

    MONTERAY_VERIFY( pMatList->materialID, "MonteRayMaterialListHost::write -- no materials loaded" )
    for( unsigned i=0; i<pMatList->numMaterials; ++i ){
       binaryIO::write(outfile, pMatList->materialID[i] );
    }

    // count the number of total isotopes - may be duplicates.
    // need to hash bin
    unsigned maxNumIsotopes = 0;
    for( unsigned i=0; i<pMatList->numMaterials; ++i ){
        if( pMatList->materials[i] != 0 ) {
            maxNumIsotopes += pMatList->materials[i]->numIsotopes;
        }
    }
    binaryIO::write(outfile, maxNumIsotopes );
    binaryIO::write(outfile, nBinsHash );

    for( unsigned i=0; i<pMatList->numMaterials; ++i ){
        MonteRayMaterialHost mat(1);
        if( getPtr()->materials[i] != 0 ) {
            mat.load( pMatList->materials[i] );
            mat.write( outfile );
        }
    }
}

void MonteRayMaterialListHost::read(std::istream& infile) {
    unsigned version;
    binaryIO::read(infile, version );

    unsigned num;
    binaryIO::read(infile, num);

    reallocate( num );

    for( unsigned i=0; i<num; ++i ){
        binaryIO::read(infile, pMatList->materialID[i] );
    }

    // need to construct a new hash bin
    unsigned maxNumIsotopes;
    binaryIO::read(infile, maxNumIsotopes );
    binaryIO::read(infile, nBinsHash );
    if( pHash ) {
        delete pHash;
    }
    pHash = new HashLookupHost( maxNumIsotopes, nBinsHash);

    for( unsigned i=0; i<num; ++i ){
        ownedMaterials.push_back( new MonteRayMaterialHost(1) ); // TODO: need shared ptr here
        ownedMaterials.back()->read(infile );
        ownedMaterials.back()->copyToGPU();
        add( i, *(ownedMaterials.back()), pMatList->materialID[i] );
    }

}

gpuFloatType_t
MonteRayMaterialListHost::getTotalXS(unsigned i, gpuFloatType_t E, gpuFloatType_t density, ParticleType_t ParticleType) const {
    if( ParticleType == neutron ) {
        unsigned index = pHash->getHashBin(E);
        return MonteRay::getTotalXS( pMatList, i, pHash->getPtr(), index, E, density);
    } else {
        return MonteRay::getTotalXS( pMatList, i, E, density);
    }
}

unsigned
MonteRayMaterialListHost::materialIDtoIndex(unsigned id) const {
    for( unsigned i=0; i < getNumberMaterials(); ++i ){
        if( id == getMaterialID(i) ) {
            return i;
        }
    }
    std::stringstream msg;
    msg << "Can't find index of material ID!\n";
    msg << "Material ID = " << id << "\n";
    msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRayMaterialList::materialIDtoIndex" << "\n\n";
    throw std::runtime_error( msg.str() );
}

}
