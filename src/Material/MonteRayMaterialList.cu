#include "MonteRayMaterialList.hh"

#include "MonteRayMaterial.hh"
#include "GPUErrorCheck.hh"
#include "MonteRay_binaryIO.hh"
#include "HashLookup.hh"

namespace MonteRay{

void ctor(MonteRayMaterialList* ptr, unsigned num ) {
    if( num <=0 ) { num = 1; }
    ptr->numMaterials = num;

    unsigned allocSize = sizeof(unsigned)*num;

    ptr->materialID  = (unsigned*) malloc( allocSize);
    if(ptr->materialID == 0) abort ();

    allocSize = sizeof(MonteRayMaterial*)*num;
    ptr->materials = (MonteRayMaterial**) malloc( allocSize );
    if(ptr->materials == 0) abort ();

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
    CUDA_CHECK_RETURN( cudaMalloc(&pCopy->materialID, allocSize ));

    // materials
    allocSize = sizeof(MonteRayMaterial*)*num;
    CUDA_CHECK_RETURN( cudaMalloc(&pCopy->materials, allocSize ));
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
    cudaFree( ptr->materialID );
    cudaFree( ptr->materials );
#endif
}

MonteRayMaterialListHost::MonteRayMaterialListHost(unsigned num, unsigned maxNumIsotopes, unsigned nBins) {
    pMatList = new MonteRayMaterialList;
    ctor( pMatList, num);
    temp = NULL;
    ptr_device = NULL;
    cudaCopyMade = false;

    pHash = new HashLookupHost( maxNumIsotopes, nBins);

#ifdef __CUDACC__
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
        cudaFree( ptr_device );
    }
    free( material_device_ptr_list );
#endif
}

void MonteRayMaterialListHost::copyToGPU(void) {
#ifdef __CUDACC__
    pHash->copyToGPU();
    cudaCopyMade = true;
    temp = new MonteRayMaterialList;
    copy(temp, pMatList);

    unsigned num = pMatList->numMaterials;

    temp->numMaterials = pMatList->numMaterials;

    // allocate target struct
    CUDA_CHECK_RETURN( cudaMalloc(&ptr_device, sizeof( MonteRayMaterialList ) ));

    // allocate target dynamic memory
    cudaCtor( temp, pMatList);

    unsigned allocSize = sizeof(unsigned)*num;
    CUDA_CHECK_RETURN( cudaMemcpy(temp->materialID, pMatList->materialID, allocSize, cudaMemcpyHostToDevice));

    allocSize = sizeof(MonteRayCrossSection*)*num;
    CUDA_CHECK_RETURN( cudaMemcpy(temp->materials, material_device_ptr_list, allocSize, cudaMemcpyHostToDevice));

    // copy data
    CUDA_CHECK_RETURN( cudaMemcpy(ptr_device, temp, sizeof( MonteRayMaterialList ), cudaMemcpyHostToDevice));
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
    //	printf("Debug: MonteRayMaterials::getTotalXS(const MonteRayMaterialList* ptr, unsigned i, const HashLookup* pHash, unsigned HashBin, gpuFloatType_t E, gpuFloatType_t density)\n");
    return getTotalXS( getMaterial(ptr,i), pHash, HashBin, E, density );
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

CUDA_CALLABLE_KERNEL void kernelGetTotalXS(struct MonteRayMaterialList* pMatList, unsigned matIndex, const HashLookup* pHash, unsigned HashBin, gpuFloatType_t E, gpuFloatType_t density, gpuFloatType_t* results){
    results[0] = getTotalXS(pMatList, matIndex, pHash, HashBin, E, density);
    return;
}

gpuFloatType_t MonteRayMaterialListHost::launchGetTotalXS(unsigned i, gpuFloatType_t E, gpuFloatType_t density) const {

    typedef gpuFloatType_t type_t;
    type_t result[1];
    unsigned HashBin = getHashBin( pHash->getPtr(), E);

#ifdef __CUDACC__
    type_t* result_device;

    CUDA_CHECK_RETURN( cudaMalloc( &result_device, sizeof( type_t) * 1 ));

    cudaEvent_t sync;
    cudaEventCreate(&sync);
    kernelGetTotalXS<<<1,1>>>(ptr_device, i, pHash->getPtrDevice(), HashBin, E, density, result_device);
    gpuErrchk( cudaPeekAtLastError() );
    cudaEventRecord(sync, 0);
    cudaEventSynchronize(sync);

    CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(type_t)*1, cudaMemcpyDeviceToHost));

    cudaFree( result_device );

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


void MonteRayMaterialListHost::write(std::ostream& outfile) const {
    unsigned realNumMaterials = 0;
    for( unsigned i=0; i<getNumberMaterials(); ++i ){
        if( getPtr()->materials[i] != 0 ) {
            ++realNumMaterials;
        }
    }
    binaryIO::write(outfile, realNumMaterials );

    for( unsigned i=0; i<getNumberMaterials(); ++i ){
        if( getPtr()->materials[i] != 0 ) {
            binaryIO::write(outfile, getMaterialID(i) );
        }
    }

    for( unsigned i=0; i<getNumberMaterials(); ++i ){
        MonteRayMaterialHost mat(1);
        if( getPtr()->materials[i] != 0 ) {
            mat.load( getPtr()->materials[i] );
            mat.write( outfile );
        }
    }
}

void MonteRayMaterialListHost::read(std::istream& infile) {
    unsigned num;
    binaryIO::read(infile, num);
    dtor( pMatList );
    ctor( pMatList, num );

    for( unsigned i=0; i<num; ++i ){
        unsigned id;
        binaryIO::read(infile, id );
        pMatList->materialID[i] = id;
    }
    for( unsigned i=0; i<num; ++i ){
        MonteRayMaterialHost* mat = new MonteRayMaterialHost(1); // TODO: need shared ptr here
        mat->read(infile);
        add( i, *mat, pMatList->materialID[i] );
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
