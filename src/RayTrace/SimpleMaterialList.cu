#include "SimpleMaterialList.h"

#include "binaryIO.h"

namespace MonteRay{

void ctor(SimpleMaterialList* ptr, unsigned num ) {
    if( num <=0 ) { num = 1; }
    ptr->numMaterials = num;

    unsigned allocSize = sizeof(unsigned)*num;

    ptr->materialID  = (unsigned*) malloc( allocSize);
    if(ptr->materialID == 0) abort ();

    allocSize = sizeof(SimpleMaterial*)*num;
    ptr->materials = (SimpleMaterial**) malloc( allocSize );
    if(ptr->materials == 0) abort ();

    for( unsigned i=0; i<num; ++i ){
        ptr->materialID[i] = 0;

        ptr->materials[i] = 0; // set to null ptr
    }
}

#ifdef CUDA
void cudaCtor(SimpleMaterialList* pCopy, unsigned num) {

	pCopy->numMaterials = num;

	// materialID
	unsigned allocSize = sizeof(unsigned)*num;
	CUDA_CHECK_RETURN( cudaMalloc(&pCopy->materialID, allocSize ));
	gpuErrchk( cudaPeekAtLastError() );

    // materials
	allocSize = sizeof(SimpleMaterial*)*num;
	CUDA_CHECK_RETURN( cudaMalloc(&pCopy->materials, allocSize ));
	gpuErrchk( cudaPeekAtLastError() );
}

void cudaCtor(struct SimpleMaterialList* pCopy, struct SimpleMaterialList* pOrig){
	unsigned num = pOrig->numMaterials;
	cudaCtor( pCopy, num);
}

#endif

void dtor(SimpleMaterialList* ptr) {
    if( ptr->materialID != 0 ) {
        free( ptr->materialID );
        ptr->materialID = 0;
    }

    if( ptr->materials != 0 ) {
        free( ptr->materials );
        ptr->materials = 0;
    }
}

#ifdef CUDA
void cudaDtor(SimpleMaterialList* ptr) {
	cudaFree( ptr->materialID );
	cudaFree( ptr->materials );
}
#endif

SimpleMaterialListHost::SimpleMaterialListHost(unsigned num, unsigned maxNumIsotopes, unsigned nBins) {
     pMatList = new SimpleMaterialList;
     ctor( pMatList, num);
     temp = NULL;
     ptr_device = NULL;
     cudaCopyMade = false;

     pHash = new HashLookupHost( maxNumIsotopes, nBins);

#ifdef CUDA
     material_device_ptr_list = (SimpleMaterial**) malloc( sizeof(SimpleMaterial* )*num );
     for( unsigned i=0; i< num; ++i ){
     	material_device_ptr_list[i] = 0;
     }
#endif
}

SimpleMaterialListHost::~SimpleMaterialListHost() {
     dtor( pMatList );
     delete pMatList;
     delete pHash;
#ifdef CUDA
     if( cudaCopyMade ) {
       	cudaDtor( temp );
       	delete temp;
     	cudaFree( ptr_device );
     }
     free( material_device_ptr_list );
#endif
 }

void SimpleMaterialListHost::copyToGPU(void) {
#ifdef CUDA
	pHash->copyToGPU();
	cudaCopyMade = true;
    temp = new SimpleMaterialList;
    copy(temp, pMatList);

	unsigned num = pMatList->numMaterials;

	temp->numMaterials = pMatList->numMaterials;

	// allocate target struct
	CUDA_CHECK_RETURN( cudaMalloc(&ptr_device, sizeof( SimpleMaterialList ) ));
	gpuErrchk( cudaPeekAtLastError() );

	// allocate target dynamic memory
	cudaCtor( temp, pMatList);

	unsigned allocSize = sizeof(unsigned)*num;
	CUDA_CHECK_RETURN( cudaMemcpy(temp->materialID, pMatList->materialID, allocSize, cudaMemcpyHostToDevice));
	gpuErrchk( cudaPeekAtLastError() );

	allocSize = sizeof(SimpleCrossSection*)*num;
	CUDA_CHECK_RETURN( cudaMemcpy(temp->materials, material_device_ptr_list, allocSize, cudaMemcpyHostToDevice));
	gpuErrchk( cudaPeekAtLastError() );

	// copy data
	CUDA_CHECK_RETURN( cudaMemcpy(ptr_device, temp, sizeof( SimpleMaterialList ), cudaMemcpyHostToDevice));
	gpuErrchk( cudaPeekAtLastError() );
#endif
}

void copy(SimpleMaterialList* pCopy, const SimpleMaterialList* const pOrig ) {
    unsigned num = pOrig->numMaterials;
    if( num <=0 ) { num = 1; }

    ctor( pCopy, num);

    for( unsigned i=0; i<num; ++i ){
        pCopy->materialID[i] = pOrig->materialID[i];
        if( pOrig->materials[i] != 0 ) {
            if( pCopy->materials[i] == 0 ) {
                pCopy->materials[i] = new SimpleMaterial; // TODO: memory leak -- need shared ptr
            }
            copy( pCopy->materials[i], pOrig->materials[i]);
        }
    }
}

#ifdef CUDA
__device__ __host__
#endif
unsigned getNumberMaterials(SimpleMaterialList* ptr) {
    return ptr->numMaterials;
}

#ifdef CUDA
__device__ __host__
#endif
unsigned getMaterialID(SimpleMaterialList* ptr, unsigned i ) {
    return ptr->materialID[i];
}

#ifdef CUDA
__device__ __host__
#endif
SimpleMaterial* getMaterial(SimpleMaterialList* ptr, unsigned i ){
    return ptr->materials[i];
}

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getTotalXS(SimpleMaterialList* ptr, unsigned i, HashLookup* pHash, unsigned HashBin, gpuFloatType_t E, gpuFloatType_t density) {
    return getTotalXS( ptr->materials[i], pHash, HashBin, E, density );
}

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getTotalXS(SimpleMaterialList* ptr, unsigned i, gpuFloatType_t E, gpuFloatType_t density) {
    return getTotalXS( ptr->materials[i], E, density );
}

#ifdef CUDA
__device__ __host__
#endif
unsigned materialIDtoIndex(SimpleMaterialList* ptr, unsigned id ) {
    for( unsigned i=0; i < ptr->numMaterials; ++i ){
        if( id == ptr->materialID[i] ) {
            return i;
        }
    }

    printf("Error: materialIDtoIndex -- id=%d not found.  %s %d\n", id, __FILE__, __LINE__);
    abort;
    return 0;
}

#ifdef CUDA
__global__ void kernelGetTotalXS(struct SimpleMaterialList* pMatList, unsigned matIndex, HashLookup* pHash, unsigned HashBin, gpuFloatType_t E, gpuFloatType_t density, gpuFloatType_t* results){
    results[0] = getTotalXS(pMatList, matIndex, pHash, HashBin, E, density);
    return;
}
#endif

gpuFloatType_t SimpleMaterialListHost::launchGetTotalXS(unsigned i, gpuFloatType_t E, gpuFloatType_t density) const {
#ifdef CUDA
	typedef gpuFloatType_t type_t;

	type_t* result_device;
	type_t result[1];
	CUDA_CHECK_RETURN( cudaMalloc( &result_device, sizeof( type_t) * 1 ));
	gpuErrchk( cudaPeekAtLastError() );

	unsigned HashBin = getHashBin( pHash->getPtr(), E);

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	kernelGetTotalXS<<<1,1>>>(ptr_device, i, pHash->getPtrDevice(), HashBin, E, density, result_device);
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);

    gpuErrchk( cudaPeekAtLastError() );

	CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(type_t)*1, cudaMemcpyDeviceToHost));
	gpuErrchk( cudaPeekAtLastError() );

	cudaFree( result_device );
	return result[0];
#endif
}


void SimpleMaterialListHost::add( unsigned index, SimpleMaterialHost& mat, unsigned id) {
    if( index >= getNumberMaterials() ) {
        fprintf(stderr, "SimpleMaterialListHost::add -- index > number of allocated materials.  %s %d\n", __FILE__, __LINE__);
        exit(1);
    }

    pMatList->materialID[index] = id;
    pMatList->materials[index] = mat.getPtr();

#ifdef CUDA
    material_device_ptr_list[index] = mat.ptr_device;
#endif
    for( unsigned i = 0; i < mat.getNumIsotopes(); ++i ){
//    	std::cout << "Debug: SimpleMaterialListHost::add -- i=" << i << "\n";
    	int currentID = mat.getID(i);
//    	std::cout << "Debug: SimpleMaterialListHost::add -- currentID=" << currentID << "\n";
    	if( currentID < 0 ) {
    		pHash->addIsotope( mat.getPtr()->xs[i] );
    	}
    }
}

#ifndef CUDA
void SimpleMaterialListHost::add( unsigned index, SimpleMaterial* mat, unsigned id) {
    if( index >= getNumberMaterials() ) {
        fprintf(stderr, "SimpleMaterialListHost::add -- index > number of allocated materials.  %s %d\n", __FILE__, __LINE__);
        exit(1);
    }

    pMatList->materialID[index] = id;
    pMatList->materials[index] = mat;

    for( i = 0; i < getNumIsotopes(mat); ++i ){
    	int currentID = getID(mat, i);
    	if( currentID < 0 ) {
    		pHash->addIsotope( mat->xs[i] );
    	}
    }
}
#endif

void SimpleMaterialListHost::write(std::ostream& outfile) const {


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
        SimpleMaterialHost mat(1);
        if( getPtr()->materials[i] != 0 ) {
            mat.load( getPtr()->materials[i] );
            mat.write( outfile );
        }
    }
}

void SimpleMaterialListHost::read(std::istream& infile) {
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
        SimpleMaterialHost* mat = new SimpleMaterialHost(1); // TODO: need shared ptr here
        mat->read(infile);
        add( i, *mat, pMatList->materialID[i] );
    }
}

}
