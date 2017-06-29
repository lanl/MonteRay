#include "MonteRay_MaterialProperties.hh"

#include <iostream>
#include <fstream>
#include <ostream>

#include "GPUErrorCheck.hh"
#include "MonteRay_binaryIO.hh"

namespace MonteRay{
typedef MonteRay_MaterialProperties_Data::offset_t offset_t;
typedef MonteRay_MaterialProperties_Data::MatID_t MatID_t;
typedef MonteRay_MaterialProperties_Data::Density_t Density_t;


void copy(MonteRay_MaterialProperties_Data& theCopy, const MonteRay_MaterialProperties_Data& theOrig) {
	theCopy = theOrig;
}

void copy(MonteRay_MaterialProperties_Data* pCopy, const MonteRay_MaterialProperties_Data* pOrig) {
	copy( *pCopy, *pOrig);
}

void ctor(MonteRay_MaterialProperties_Data* ptr, unsigned numCells, unsigned numComponents ) {
    if( numCells <=0 ) { numCells = 1; }
    if( numComponents <=0 ) { numComponents = 1; }

    ptr->numCells = numCells;
    ptr->numMaterialComponents = numComponents;

    unsigned long long allocSize = sizeof(offset_t)*(numCells+1);
    ptr->offset = (offset_t*) malloc( allocSize);
    if(ptr->offset == 0) abort ();

    allocSize = sizeof(MatID_t)*numComponents;
    ptr->ID = (MatID_t*) malloc( allocSize);
    if(ptr->ID == 0) abort ();

    allocSize = sizeof(Density_t)*numComponents;
    ptr->density = (Density_t*) malloc( allocSize);
    if(ptr->density == 0) abort ();

    for( unsigned i=0; i<numCells; ++i ){
    	ptr->offset[i] = 0;
    }

    for( unsigned i=0; i<numComponents; ++i ){
    	ptr->ID[i] = 0;
    	ptr->density[i] = 0.0;
    }
}

void cudaCtor(MonteRay_MaterialProperties_Data* ptr, unsigned numCells, unsigned numComponents ) {
    if( numCells <=0 ) { numCells = 1; }
    if( numComponents <=0 ) { numComponents = 1; }

    ptr->numCells = numCells;
    ptr->numMaterialComponents = numComponents;

    unsigned long long allocSize = sizeof(offset_t)*(numCells+1);
    CUDA_CHECK_RETURN( cudaMalloc(&ptr->offset, allocSize ));

    allocSize = sizeof(MatID_t)*numComponents;
    CUDA_CHECK_RETURN( cudaMalloc(&ptr->ID, allocSize ));

    allocSize = sizeof(Density_t)*numComponents;
    CUDA_CHECK_RETURN( cudaMalloc(&ptr->density, allocSize ));
}

void cudaCtor(struct MonteRay_MaterialProperties_Data* pCopy, struct MonteRay_MaterialProperties_Data* pOrig){
	unsigned numCells = pOrig->numCells;
	unsigned numComponents = pOrig->numMaterialComponents;
	cudaCtor( pCopy, numCells, numComponents);
}


void dtor(struct MonteRay_MaterialProperties_Data* ptr){
    if( ptr->offset != 0 ) {
        free(ptr->offset);
        ptr->offset = 0;
    }
    if( ptr->ID != 0 ) {
    	free(ptr->ID);
    	ptr->ID = 0;
    }
    if( ptr->density != 0 ) {
    	free(ptr->density);
    	ptr->density = 0;
    }
}

#ifdef CUDA
void cudaDtor(MonteRay_MaterialProperties_Data* ptr) {
	cudaFree( ptr->offset );
	cudaFree( ptr->ID );
	cudaFree( ptr->density );
}
#endif

void MonteRay_MaterialProperties::cudaDtor(void) {
#ifdef CUDA
    if( cudaCopyMade ) {
        cudaFree( ptrData_device );
        MonteRay::cudaDtor( tempData );
        delete tempData;
    }
#endif
}

void MonteRay_MaterialProperties::copyToGPU(void) {
	cudaCopyMade = true;
	tempData = new MonteRay_MaterialProperties_Data;

	// allocate target dynamic memory
	MonteRay::cudaCtor( tempData, size(), numMatSpecs() );

	// allocate target struct
	CUDA_CHECK_RETURN( cudaMalloc(&ptrData_device, sizeof( MonteRay_MaterialProperties_Data) ));

	// copy allocated data arrays
	unsigned long long allocSize = sizeof(offset_t)*(tempData->numCells+1);
	CUDA_CHECK_RETURN( cudaMemcpy(tempData->offset, getOffsetData(), allocSize, cudaMemcpyHostToDevice));

	allocSize = sizeof(MatID_t)*tempData->numMaterialComponents;
	CUDA_CHECK_RETURN( cudaMemcpy(tempData->ID, getMaterialIDData(), allocSize, cudaMemcpyHostToDevice));

	allocSize = sizeof(Density_t)*tempData->numMaterialComponents;
	CUDA_CHECK_RETURN( cudaMemcpy(tempData->density, getMaterialDensityData(), allocSize, cudaMemcpyHostToDevice));

	// copy struct
	CUDA_CHECK_RETURN( cudaMemcpy(ptrData_device, tempData, sizeof( MonteRay_MaterialProperties_Data ), cudaMemcpyHostToDevice));
}

#ifdef CUDA
__device__ __host__
#endif
size_t getNumCells(struct MonteRay_MaterialProperties_Data* ptr ) {
    return ptr->numCells;
}

#ifdef CUDA
__device__ __host__
#endif
offset_t getNumMats(struct MonteRay_MaterialProperties_Data* ptr, unsigned i ){
    return ptr->offset[i+1] - ptr->offset[i];
}

#ifdef CUDA
__device__ __host__
#endif
Density_t getDensity(struct MonteRay_MaterialProperties_Data* ptr, unsigned cellNum, unsigned matNum ){
    return ptr->density[ ptr->offset[cellNum] + matNum];
}

#ifdef CUDA
__device__ __host__
#endif
MatID_t getMatID(struct MonteRay_MaterialProperties_Data* ptr, unsigned cellNum, unsigned matNum ){
	return ptr->ID[ ptr->offset[cellNum] + matNum];
}

#ifdef CUDA
__global__ void kernelGetNumCells(MonteRay_MaterialProperties_Data* mp, unsigned* results ) {
     results[0] = getNumCells(mp);
}
#endif


#ifdef CUDA
__global__ void kernelGetNumMaterials(MonteRay_MaterialProperties_Data* mp, unsigned cellNum, MonteRay_MaterialProperties_Data::Material_Index_t* results ) {
	results[0] = getNumMats(mp, cellNum);
}
#endif

#ifdef CUDA
__global__ void kernelGetMaterialID(MonteRay_MaterialProperties_Data* mp, unsigned cellNum, unsigned i, MonteRay_MaterialProperties_Data::MatID_t* results ) {
	results[0] = getMatID(mp, cellNum, i);
}
#endif

#ifdef CUDA
__global__ void kernelGetMaterialDensity(MonteRay_MaterialProperties_Data* mp, unsigned cellNum, unsigned i, MonteRay_MaterialProperties_Data::Density_t* results ) {
	results[0] = getDensity(mp, cellNum, i);
}
#endif


#ifdef CUDA
__global__ void kernelSumMatDensity(MonteRay_MaterialProperties_Data* mp, MonteRay_MaterialProperties_Data::MatID_t matIndex, MonteRay_MaterialProperties_Data::Density_t* results ) {
    gpuFloatType_t sum = 0.0f;
    for( unsigned cell=0; cell < getNumCells(mp); ++cell) {
         for( unsigned matNum=0; matNum < getNumMats(mp, cell); ++matNum ) {

        	 Density_t density = getDensity(mp, cell, matNum);
        	 MatID_t matID = getMatID(mp, cell, matNum);

             if( matID == matIndex ) {
                 sum += density;
             }
         }
     }
     results[0] = sum;
}
#endif


size_t MonteRay_MaterialProperties::launchGetNumCells(void) const{
	typedef unsigned type_t;

	type_t* result_device;
	type_t result[1];
	CUDA_CHECK_RETURN( cudaMalloc( &result_device, sizeof( type_t) * 1 ));

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	kernelGetNumCells<<<1,1>>>(ptrData_device, result_device);
    gpuErrchk( cudaPeekAtLastError() );
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);

	CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(type_t)*1, cudaMemcpyDeviceToHost));

	cudaFree( result_device );
	return result[0];
}


MonteRay_MaterialProperties::Material_Index_t MonteRay_MaterialProperties::launchGetNumMaterials( Cell_Index_t cellID ) const {
	typedef Material_Index_t type_t;

	type_t* result_device;
	type_t result[1];
	CUDA_CHECK_RETURN( cudaMalloc( &result_device, sizeof( type_t) * 1 ));

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	kernelGetNumMaterials<<<1,1>>>(ptrData_device, cellID, result_device);
    gpuErrchk( cudaPeekAtLastError() );
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);

	CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(type_t)*1, cudaMemcpyDeviceToHost));

	cudaFree( result_device );
	return result[0];
}


MonteRay_MaterialProperties::MatID_t MonteRay_MaterialProperties::launchGetMaterialID( Cell_Index_t cellID, Material_Index_t i ) const {
	typedef MatID_t type_t;

	type_t* result_device;
	type_t result[1];
	CUDA_CHECK_RETURN( cudaMalloc( &result_device, sizeof( type_t) * 1 ));

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	kernelGetMaterialID<<<1,1>>>(ptrData_device, cellID, i, result_device);
    gpuErrchk( cudaPeekAtLastError() );
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);

	CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(type_t)*1, cudaMemcpyDeviceToHost));

	cudaFree( result_device );
	return result[0];
}

MonteRay_MaterialProperties::Density_t MonteRay_MaterialProperties::launchGetMaterialDensity( Cell_Index_t cellID, Material_Index_t i ) const {
	typedef Density_t type_t;

	type_t* result_device;
	type_t result[1];
	CUDA_CHECK_RETURN( cudaMalloc( &result_device, sizeof( type_t) * 1 ));

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	kernelGetMaterialDensity<<<1,1>>>(ptrData_device, cellID, i, result_device);
    gpuErrchk( cudaPeekAtLastError() );
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);

	CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(type_t)*1, cudaMemcpyDeviceToHost));

	cudaFree( result_device );
	return result[0];
}


Density_t MonteRay_MaterialProperties::launchSumMatDensity(MatID_t matID) const{
	typedef Density_t type_t;

	type_t* result_device;
	type_t result[1];
	CUDA_CHECK_RETURN( cudaMalloc( &result_device, sizeof( type_t) * 1 ));

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	kernelSumMatDensity<<<1,1>>>(ptrData_device, matID, result_device);
    gpuErrchk( cudaPeekAtLastError() );
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);

	CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(type_t)*1, cudaMemcpyDeviceToHost));

	cudaFree( result_device );
	return result[0];
}


Density_t MonteRay_MaterialProperties::sumMatDensity( MatID_t matIndex) const {
	Density_t sum = 0.0f;
    for( unsigned cell=0; cell < size(); ++cell) {
         for( unsigned matNum=0; matNum < getNumMaterials(cell); ++matNum ) {

        	 Density_t density = getMaterialDensity(cell, matNum);
             MatID_t matID = getMaterialID(cell, matNum);

             if( matID == matIndex ) {
                 sum += density;
             }
         }
     }
     return sum;
}


}
