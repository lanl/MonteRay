#include "MonteRay_MaterialProperties.hh"

#include <iostream>
#include <fstream>
#include <ostream>

#ifndef __CUDACC__
#include <cstring>
#endif

#include "GPUErrorCheck.hh"
#include "MonteRay_binaryIO.hh"
#include "MonteRayMemory.hh"
#include "MonteRayParallelAssistant.hh"

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

#ifdef __CUDACC__
    unsigned long long allocSize = sizeof(offset_t)*(numCells+1);
    ptr->offset = (offset_t*) MONTERAYDEVICEALLOC( allocSize, std::string("MonteRay_MaterialProperties_Data::offset") );

    allocSize = sizeof(MatID_t)*numComponents;
    ptr->ID = (MatID_t*) MONTERAYDEVICEALLOC( allocSize, std::string("MonteRay_MaterialProperties_Data::ID") );

    allocSize = sizeof(Density_t)*numComponents;
    ptr->density = (Density_t*) MONTERAYDEVICEALLOC( allocSize, std::string("MonteRay_MaterialProperties_Data::density") );
#endif
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

void cudaDtor(MonteRay_MaterialProperties_Data* ptr) {
#ifdef __CUDACC__
    MonteRayDeviceFree( ptr->offset );
    MonteRayDeviceFree( ptr->ID );
    MonteRayDeviceFree( ptr->density );
#endif
}


void MonteRay_MaterialProperties::cudaDtor(void) {
#ifdef __CUDACC__
    if( cudaCopyMade ) {
        MonteRayDeviceFree( ptrData_device );
        MonteRay::cudaDtor( tempData );
        delete tempData;
    }
#endif
}

void MonteRay_MaterialProperties::copyToGPU(void) {
#ifdef __CUDACC__
    if( ! MonteRay::isWorkGroupMaster() ) return;
    cudaDtor();

    cudaCopyMade = true;

    tempData = new MonteRay_MaterialProperties_Data;
    // allocate target dynamic memory
    MonteRay::cudaCtor( tempData, size(), numMatSpecs() );

    // allocate target struct
    ptrData_device = (MonteRay_MaterialProperties_Data*) MONTERAYDEVICEALLOC( sizeof( MonteRay_MaterialProperties_Data), std::string("MonteRay_MaterialProperties::ptrData_device") );

    // copy allocated data arrays
    unsigned long long allocSize = sizeof(offset_t)*(tempData->numCells+1);
    CUDA_CHECK_RETURN( cudaMemcpy(tempData->offset, getOffsetData(), allocSize, cudaMemcpyHostToDevice));

    allocSize = sizeof(MatID_t)*tempData->numMaterialComponents;
    CUDA_CHECK_RETURN( cudaMemcpy(tempData->ID, getMaterialIDData(), allocSize, cudaMemcpyHostToDevice));

    allocSize = sizeof(Density_t)*tempData->numMaterialComponents;
    CUDA_CHECK_RETURN( cudaMemcpy(tempData->density, getMaterialDensityData(), allocSize, cudaMemcpyHostToDevice));

    // copy struct
    CUDA_CHECK_RETURN( cudaMemcpy(ptrData_device, tempData, sizeof( MonteRay_MaterialProperties_Data ), cudaMemcpyHostToDevice));
#else

    delete ptrData;

    ptrData = new MonteRay_MaterialProperties_Data;

    // allocate target dynamic memory
    MonteRay::ctor( ptrData, size(), numMatSpecs() );

    unsigned long long allocSize = sizeof(offset_t)*(ptrData->numCells+1);
    memcpy( ptrData->offset,  getOffsetData(), allocSize);

    allocSize = sizeof(MatID_t)*ptrData->numMaterialComponents;
    memcpy( ptrData->ID,      getMaterialIDData(), allocSize);

    allocSize = sizeof(Density_t)*ptrData->numMaterialComponents;
    memcpy( ptrData->density, getMaterialDensityData(), allocSize);
#endif
}

CUDA_CALLABLE_MEMBER
size_t getNumCells(const struct MonteRay_MaterialProperties_Data* ptr ) {
    return ptr->numCells;
}

CUDA_CALLABLE_MEMBER
offset_t getNumMats(const struct MonteRay_MaterialProperties_Data* ptr, unsigned i ){
    MONTERAY_ASSERT_MSG( i < ptr->numCells, "MonteRay::getNumMats(MonteRay_MaterialProperties_Data*,i)  -- requested cell number exceeds number of allocated cells!" );
    return ptr->offset[i+1] - ptr->offset[i];
}

CUDA_CALLABLE_MEMBER
Density_t getDensity(const struct MonteRay_MaterialProperties_Data* ptr, unsigned cellNum, unsigned matNum ){

    MONTERAY_ASSERT_MSG( cellNum < ptr->numCells, "MonteRay::getDensity(MonteRay_MaterialProperties_Data*,cellNum,matNum)  -- requested cell number exceeds number of allocated cells!" );
    MONTERAY_ASSERT_MSG( matNum < getNumMats(ptr,cellNum), "MonteRay::getDensity(MonteRay_MaterialProperties_Data*,cellNum,matNum)  -- requested material exceeds number of materials in the cell!" );

    const size_t index = ptr->offset[cellNum] + matNum;
    MONTERAY_ASSERT_MSG( index < ptr->numMaterialComponents, "MonteRay::getDensity(MonteRay_MaterialProperties_Data*,cellNum,matNum)  -- requested material index exceeds number of total number of materials.!" );

    return ptr->density[ index ];
}

CUDA_CALLABLE_MEMBER
MatID_t getMatID(const struct MonteRay_MaterialProperties_Data* ptr, unsigned cellNum, unsigned matNum ){
    MONTERAY_ASSERT_MSG( cellNum < ptr->numCells, "MonteRay::getMatID(MonteRay_MaterialProperties_Data*,cellNum,matNum)  -- requested cell number exceeds number of allocated cells!" );
    MONTERAY_ASSERT_MSG( matNum < getNumMats(ptr,cellNum), "MonteRay::getMatID(MonteRay_MaterialProperties_Data*,cellNum,matNum)  -- requested material exceeds number of materials in the cell!" );

    const size_t index = ptr->offset[cellNum] + matNum;
    MONTERAY_ASSERT_MSG( index < ptr->numMaterialComponents, "MonteRay::getDensity(MonteRay_MaterialProperties_Data*,cellNum,matNum)  -- requested material index exceeds number of total number of materials.!" );

    return ptr->ID[ index ];
}

CUDA_CALLABLE_KERNEL  kernelGetNumCells(MonteRay_MaterialProperties_Data* mp, unsigned* results ) {
    results[0] = getNumCells(mp);
}

CUDA_CALLABLE_KERNEL  kernelGetNumMaterials(MonteRay_MaterialProperties_Data* mp, unsigned cellNum, MonteRay_MaterialProperties_Data::Material_Index_t* results ) {
    results[0] = getNumMats(mp, cellNum);
}

CUDA_CALLABLE_KERNEL  kernelGetMaterialID(MonteRay_MaterialProperties_Data* mp, unsigned cellNum, unsigned i, MonteRay_MaterialProperties_Data::MatID_t* results ) {
    results[0] = getMatID(mp, cellNum, i);
}

CUDA_CALLABLE_KERNEL  kernelGetMaterialDensity(MonteRay_MaterialProperties_Data* mp, unsigned cellNum, unsigned i, MonteRay_MaterialProperties_Data::Density_t* results ) {
    results[0] = getDensity(mp, cellNum, i);
}

CUDA_CALLABLE_KERNEL  kernelSumMatDensity(MonteRay_MaterialProperties_Data* mp, MonteRay_MaterialProperties_Data::MatID_t matIndex, MonteRay_MaterialProperties_Data::Density_t* results ) {
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

size_t MonteRay_MaterialProperties::launchGetNumCells(void) const{
    typedef unsigned type_t;
    type_t result[1];

#ifdef __CUDACC__
    type_t* result_device;
    result_device = (type_t*) MONTERAYDEVICEALLOC( sizeof( type_t) * 1, std::string("MonteRay_MaterialProperties::launchGetNumCells::result_device") );

    cudaEvent_t sync;
    cudaEventCreate(&sync);
    kernelGetNumCells<<<1,1>>>(ptrData_device, result_device);
    gpuErrchk( cudaPeekAtLastError() );
    cudaEventRecord(sync, 0);
    cudaEventSynchronize(sync);

    CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(type_t)*1, cudaMemcpyDeviceToHost));

    MonteRayDeviceFree( result_device );
#else
    kernelGetNumCells(ptrData, result);
#endif

    return result[0];
}


MonteRay_MaterialProperties::Material_Index_t MonteRay_MaterialProperties::launchGetNumMaterials( Cell_Index_t cellID ) const {
    typedef Material_Index_t type_t;
    type_t result[1];

#ifdef __CUDACC__
    type_t* result_device;
    result_device = (type_t*) MONTERAYDEVICEALLOC( sizeof( type_t) * 1, std::string("MonteRay_MaterialProperties::launchGetNumMaterials::result_device") );

    cudaEvent_t sync;
    cudaEventCreate(&sync);
    kernelGetNumMaterials<<<1,1>>>(ptrData_device, cellID, result_device);
    gpuErrchk( cudaPeekAtLastError() );
    cudaEventRecord(sync, 0);
    cudaEventSynchronize(sync);

    CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(type_t)*1, cudaMemcpyDeviceToHost));

    MonteRayDeviceFree( result_device );
#else
    kernelGetNumMaterials(ptrData, cellID, result);
#endif
    return result[0];
}


MonteRay_MaterialProperties::MatID_t MonteRay_MaterialProperties::launchGetMaterialID( Cell_Index_t cellID, Material_Index_t i ) const {
    typedef MatID_t type_t;
    type_t result[1];

#ifdef __CUDACC__
    type_t* result_device;
    result_device = (type_t*) MONTERAYDEVICEALLOC( sizeof( type_t) * 1, std::string("MonteRay_MaterialProperties::launchGetMaterialID::result_device") );

    cudaEvent_t sync;
    cudaEventCreate(&sync);
    kernelGetMaterialID<<<1,1>>>(ptrData_device, cellID, i, result_device);
    gpuErrchk( cudaPeekAtLastError() );
    cudaEventRecord(sync, 0);
    cudaEventSynchronize(sync);

    CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(type_t)*1, cudaMemcpyDeviceToHost));

    MonteRayDeviceFree( result_device );
#else
    kernelGetMaterialID(ptrData, cellID, i, result);
#endif

    return result[0];
}

MonteRay_MaterialProperties::Density_t MonteRay_MaterialProperties::launchGetMaterialDensity( Cell_Index_t cellID, Material_Index_t i ) const {
    typedef Density_t type_t;
    type_t result[1];

#ifdef __CUDACC__
    type_t* result_device;
    result_device = (type_t*) MONTERAYDEVICEALLOC( sizeof( type_t) * 1, std::string("MonteRay_MaterialProperties::launchGetMaterialDensity::result_device") );

    cudaEvent_t sync;
    cudaEventCreate(&sync);
    kernelGetMaterialDensity<<<1,1>>>(ptrData_device, cellID, i, result_device);
    gpuErrchk( cudaPeekAtLastError() );
    cudaEventRecord(sync, 0);
    cudaEventSynchronize(sync);

    CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(type_t)*1, cudaMemcpyDeviceToHost));

    MonteRayDeviceFree( result_device );
#else
    kernelGetMaterialDensity(ptrData, cellID, i, result);
#endif

    return result[0];
}

Density_t MonteRay_MaterialProperties::launchSumMatDensity(MatID_t matID) const{
    typedef Density_t type_t;

    type_t* result_device;
    type_t result[1];

#ifdef __CUDACC__
    result_device = (type_t*) MONTERAYDEVICEALLOC( sizeof( type_t) * 1, std::string("MonteRay_MaterialProperties::launchSumMatDensity::result_device") );

    cudaEvent_t sync;
    cudaEventCreate(&sync);
    kernelSumMatDensity<<<1,1>>>(ptrData_device, matID, result_device);
    gpuErrchk( cudaPeekAtLastError() );
    cudaEventRecord(sync, 0);
    cudaEventSynchronize(sync);

    CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(type_t)*1, cudaMemcpyDeviceToHost));

    MonteRayDeviceFree( result_device );
#else
    kernelSumMatDensity(ptrData, matID, result);
#endif
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



void 
MonteRay_MaterialProperties::setCellTemperatureCelsius( const Cell_Index_t cell, const Temperature_t temperatureCelsius){
    std::stringstream msg;
    msg << "Disabled in MonteRay!\n";
    msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_MaterialProperties::setCellTemperatureCelsius" << "\n\n";
    throw std::runtime_error( msg.str() );
//    checkCellIndex( cell, __FILE__, __LINE__);
//    double temp = (temperatureCelsius + 273.15)  / mcatk::Constants::MeVtoKelvin;
//    setCellTemperature(cell, temp);
}

MonteRay_MaterialProperties::Temperature_t
MonteRay_MaterialProperties::getTemperatureCelsius( Cell_Index_t cellID) const {
    std::stringstream msg;
    msg << "Disabled in MonteRay!\n";
    msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_MaterialProperties::getTemperatureCelsius" << "\n\n";
    throw std::runtime_error( msg.str() );
//    checkCellIndex( cellID, __FILE__, __LINE__);
//    double temp = getTemperature(cellID);
//    return temp * mcatk::Constants::MeVtoKelvin - 273.15;
}

void
MonteRay_MaterialProperties::add( MonteRay::MonteRay_CellProperties cell ) {
    pMemoryLayout->add( cell );
}

void
MonteRay_MaterialProperties::addCellMaterial( Cell_Index_t cellID, MatID_t id, Density_t den ) {
    forceCheckCellIndex( cellID );
    if( containsMaterial(cellID, id) ) {
        std::stringstream msg;
        msg << "Unable to add material to cell. Material already exists!\n";
        msg << "cell index = " << cellID << ", material ID = " << id << "\n";
        msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "'MonteRay_MaterialProperties::addCellMaterial" << "\n\n";
        throw std::runtime_error( msg.str() );
    }
    pMemoryLayout->addCellMaterial( cellID, id, den );
}

void
MonteRay_MaterialProperties::removeMaterial( Cell_Index_t cellID, MatID_t id ) {
    checkCellIndex( cellID, __FILE__, __LINE__);
    pMemoryLayout->removeMaterial( cellID, id );
}

/// Provides a safe external method for returning the material ID by cell, checks for null material
MonteRay_MaterialProperties::MatID_t
MonteRay_MaterialProperties::getMaterialID( Cell_Index_t cellID, Material_Index_t i ) const {
    MatID_t ID = getMaterialIDNotSafe(cellID, i);

    if( ID == MonteRay_MaterialSpec::NULL_MATERIAL ) {
        std::stringstream msg;
        msg << "Returning NULL_MATERIAL material ID, avoid external call to getMaterialID, call getFuncSumByCell instead!\n";
        msg << "cell index = " << cellID << ", material index = " << i << "\n";
        msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_MaterialProperties::getMaterialID" << "\n\n";
        throw std::runtime_error( msg.str() );
    }
    return ID;
}

void
MonteRay_MaterialProperties::writeToFile( const std::string& filename) const {
    std::ofstream out;
    out.open( filename.c_str(), std::ios::binary | std::ios::out);
    write( out );
    out.close();
}

void
MonteRay_MaterialProperties::readFromFile( const std::string& filename) {
    std::ifstream in;
    in.open( filename.c_str(), std::ios::binary | std::ios::in);
    if( ! in.good() ) {
        throw std::runtime_error( "MonteRay_MaterialProperties::readFromFile -- can't open file for reading" );
    }
    read( in );
    in.close();
}

void
MonteRay_MaterialProperties::write(std::ostream& outfile) const{
    pMemoryLayout->write(outfile);
}

void
MonteRay_MaterialProperties::read(std::istream& infile){
    pMemoryLayout->read(infile);
    setupPtrData();
}

} /* End namespace MonteRay */
