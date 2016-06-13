#include "SimpleMaterialProperties.h"

#include <iostream>
#include <fstream>
#include <ostream>

#include "binaryIO.h"

void copy(SimpleCellProperties& theCopy, const SimpleCellProperties& theOrig) {
//	theCopy.numMats = theOrig.numMats;
//	for( unsigned i=0; i < theOrig.numMats; ++i ) {
//		theCopy.matID[i] = theOrig.matID[i];
//		theCopy.density[i] = theOrig.matID[i];
//	}
	theCopy = theOrig;
}

void copy(SimpleCellProperties* pCopy, const SimpleCellProperties* pOrig) {
	copy( *pCopy, *pOrig);
}

void ctor(SimpleMaterialProperties* ptr, unsigned num ) {
    if( num <=0 ) { num = 1; }
    ptr->numCells = num;

    unsigned long long allocSize = sizeof(struct SimpleCellProperties)*num;
    ptr->props = (SimpleCellProperties*) malloc( allocSize);
    if(ptr->props == 0) abort ();

    SimpleCellProperties defaultCellProps;
    defaultCellProps.numMats = 0;
    for( unsigned i=0; i<NMAX_MATERIALS; ++i) {
        defaultCellProps.density[i] = 0.0;
        defaultCellProps.matID[i] = 0U;
    }

    for( unsigned i=0; i<num; ++i ){
        copy( ptr->props[i], defaultCellProps);
    }
}

#ifdef CUDA
void cudaCtor(SimpleMaterialProperties* pCopy, unsigned num) {
	pCopy->numCells = num;

	// props
	unsigned long long allocSize = sizeof(SimpleCellProperties)*num;
	CUDA_CHECK_RETURN( cudaMalloc(&pCopy->props, allocSize ));
	gpuErrchk( cudaPeekAtLastError() );
}

void cudaCtor(struct SimpleMaterialProperties* pCopy, struct SimpleMaterialProperties* pOrig){
	unsigned num = pOrig->numCells;
	cudaCtor( pCopy, num);
}
#endif

void dtor(struct SimpleMaterialProperties* ptr){
    if( ptr->props != 0 ) {
        free(ptr->props);
        ptr->props = 0;
    }
}

#ifdef CUDA
void cudaDtor(SimpleMaterialProperties* ptr) {
	cudaFree( ptr->props );
}
#endif

SimpleMaterialPropertiesHost::SimpleMaterialPropertiesHost(unsigned numCells) {
     ptr = new SimpleMaterialProperties;
     ctor( ptr, numCells );
     ptr_device = NULL;
     temp = NULL;
     cudaCopyMade = false;
}

SimpleMaterialPropertiesHost::~SimpleMaterialPropertiesHost() {
    if( ptr != 0 ) {
        dtor( ptr );
        delete ptr;
        ptr = 0;
    }

#ifdef CUDA
    if( cudaCopyMade ) {
        cudaFree( ptr_device );
        cudaDtor( temp );
        delete temp;
    }
#endif
}


void SimpleMaterialPropertiesHost::copyToGPU(void) {
#ifdef CUDA
	cudaCopyMade = true;
	temp = new SimpleMaterialProperties;
    copy(temp, ptr);

	unsigned num = ptr->numCells;

	// allocate target struct
	CUDA_CHECK_RETURN( cudaMalloc(&ptr_device, sizeof( SimpleMaterialProperties) ));
	gpuErrchk( cudaPeekAtLastError() );

	// allocate target dynamic memory
	cudaCtor( temp, ptr);

	unsigned long long allocSize = sizeof(SimpleCellProperties)*num;
	CUDA_CHECK_RETURN( cudaMemcpy(temp->props, ptr->props, allocSize, cudaMemcpyHostToDevice));
	gpuErrchk( cudaPeekAtLastError() );

	// copy data
	CUDA_CHECK_RETURN( cudaMemcpy(ptr_device, temp, sizeof( SimpleMaterialProperties ), cudaMemcpyHostToDevice));
	gpuErrchk( cudaPeekAtLastError() );

#endif
}

void copy(struct SimpleMaterialProperties* pCopy, struct SimpleMaterialProperties* pOrig) {
	unsigned num = pOrig->numCells;
    if( num <=0 ) { num = 1; }

    ctor( pCopy, num);

    for( unsigned i=0; i<num; ++i ){
    	copy( pCopy->props[i], pOrig->props[i] );
    }
}

#ifdef CUDA
__device__ __host__
#endif
unsigned getNumCells(struct SimpleMaterialProperties* ptr ) {
    return ptr->numCells;
}

#ifdef CUDA
__device__ __host__
#endif
unsigned getNumMats(struct SimpleMaterialProperties* ptr, unsigned i ){
    return ptr->props[i].numMats;
}

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getDensity(struct SimpleMaterialProperties* ptr, unsigned cellNum, unsigned matNum ){
    return ptr->props[cellNum].density[matNum];
}

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getMatID(struct SimpleMaterialProperties* ptr, unsigned cellNum, unsigned matNum ){
    return ptr->props[cellNum].matID[matNum];
}

#ifdef CUDA
__global__ void kernelGetNumCells(SimpleMaterialProperties* mp, unsigned* results ) {
     results[0] = getNumCells(mp);
}
#endif

#ifdef CUDA
__global__ void kernelSumMatDensity(SimpleMaterialProperties* mp, unsigned matIndex, gpuFloatType_t* results ) {
    gpuFloatType_t sum = 0.0f;
    for( unsigned cell=0; cell < getNumCells(mp); ++cell) {
         for( unsigned matNum=0; matNum < getNumMats(mp, cell); ++matNum ) {

             gpuFloatType_t density = getDensity(mp, cell, matNum);
             unsigned matID = getMatID(mp, cell, matNum);

             if( matID == matIndex ) {
                 sum += density;
             }
         }
     }
     results[0] = sum;
}
#endif

void addDensityAndID(struct SimpleMaterialProperties* ptr, unsigned cellNum, gpuFloatType_t density, unsigned matID ) {
    if( density <= 0.0 ) return;

    unsigned matNum = ptr->props[cellNum].numMats;
    if( matNum > NMAX_MATERIALS ) {
        fprintf(stderr, "addDensityAndID( SimpleMaterialProperties* ) -- cell has exceeded cell properties size.  %s %d\n", __FILE__, __LINE__);
        exit(1);
    }
    ptr->props[cellNum].density[matNum] = density;
    ptr->props[cellNum].matID[matNum] = matID;

    ptr->props[cellNum].numMats += 1;
}

#ifndef CUDA
#include "ReadLnk3dnt.hh"
void SimpleMaterialPropertiesHost::loadFromLnk3dnt(const std::string& filename ) {
    ReadLnk3dnt file( filename);
    file.ReadMatData( );
    unsigned numCells = file.getNumTotalCells();

    dtor( ptr );
    ctor( ptr, numCells );

    unsigned numMats = file.MaxMaterialsPerCell();
    std::cout << "Debug: numMats=" << numMats << std::endl;

    for( unsigned cell=0; cell < numCells; ++cell) {
        for( unsigned matNum=0; matNum < numMats; ++matNum ) {
            gpuFloatType_t density = file.getDensity(cell, matNum);
            unsigned matID = file.getMatID(cell, matNum);

            matID -= 2;  // decrement partisn id numbers as they start with 1, and 1 is the ghost mat
            if( matID < 0 ) {
                // partisn ghost id
                continue;
            }

            addDensityAndID(cell, density, matID);
        }
    }
}
#endif

unsigned SimpleMaterialPropertiesHost::launchGetNumCells(void) const{
#ifdef CUDA
	typedef unsigned type_t;

	type_t* result_device;
	type_t result[1];
	CUDA_CHECK_RETURN( cudaMalloc( &result_device, sizeof( type_t) * 1 ));
	gpuErrchk( cudaPeekAtLastError() );

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	kernelGetNumCells<<<1,1>>>(ptr_device, result_device);
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);

    gpuErrchk( cudaPeekAtLastError() );

	CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(type_t)*1, cudaMemcpyDeviceToHost));
	gpuErrchk( cudaPeekAtLastError() );

	cudaFree( result_device );
	return result[0];
#endif
}

unsigned SimpleMaterialPropertiesHost::launchSumMatDensity(unsigned matID) const{
#ifdef CUDA
	typedef gpuFloatType_t type_t;

	type_t* result_device;
	type_t result[1];
	CUDA_CHECK_RETURN( cudaMalloc( &result_device, sizeof( type_t) * 1 ));
	gpuErrchk( cudaPeekAtLastError() );

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	kernelSumMatDensity<<<1,1>>>(ptr_device, matID, result_device);
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);

    gpuErrchk( cudaPeekAtLastError() );

	CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(type_t)*1, cudaMemcpyDeviceToHost));
	gpuErrchk( cudaPeekAtLastError() );

	cudaFree( result_device );
	return result[0];
#endif
}


gpuFloatType_t SimpleMaterialPropertiesHost::sumMatDensity( unsigned matIndex) const {
    gpuFloatType_t sum = 0.0f;
    for( unsigned cell=0; cell < getNumCells(); ++cell) {
         for( unsigned matNum=0; matNum < getNumMats(cell); ++matNum ) {

             gpuFloatType_t density = getDensity(cell, matNum);
             unsigned matID = getMatID(cell, matNum);

             if( matID == matIndex ) {
                 sum += density;
             }
         }
     }
     return sum;
}

void SimpleMaterialPropertiesHost::write(std::ostream& outf, const SimpleCellProperties& cellProp) const {
    binaryIO::write(outf, cellProp.numMats );
    for( unsigned i=0; i< cellProp.numMats; ++i ) {
        binaryIO::write(outf, cellProp.matID[i] ) ;
    }
    for( unsigned i=0; i< cellProp.numMats; ++i ) {
        binaryIO::write(outf, cellProp.density[i] ) ;
    }
}

void SimpleMaterialPropertiesHost::read(std::istream& outf, SimpleCellProperties& cellProp) {
    binaryIO::read(outf, cellProp.numMats );
    for( unsigned i=0; i< cellProp.numMats; ++i ) {
        binaryIO::read(outf, cellProp.matID[i] ) ;
    }
    for( unsigned i=0; i< cellProp.numMats; ++i ) {
        binaryIO::read(outf, cellProp.density[i] ) ;
    }
}

void SimpleMaterialPropertiesHost::write(std::ostream& outf) const{
    binaryIO::write(outf, ptr->numCells );
    for( unsigned i=0; i< ptr->numCells; ++i ) {
        write( outf, ptr->props[i] );
    }
}

void SimpleMaterialPropertiesHost::read(std::istream& infile) {
	unsigned num = 0;
    binaryIO::read(infile, num);
    dtor( ptr );
    ctor( ptr, num );

    for( unsigned i=0; i<num; ++i ){
        read(infile, ptr->props[i] );
    }
}


void SimpleMaterialPropertiesHost::write( const std::string& filename ) const {
    std::ofstream outfile;

    outfile.open( filename.c_str(), std::ios::binary | std::ios::out);
    if( ! outfile.is_open() ) {
        fprintf(stderr, "SimpleMaterialPropertiesHost::write -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
        exit(1);
    }
    assert( outfile.good() );
    outfile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
    write( outfile );
    outfile.close();
}

void SimpleMaterialPropertiesHost::read( const std::string& filename ) {
    std::ifstream infile;
    if( infile.is_open() ) {
        infile.close();
    }
    infile.open( filename.c_str(), std::ios::binary | std::ios::in);

    if( ! infile.is_open() ) {
        fprintf(stderr, "SimpleMaterialPropertiesHost::read -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
        exit(1);
    }
    assert( infile.good() );
    infile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
    read(infile);
    infile.close();
}

