#include "CollisionPoints.h"

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstring>

#include "GPUErrorCheck.hh"
#include "MonteRay_binaryIO.hh"

namespace MonteRay{

void ctor(CollisionPoints* ptr, CollisionPointsSize_t num){
    if( num <=0 ) { num = 1; }
    ptr->capacity = num;
    ptr->size = 0;

    CollisionPointsSize_t allocSize = sizeof(gpuParticle_t)*num;
    ptr->points = (gpuParticle_t*) malloc( allocSize );
}

void dtor(CollisionPoints* ptr){
    free( ptr->points );
}

void copy(CollisionPoints* pCopy, const CollisionPoints* const pOrig ){
    CollisionPointsSize_t num = pOrig->capacity;
    if( num <=0 ) { num = 1; }

    ctor( pCopy, num);
    pCopy->size = pOrig->size;
    std::memcpy( pCopy->points, pOrig->points, pOrig->size * sizeof( gpuParticle_t ) );
}

#ifdef CUDA
__device__ __host__
#endif
CollisionPointsSize_t capacity(CollisionPoints* ptr){
    return ptr->capacity;
}

#ifdef CUDA
__device__ __host__
#endif
CollisionPointsSize_t size(CollisionPoints* ptr) {
    return ptr->size;
}

#ifdef CUDA
__device__ __host__
#endif
CollisionPosition_t getPosition( CollisionPoints* ptr, CollisionPointsSize_t i){
    return ptr->points[i].pos;
}

#ifdef CUDA
__device__ __host__
#endif
CollisionDirection_t getDirection( CollisionPoints* ptr, CollisionPointsSize_t i){
    return ptr->points[i].dir;
}

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getEnergy( CollisionPoints* ptr, CollisionPointsSize_t i){
    return ptr->points[i].energy;
}

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getWeight( CollisionPoints* ptr, CollisionPointsSize_t i){
    return ptr->points[i].weight;
}

#ifdef CUDA
__device__ __host__
#endif
unsigned getIndex( CollisionPoints* ptr, CollisionPointsSize_t i) {
    return ptr->points[i].index;
}

#ifdef CUDA
__device__ __host__
#endif
void clear(CollisionPoints* ptr ) {
    ptr->size = 0;
}

#ifdef CUDA
__device__ __host__
#endif
gpuParticle_t pop(CollisionPoints* ptr ) {

#if !defined( RELEASE )
    if( ptr->size == 0 ) {
        printf("pop(CollisionPoints*) -- no points.  %s %d\n", __FILE__, __LINE__);
        ABORT( "CollisionPoints.cu -- pop" );
    }
#endif

    ptr->size -= 1;
    return ptr->points[ptr->size];
}

#ifdef CUDA
__device__ __host__
#endif
gpuParticle_t getParticle(CollisionPoints* ptr, CollisionPointsSize_t i){
#if !defined( RELEASE )
    if( i >= ptr->size ) {
        printf("pop(CollisionPoints*) -- index exceeds size.  %s %d\n", __FILE__, __LINE__);
        ABORT( "CollisionPoints.cu -- getParticle" );
    }
#endif
    return ptr->points[i];
}

CollisionPointsHost::CollisionPointsHost( unsigned num) :
    ptrPoints( new CollisionPoints ),
    numCollisionOnFile( 0 ),
    currentVersion( 0 ),
    position( 0 ),
    headerPos(0 ),
    currentParticlePos(0)
{
    ctor( ptrPoints, num );
    cudaCopyMade = false;
    temp = NULL;
}

CollisionPointsHost::~CollisionPointsHost() {
    dtor( ptrPoints );
    delete ptrPoints;

    if( io.is_open() ) {
        if( iomode == "out" ){
            closeOutput();
        } else if( iomode == "in" ){
            closeInput();
        }
    }

#ifdef CUDA
    if( cudaCopyMade ) {
        cudaFree( temp->points );
        cudaFree( ptrPoints_device );
        delete temp;
    }
#endif
}

void CollisionPointsHost::add( gpuFloatType_t x, gpuFloatType_t y, gpuFloatType_t z,
        gpuFloatType_t u, gpuFloatType_t v, gpuFloatType_t w,
        gpuFloatType_t energy, gpuFloatType_t weight, unsigned index) {
    gpuParticle_t particle;
    particle.pos[0] = x;
    particle.pos[1] = y;
    particle.pos[2] = z;
    particle.dir[0] = u;
    particle.dir[1] = v;
    particle.dir[2] = w;
    particle.energy = energy;
    particle.weight = weight;
    particle.index = index;
    add( particle );
}

void CollisionPointsHost::add( const gpuParticle_t& particle) {
    CollisionPointsSize_t currentLocation = size();
    if( currentLocation >= ptrPoints->capacity ) {
        fprintf(stderr, "CollisionPointsHost::add -- index > number of allocated points.  %s %d\n", __FILE__, __LINE__);
        exit(1);
    }
    ptrPoints->points[currentLocation] = particle;
    ptrPoints->size += 1;
}

void CollisionPointsHost::add( const gpuParticle_t* particle, unsigned N ) {
    CollisionPointsSize_t currentLocation = size();
    if( currentLocation+N-1 >= ptrPoints->capacity ) {
        fprintf(stderr, "CollisionPointsHost::add -- index > number of allocated points.  %s %d\n", __FILE__, __LINE__);
        exit(1);
    }
    std::memcpy(ptrPoints->points+currentLocation, particle, N*sizeof( gpuParticle_t ) );
//    for( unsigned i=0; i<N; ++i) {
//    	ptrPoints->points[currentLocation+i] = particle[i];
//    }
    ptrPoints->size += N;
}

void CollisionPointsHost::add( const void* voidPtrParticle, unsigned N ) {
	const gpuParticle_t* ptrParticle = (const gpuParticle_t*) voidPtrParticle;
	add( ptrParticle, N);
}

void CollisionPointsHost::writeHeader(std::fstream& infile){
    infile.seekp(0, std::ios::beg); // reposition to start of file

    unsigned version = currentVersion;
    binaryIO::write(infile,version);
    binaryIO::write(infile,numCollisionOnFile);

    headerPos = infile.tellg();
}

void CollisionPointsHost::CopyToGPU(void) {
 	copyToGPU();
 }


void CollisionPointsHost::copyToGPU(void) {
#ifdef CUDA

        if( !cudaCopyMade ) {
        	// first pass allocate memory

        	cudaCopyMade = true;

        	temp = new CollisionPoints;

        	// allocate target struct
        	CUDA_CHECK_RETURN( cudaMalloc(&ptrPoints_device, sizeof( CollisionPoints) ));

        	// allocate target dynamic memory
        	CUDA_CHECK_RETURN( cudaMalloc(&(temp->points), sizeof( gpuParticle_t ) * capacity() ));

        }  else {
        	if( ptrPoints->capacity != temp->capacity ) {
        		// resize

        		cudaFree( temp->points );
        		// allocate target dynamic memory
        		CUDA_CHECK_RETURN( cudaMalloc(&(temp->points), sizeof( gpuParticle_t ) * capacity() ));
        	}
        }

    	temp->capacity = ptrPoints->capacity;
    	temp->size = ptrPoints->size;

        // copy struct data
        CUDA_CHECK_RETURN( cudaMemcpy(ptrPoints_device, temp, sizeof( CollisionPoints ), cudaMemcpyHostToDevice));

        // copy points data into allocated memory
        CUDA_CHECK_RETURN( cudaMemcpy(temp->points, ptrPoints->points, sizeof( gpuParticle_t ) * capacity(), cudaMemcpyHostToDevice));
#endif
    }


void CollisionPointsHost::readHeader(std::fstream& infile){
    if( ! infile.good() ) {
        fprintf(stderr, "CollisionPointsHost::readHeader -- Failure prior to reading header.  %s %d\n", __FILE__, __LINE__);
        exit(1);
    }
    try{
        binaryIO::read(infile,currentVersion);
        binaryIO::read(infile,numCollisionOnFile);
    }
    catch( std::iostream::failure& e  ) {
        std::string message = "CollisionPointsHost::readHeader -- Failure during reading of header. -- ";
        if( infile.eof() ) {
            message += "End-of-file failure";
        } else {
            message += "Unknown failure";
        }
        fprintf(stderr, "CollisionPointsHost::readHeader -- %s.  %s %d\n", message.c_str(), __FILE__, __LINE__);
        exit(1);
    }
}

void CollisionPointsHost::setFilename(const std::string& file ) {
    filename = file;
}

void CollisionPointsHost::openOutput( const std::string& file){
    setFilename( file );
    openOutput(io);
}

void CollisionPointsHost::openOutput(std::fstream& outfile) {
    iomode = "out";
    outfile.open( filename.c_str(), std::ios::binary | std::ios::out);
    if( ! outfile.is_open() ) {
        fprintf(stderr, "CollisionPointsHost::openOutput -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
        exit(1);
    }
    assert( outfile.good() );
    outfile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
    writeHeader(outfile);
}

void CollisionPointsHost::updateHeader(std::fstream& outfile) {
    outfile.seekg(0, std::ios::beg); // reposition to start of file

    binaryIO::write(io,currentVersion);
    binaryIO::write(io,numCollisionOnFile);
}

void CollisionPointsHost::resetFile(void){
    numCollisionOnFile = 0;
    updateHeader(io);
}

void CollisionPointsHost::openInput( const std::string& file){
    setFilename( file );
    openInput(io);
}

void CollisionPointsHost::openInput( std::fstream& infile){
    iomode = "in";
    if( infile.is_open() ) {
        closeInput(infile);
    }
    infile.open( filename.c_str(), std::ios::binary | std::ios::in);

    if( ! infile.is_open() ) {
        fprintf(stderr, "CollisionPointsHost::openInput -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
        exit(1);
    }
    assert( infile.good() );
    infile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
    readHeader(infile);
}

void CollisionPointsHost::closeOutput(void) {
    closeOutput(io);
}

void CollisionPointsHost::closeOutput(std::fstream& outfile) {
    if( outfile.is_open() ) {
        updateHeader(outfile);
        outfile.close();
    }
}

void CollisionPointsHost::closeInput(void) {
    closeInput(io);
}

///\brief Close the input file
void CollisionPointsHost::closeInput(std::fstream& infile) {
    if( infile.is_open() ) {
        infile.close();
    }
}

void CollisionPointsHost::writeParticle(const gpuParticle_t& particle){
    binaryIO::write(io, particle );
    ++numCollisionOnFile;
}

gpuParticle_t CollisionPointsHost::readParticle(void){
    ++currentParticlePos;
    if( currentParticlePos > numCollisionOnFile ) {
        fprintf(stderr, "CollisionPointsHost::readParticle -- Exhausted particles on the file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
        exit(1);
    }
    gpuParticle_t particle;
    try{
        binaryIO::read(io, particle);
    }
    catch( std::fstream::failure& e  ) {
        std::string message = "CollisionPointsHost::readParticle -- Failure during reading of a collision. -- ";
        if( io.eof() ) {
            message += "End-of-file failure";
        } else {
            message += "Unknown failure";
        }
        fprintf(stderr, "CollisionPointsHost::readParticle -- %s.  %s %d\n", message.c_str(), __FILE__, __LINE__);
        exit(1);
    }
    return particle;
}

void CollisionPointsHost::readToMemory( const std::string& file ){
    openInput( file );
    dtor( ptrPoints );
    ctor( ptrPoints, getNumCollisionsOnFile() );

    for( unsigned i=0; i< getNumCollisionsOnFile(); ++i ) {
        add( readParticle() );
    }
    closeInput();
}


bool CollisionPointsHost::readToBank( const std::string& file, unsigned start ){
    openInput( file );
    unsigned offset = start * ( sizeof(gpuFloatType_t)*8+sizeof(unsigned));
    io.seekg( offset, std::ios::cur); // reposition to start of file

    clear();
    unsigned nRead = 0;
    for( unsigned i=0; i< capacity(); ++i ) {
        add( readParticle() );
        ++nRead;
        if( numCollisionOnFile == nRead + start) {
        	break;
        }
    }
    closeInput();

    if( nRead < capacity() ) {
    	return true; // return end = true
    }
    return false; // return end = false
}

}
