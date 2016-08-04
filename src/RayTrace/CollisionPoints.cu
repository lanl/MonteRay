#include "CollisionPoints.h"

#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "binaryIO.h"
namespace MonteRay{

void copy(CollisionPosition_t* pCopy, const CollisionPosition_t* const pOrig ){
    pCopy->x = pOrig->x;
    pCopy->y = pOrig->y;
    pCopy->z = pOrig->z;
}

void copy(CollisionPosition_t Copy, const CollisionPosition_t& Orig ) {
    Copy.x = Orig.x;
    Copy.y = Orig.y;
    Copy.z = Orig.z;
}

void copy(CollisionDirection_t* pCopy, const CollisionDirection_t* const pOrig ){
    pCopy->u = pOrig->u;
    pCopy->v = pOrig->v;
    pCopy->w = pOrig->w;
}

void copy(CollisionDirection_t Copy, const CollisionDirection_t& Orig ) {
    Copy.u = Orig.u;
    Copy.v = Orig.v;
    Copy.w = Orig.w;
}

void ctor(CollisionPoints* ptr, CollisionPointsSize_t num){
    if( num <=0 ) { num = 1; }
    ptr->capacity = num;
    ptr->size = 0;

    CollisionPointsSize_t allocSize = sizeof(CollisionPosition_t)*num;
    ptr->pos = (CollisionPosition_t*) malloc( allocSize );

    allocSize = sizeof(CollisionDirection_t)*num;
    ptr->dir = (CollisionDirection_t*) malloc( allocSize );

    allocSize = sizeof(gpuFloatType_t)*num;
    ptr->energy = (gpuFloatType_t*) malloc( allocSize );

    allocSize = sizeof(gpuFloatType_t)*num;
    ptr->weight = (gpuFloatType_t*) malloc( allocSize );

    allocSize = sizeof(unsigned)*num;
    ptr->index = (unsigned*) malloc( allocSize );
}

void dtor(CollisionPoints* ptr){
    free( ptr->pos );
    free( ptr->dir );
    free( ptr->energy );
    free( ptr->weight );
    free( ptr->index );
}

void copy(CollisionPoints* pCopy, const CollisionPoints* const pOrig ){
    CollisionPointsSize_t num = pOrig->capacity;
    if( num <=0 ) { num = 1; }
    pCopy->size = pOrig->size;

    ctor( pCopy, num);
    for( CollisionPointsSize_t i=0; i<pCopy->size; ++i ){
        copy( pCopy->pos[i], pOrig->pos[i] );
        copy( pCopy->dir[i], pOrig->dir[i] );

        pCopy->energy[i] = pOrig->energy[i];
        pCopy->weight[i] = pOrig->weight[i];
        pCopy->index[i] = pOrig->index[i];
    }
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
    return ptr->pos[i];
}

#ifdef CUDA
__device__ __host__
#endif
CollisionDirection_t getDirection( CollisionPoints* ptr, CollisionPointsSize_t i){
    return ptr->dir[i];
}

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getEnergy( CollisionPoints* ptr, CollisionPointsSize_t i){
    return ptr->energy[i];
}

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getWeight( CollisionPoints* ptr, CollisionPointsSize_t i){
    return ptr->weight[i];
}

#ifdef CUDA
__device__ __host__
#endif
unsigned getIndex( CollisionPoints* ptr, CollisionPointsSize_t i) {
    return ptr->index[i];
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
        abort;
    }
#endif

    gpuParticle_t particle;
    particle = getParticle(ptr, ptr->size-1);
    ptr->size -= 1;
    return particle;
}

#ifdef CUDA
__device__ __host__
#endif
gpuParticle_t getParticle(CollisionPoints* ptr, CollisionPointsSize_t i){
    gpuParticle_t particle;
#if !defined( RELEASE )
    if( i >= ptr->size ) {
        printf("pop(CollisionPoints*) -- index exceeds size.  %s %d\n", __FILE__, __LINE__);
        abort;
    }
#endif
    CollisionPointsSize_t currentLocation = i;
    particle.pos = ptr->pos[currentLocation];
    particle.dir = ptr->dir[currentLocation];
    particle.energy = ptr->energy[currentLocation];
    particle.weight = ptr->weight[currentLocation];
    particle.index = ptr->index[currentLocation];
    return particle;
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
        cudaFree( temp->pos );
        cudaFree( temp->dir );
        cudaFree( temp->energy );
        cudaFree( temp->weight );
        cudaFree( temp->index );
        cudaFree( ptrPoints_device );
        delete temp;
    }
#endif
}

void CollisionPointsHost::add( gpuFloatType_t x, gpuFloatType_t y, gpuFloatType_t z,
        gpuFloatType_t u, gpuFloatType_t v, gpuFloatType_t w,
        gpuFloatType_t energy, gpuFloatType_t weight, unsigned index) {
    gpuParticle_t particle;
    CollisionPosition_t pos; pos.x=x; pos.y=y; pos.z=z;
    CollisionDirection_t dir; dir.u=u; dir.v=v; dir.w=w;
    particle.pos = pos;
    particle.dir = dir;
    particle.energy = energy;
    particle.weight = weight;
    particle.index = index;
    add( particle );
}

void CollisionPointsHost::add( gpuParticle_t particle) {
    CollisionPointsSize_t currentLocation = size();
    if( currentLocation >= ptrPoints->capacity ) {
        fprintf(stderr, "CollisionPointsHost::add -- index > number of allocated points.  %s %d\n", __FILE__, __LINE__);
        exit(1);
    }
    ptrPoints->pos[currentLocation] = particle.pos;
    ptrPoints->dir[currentLocation] = particle.dir;
    ptrPoints->energy[currentLocation] = particle.energy;
    ptrPoints->weight[currentLocation] = particle.weight;
    ptrPoints->index[currentLocation] = particle.index;
    ptrPoints->size += 1;
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
        gpuErrchk( cudaPeekAtLastError() );

        if( !cudaCopyMade ) {
        	// first pass allocate memory

        	cudaCopyMade = true;

        	unsigned num = sizeof( gpuFloatType_t ) * capacity();

        	temp = new CollisionPoints;
        	temp->capacity = ptrPoints->capacity;
        	temp->size = ptrPoints->size;

        	// allocate target struct

        	CUDA_CHECK_RETURN( cudaMalloc(&ptrPoints_device, sizeof( CollisionPoints) ));
        	gpuErrchk( cudaPeekAtLastError() );

        	// allocate target dynamic memory
        	CUDA_CHECK_RETURN( cudaMalloc(&temp->pos, sizeof( CollisionPosition_t ) * capacity() ));
        	gpuErrchk( cudaPeekAtLastError() );

        	CUDA_CHECK_RETURN( cudaMalloc(&temp->dir, sizeof( CollisionDirection_t ) * capacity() ));
        	gpuErrchk( cudaPeekAtLastError() );

        	CUDA_CHECK_RETURN( cudaMalloc(&temp->energy, sizeof( gpuFloatType_t ) * capacity() ));
        	gpuErrchk( cudaPeekAtLastError() );

        	CUDA_CHECK_RETURN( cudaMalloc(&temp->weight, sizeof( gpuFloatType_t ) * capacity() ));
        	gpuErrchk( cudaPeekAtLastError() );

        	CUDA_CHECK_RETURN( cudaMalloc(&temp->index, sizeof( unsigned ) * capacity() ));
        	gpuErrchk( cudaPeekAtLastError() );
        }

        // copy data
        CUDA_CHECK_RETURN( cudaMemcpy(ptrPoints_device, temp, sizeof( CollisionPoints ), cudaMemcpyHostToDevice));
        gpuErrchk( cudaPeekAtLastError() );

        CUDA_CHECK_RETURN( cudaMemcpy(temp->pos, ptrPoints->pos, sizeof( CollisionPosition_t ) * capacity(), cudaMemcpyHostToDevice));
        gpuErrchk( cudaPeekAtLastError() );

        CUDA_CHECK_RETURN( cudaMemcpy(temp->dir, ptrPoints->dir, sizeof( CollisionDirection_t ) * capacity(), cudaMemcpyHostToDevice));
        gpuErrchk( cudaPeekAtLastError() );

        CUDA_CHECK_RETURN( cudaMemcpy(temp->energy, ptrPoints->energy, sizeof( gpuFloatType_t ) * capacity(), cudaMemcpyHostToDevice));
        gpuErrchk( cudaPeekAtLastError() );

        CUDA_CHECK_RETURN( cudaMemcpy(temp->weight, ptrPoints->weight, sizeof( gpuFloatType_t ) * capacity(), cudaMemcpyHostToDevice));
        gpuErrchk( cudaPeekAtLastError() );

        CUDA_CHECK_RETURN( cudaMemcpy(temp->index, ptrPoints->index, sizeof( unsigned ) * capacity(), cudaMemcpyHostToDevice));
        gpuErrchk( cudaPeekAtLastError() );
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
    binaryIO::write(io, particle.pos.x );
    binaryIO::write(io, particle.pos.y );
    binaryIO::write(io, particle.pos.z );
    binaryIO::write(io, particle.dir.u );
    binaryIO::write(io, particle.dir.v );
    binaryIO::write(io, particle.dir.w );
    binaryIO::write(io, particle.energy );
    binaryIO::write(io, particle.weight );
    binaryIO::write(io, particle.index );
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
        binaryIO::read(io, particle.pos.x );
        binaryIO::read(io, particle.pos.y );
        binaryIO::read(io, particle.pos.z );
        binaryIO::read(io, particle.dir.u );
        binaryIO::read(io, particle.dir.v );
        binaryIO::read(io, particle.dir.w );
        binaryIO::read(io, particle.energy );
        binaryIO::read(io, particle.weight );
        binaryIO::read(io, particle.index );
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
