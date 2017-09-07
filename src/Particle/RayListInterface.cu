#include "RayListInterface.hh"

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstring>

#include "GPUErrorCheck.hh"
#include "MonteRay_binaryIO.hh"

namespace MonteRay{

RayListInterface::~RayListInterface() {
    delete ptrPoints;

    if( io.is_open() ) {
        if( iomode == "out" ){
            closeOutput();
        } else if( iomode == "in" ){
            closeInput();
        }
    }
}

void RayListInterface::add( gpuFloatType_t x, gpuFloatType_t y, gpuFloatType_t z,
        gpuFloatType_t u, gpuFloatType_t v, gpuFloatType_t w,
        gpuFloatType_t energy, gpuFloatType_t weight,
        unsigned index, DetectorIndex_t detectorIndex, ParticleType_t particleType) {
	ParticleRay_t particle;
    particle.pos[0] = x;
    particle.pos[1] = y;
    particle.pos[2] = z;
    particle.dir[0] = u;
    particle.dir[1] = v;
    particle.dir[2] = w;
    particle.energy[0] = energy;
    particle.weight[0] = weight;
    particle.index = index;
    particle.detectorIndex = detectorIndex;
    particle.particleType = particleType;
    add( particle );
}

void RayListInterface::add( const ParticleRay_t& particle) {
	ptrPoints->add( particle );
}

void RayListInterface::add( const ParticleRay_t* particle, unsigned N ) {
	for( unsigned i=0; i<N; ++i){
		add(particle[i]);
	}
}

void RayListInterface::add( const void* voidPtrParticle, unsigned N ) {
	const ParticleRay_t* ptrParticle = (const ParticleRay_t*) voidPtrParticle;
	add( ptrParticle, N);
}

void RayListInterface::copyToGPU(void) {
	ptrPoints->copyToGPU();
}

void RayListInterface::copyToCPU(void) {
	ptrPoints->copyToCPU();
}

void RayListInterface::writeHeader(std::fstream& infile){
    infile.seekp(0, std::ios::beg); // reposition to start of file

    unsigned version = currentVersion;
    binaryIO::write(infile,version);
    binaryIO::write(infile,numCollisionOnFile);

    headerPos = infile.tellg();
}

void RayListInterface::readHeader(std::fstream& infile){
//	std::cout << "Debug: RayListInterface::readHeader - starting.\n";
    if( ! infile.good() ) {
        fprintf(stderr, "RayListInterface::readHeader -- Failure prior to reading header.  %s %d\n", __FILE__, __LINE__);
        exit(1);
    }
    try{
//    	std::cout << "Debug: RayListInterface::reading version - starting.\n";
        binaryIO::read(infile,currentVersion);
//        std::cout << "Debug: RayListInterface::reading number of collisions on the file - starting.\n";
        binaryIO::read(infile,numCollisionOnFile);
    }
    catch( std::iostream::failure& e  ) {
        std::string message = "RayListInterface::readHeader -- Failure during reading of header. -- ";
        if( infile.eof() ) {
            message += "End-of-file failure";
        } else {
            message += "Unknown failure";
        }
        fprintf(stderr, "RayListInterface::readHeader -- %s.  %s %d\n", message.c_str(), __FILE__, __LINE__);
        exit(1);
    }
}

void RayListInterface::setFilename(const std::string& file ) {
    filename = file;
}

void RayListInterface::openOutput( const std::string& file){
    setFilename( file );
    openOutput(io);
}

void RayListInterface::openOutput(std::fstream& outfile) {
    iomode = "out";
    outfile.open( filename.c_str(), std::ios::binary | std::ios::out);
    if( ! outfile.is_open() ) {
        fprintf(stderr, "RayListInterface::openOutput -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
        exit(1);
    }
    assert( outfile.good() );
    outfile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
    writeHeader(outfile);
}

void RayListInterface::updateHeader(std::fstream& outfile) {
    outfile.seekg(0, std::ios::beg); // reposition to start of file

    binaryIO::write(io,currentVersion);
    binaryIO::write(io,numCollisionOnFile);
}

void RayListInterface::resetFile(void){
    numCollisionOnFile = 0;
    updateHeader(io);
}

void RayListInterface::openInput( const std::string& file){
//	std::cout << "Debug: RayListInterface::openInput(string) - starting -- setting filename.\n";
    setFilename( file );
//    std::cout << "Debug: RayListInterface::openInput(string) - opening input.\n";
    openInput(io);
//    std::cout << "Debug: RayListInterface::openInput(string) - input open.\n";
}

void RayListInterface::openInput( std::fstream& infile){
//	std::cout << "Debug: RayListInterface::openInput(fstream) - starting.\n";
    iomode = "in";
    if( infile.is_open() ) {
        closeInput(infile);
    }
    infile.open( filename.c_str(), std::ios::binary | std::ios::in);

    if( ! infile.is_open() ) {
        fprintf(stderr, "RayListInterface::openInput -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
        exit(1);
    }
    assert( infile.good() );
    infile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
//    std::cout << "Debug: RayListInterface::openInput(fstream) - reading header.\n";
    readHeader(infile);
//    std::cout << "Debug: RayListInterface::openInput(fstream) - reading header done.\n";
}

void RayListInterface::closeOutput(void) {
    closeOutput(io);
}

void RayListInterface::closeOutput(std::fstream& outfile) {
    if( outfile.is_open() ) {
        updateHeader(outfile);
        outfile.close();
    }
}

void RayListInterface::closeInput(void) {
    closeInput(io);
}

///\brief Close the input file
void RayListInterface::closeInput(std::fstream& infile) {
    if( infile.is_open() ) {
        infile.close();
    }
}

void RayListInterface::writeParticle(const ParticleRay_t& particle){
    particle.write(io);
    ++numCollisionOnFile;
}

void RayListInterface::printParticle(unsigned i, const ParticleRay_t& particle ) const {
	std::cout << "Debug: RayListInterface::printParticle -- i=" << i;
	std::cout << " x= " << particle.pos[0];
	std::cout << " y= " << particle.pos[1];
	std::cout << " z= " << particle.pos[2];
	std::cout << " u= " << particle.dir[0];
	std::cout << " v= " << particle.dir[1];
	std::cout << " w= " << particle.dir[2];
	std::cout << " E= " << particle.energy;
	std::cout << " W= " << particle.weight;
	std::cout << " index= " << particle.index;
	std::cout << " detector index= " << particle.detectorIndex;
	std::cout << " particle type= " << particle.particleType;
	std::cout << "\n";
}

ParticleRay_t RayListInterface::readParticle(void){
    ++currentParticlePos;
    if( currentParticlePos > numCollisionOnFile ) {
        fprintf(stderr, "RayListInterface::readParticle -- Exhausted particles on the file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
        exit(1);
    }
    ParticleRay_t particle;
    try{
    	particle.read(io);
    }
    catch( std::fstream::failure& e  ) {
        std::string message = "RayListInterface::readParticle -- Failure during reading of a collision. -- ";
        if( io.eof() ) {
            message += "End-of-file failure";
        } else {
            message += "Unknown failure";
        }
        fprintf(stderr, "RayListInterface::readParticle -- %s.  %s %d\n", message.c_str(), __FILE__, __LINE__);
        exit(1);
    }
    return particle;
}

void RayListInterface::readToMemory( const std::string& file ){
    openInput( file );

    delete ptrPoints;
    ptrPoints = new CollisionPoints( getNumCollisionsOnFile() );

    for( unsigned i=0; i< getNumCollisionsOnFile(); ++i ) {
        add( readParticle() );
    }
    closeInput();
}

void RayListInterface::writeBank() {
	for( unsigned i=0; i< size(); ++i ) {
		writeParticle( getParticle(i) );
	}
}

bool RayListInterface::readToBank( const std::string& file, unsigned start ){
    openInput( file );
    unsigned offset = start * ( ParticleRay_t::filesize() );
    io.seekg( offset, std::ios::cur); // reposition to offset location

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

void RayListInterface::debugPrint() const {
	for( unsigned i=0; i< size(); ++i ) {
		printParticle( i, getParticle(i) );
	}
}

}
