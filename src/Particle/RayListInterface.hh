#ifndef RAYLISTINTERFACE_HH_
#define RAYLISTINTERFACE_HH_

#include <cstring>
#include <fstream>

#include "MonteRayDefinitions.hh"
#include "GPUErrorCheck.hh"

#include "RayList.hh"

namespace MonteRay{

class RayListInterface {
public:
	typedef MonteRay::ParticleRay_t ParticleRay_t;
	typedef MonteRay::RayListSize_t RayListSize_t;

    RayListInterface( unsigned num) :
        ptrPoints( new CollisionPoints(num) )
	{}

    ~RayListInterface();

    void copyToGPU(void);
    void copyToCPU(void);

    RayListSize_t capacity(void) const { return ptrPoints->capacity(); }
    RayListSize_t size(void) const { return ptrPoints->size(); }

    CollisionPosition_t getPosition( unsigned i) const { return ptrPoints->getPosition(i); }
    CollisionDirection_t getDirection( unsigned i) const { return ptrPoints->getDirection(i); }

    gpuFloatType_t getX(unsigned i) const { return getPosition(i)[0]; }
    gpuFloatType_t getY(unsigned i) const { return getPosition(i)[1]; }
    gpuFloatType_t getZ(unsigned i) const { return getPosition(i)[2]; }
    gpuFloatType_t getU(unsigned i) const { return getDirection(i)[0]; }
    gpuFloatType_t getV(unsigned i) const { return getDirection(i)[1]; }
    gpuFloatType_t getW(unsigned i) const { return getDirection(i)[2]; }
    gpuFloatType_t getEnergy(unsigned i) const { return ptrPoints->getEnergy(i); }
    gpuFloatType_t getWeight(unsigned i) const { return ptrPoints->getWeight(i); }
    unsigned getIndex(unsigned i) const { return ptrPoints->getIndex(i); }
    DetectorIndex_t getDetectorIndex(unsigned i) const { return ptrPoints->getDetectorIndex(i); }
    ParticleType_t getParticleType(unsigned i) const { return ptrPoints->getParticleType(i); }

    ParticleRay_t getParticle(unsigned i) const { return ptrPoints->getParticle(i); }

    void add( const ParticleRay_t& );
    void add( const ParticleRay_t*, unsigned N=1 );
    void add( const void*, unsigned N=1 );

    void add( gpuFloatType_t x, gpuFloatType_t y, gpuFloatType_t z,
              gpuFloatType_t u, gpuFloatType_t v, gpuFloatType_t w,
              gpuFloatType_t energy, gpuFloatType_t weight, unsigned index,
              DetectorIndex_t detectorIndex, ParticleType_t particleType );

    void clear(void) { ptrPoints->clear(); }

    ParticleRay_t pop(void) { return ptrPoints->pop(); }

    std::string filename;
    std::string iomode;
    std::fstream io; ///< Input and output file stream

    void setFilename(const std::string& file );
    void openOutput( const std::string& filename);
    void openOutput( std::fstream& outfile );
    void closeOutput(std::fstream& outfile);
    void closeOutput();

    void openInput( const std::string& file);
    void openInput( std::fstream& infile );
    void closeInput(std::fstream& infile);
    void closeInput();

    void writeHeader(std::fstream& outfile);
    void readHeader(std::fstream& infile);
    void updateHeader(std::fstream& outfile);
    void resetFile(void);

    void writeParticle( const ParticleRay_t& );
    void writeBank();

    ParticleRay_t readParticle(void);
    void  read(std::fstream& infile);

    void readToMemory( const std::string& file );
    bool readToBank( const std::string& file, unsigned start );

    unsigned getNumCollisionsOnFile(void) const { return numCollisionOnFile; }
    unsigned getVersion(void) const { return currentVersion; }

    bool isCudaCopyMade(void) const { return cudaCopyMade; }

    void debugPrint() const;
    void printParticle(unsigned i, const ParticleRay_t& particle ) const;

    const CollisionPoints* getPtrPoints() { return ptrPoints; }

private:

    unsigned numCollisionOnFile = 0 ;
    unsigned currentVersion = 0;
    unsigned long long position = 0; ///< position in file (particle number, starting at 1)
    unsigned long long headerPos = 0; ///< position in file in bytes immediately after header (starting position of the first particle record)
    unsigned currentParticlePos = 0;
    bool cudaCopyMade = false;

    CollisionPoints* ptrPoints = NULL;

};

}
#endif /* RAYLISTINTERFACE_HH_ */
