#ifndef RAYLISTINTERFACE_HH_
#define RAYLISTINTERFACE_HH_

#include <cstring>
#include <fstream>

#include "MonteRayTypes.hh"
#include "RayList.hh"

namespace MonteRay{

template<unsigned N = 1>
class RayListInterface {
public:
    using RAY_T = MonteRay::Ray_t<N>; 
    using RAYLIST_T = MonteRay::RayList_t<N>;
    using RayListSize_t = unsigned;
    using DetectorIndex_t = unsigned;
    using ParticleType_t = MonteRay::ParticleType_t;

    RayListInterface( unsigned num);

    ~RayListInterface();

    RayListSize_t capacity(void) const { return ptrPoints->capacity(); }
    RayListSize_t size(void) const { return ptrPoints->size(); }
    auto data(void) { return ptrPoints->data(); }
    auto data(void) const { return ptrPoints->data(); }

    const auto& getPosition( unsigned i) const { return ptrPoints->getPosition(i); }
    const auto& getDirection( unsigned i) const { return ptrPoints->getDirection(i); }

    gpuFloatType_t getX(unsigned i) const { return getPosition(i)[0]; }
    gpuFloatType_t getY(unsigned i) const { return getPosition(i)[1]; }
    gpuFloatType_t getZ(unsigned i) const { return getPosition(i)[2]; }
    gpuFloatType_t getU(unsigned i) const { return getDirection(i)[0]; }
    gpuFloatType_t getV(unsigned i) const { return getDirection(i)[1]; }
    gpuFloatType_t getW(unsigned i) const { return getDirection(i)[2]; }
    gpuFloatType_t getEnergy(unsigned i, unsigned index=0) const { return ptrPoints->getEnergy(i,index); }
    gpuFloatType_t getWeight(unsigned i, unsigned index=0) const { return ptrPoints->getWeight(i,index); }
    unsigned getIndex(unsigned i) const { return ptrPoints->getIndex(i); }
    DetectorIndex_t getDetectorIndex(unsigned i) const { return ptrPoints->getDetectorIndex(i); }
    ParticleType_t getParticleType(unsigned i) const { return ptrPoints->getParticleType(i); }

    RAY_T getParticle(unsigned i) const { return ptrPoints->getParticle(i); }

    void add( const RAY_T& ray) { ptrPoints->add( ray ); }

    void add( const RAY_T* rayArray, unsigned num=1 ) { for( unsigned i=0; i<num; ++i) add( rayArray[i] );}
    void add( const void* ptrRay, unsigned num=1 );

    void add( gpuFloatType_t x, gpuFloatType_t y, gpuFloatType_t z,
            gpuFloatType_t u, gpuFloatType_t v, gpuFloatType_t w,
            gpuFloatType_t energy, gpuFloatType_t weight, gpuFloatType_t time,
            unsigned index, DetectorIndex_t detectorIndex, ParticleType_t particleType );

    void clear(void) { ptrPoints->clear(); }

    RAY_T pop(void) { return ptrPoints->pop(); }

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

    void writeParticle( const RAY_T& );
    void writeBank();

    RAY_T readParticle(void);
    void  read(std::fstream& infile);

    void readToMemory( const std::string& file );
    bool readToBank( const std::string& file, unsigned start );

    unsigned getNumCollisionsOnFile(void) const { return numCollisionOnFile; }
    unsigned getVersion(void) const { return currentVersion; }

    bool isCudaCopyMade(void) const { return cudaCopyMade; }

    void debugPrint() const;
    void printParticle(unsigned i, const RAY_T& particle ) const;

    const RAYLIST_T* getPtrPoints() const { return ptrPoints; }

private:

    unsigned numCollisionOnFile = 0 ;
    unsigned currentVersion = 1;
    unsigned long long position = 0; ///< position in file (particle number, starting at 1)
    unsigned long long headerPos = 0; ///< position in file in bytes immediately after header (starting position of the first particle record)
    unsigned currentParticlePos = 0;
    bool cudaCopyMade = false;

    RAYLIST_T* ptrPoints = NULL;

};


typedef RayListInterface<1> ParticleRayListInterface;
}
#endif /* RAYLISTINTERFACE_HH_ */
