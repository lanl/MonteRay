#ifndef COLLISIONPOINTS_HH_
#define COLLISIONPOINTS_HH_

#include <fstream>

#include "MonteRayDefinitions.hh"

namespace MonteRay{

typedef unsigned CollisionPointsSize_t;

typedef float* CollisionPosition_t;
typedef float* CollisionDirection_t;

struct gpuParticle_t {
	gpuFloatType_t pos[3];
	gpuFloatType_t dir[3];
    gpuFloatType_t energy;
    gpuFloatType_t weight;
    unsigned index;
};

struct CollisionPoints {
    CollisionPointsSize_t capacity;
    CollisionPointsSize_t size;
    gpuParticle_t* points;
};

void ctor(CollisionPoints* ptr, CollisionPointsSize_t num);
void dtor(CollisionPoints* ptr);
void copy(CollisionPoints* pCopy, const CollisionPoints* const pOrig );

#ifdef CUDA
__device__ __host__
#endif
CollisionPointsSize_t capacity(CollisionPoints* ptr);

#ifdef CUDA
__device__ __host__
#endif
CollisionPointsSize_t size(CollisionPoints* ptr);

#ifdef CUDA
__device__ __host__
#endif
CollisionPosition_t getPosition( CollisionPoints* ptr, CollisionPointsSize_t i);

#ifdef CUDA
__device__ __host__
#endif
CollisionDirection_t getDirection( CollisionPoints* ptr, CollisionPointsSize_t i);

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getEnergy( CollisionPoints* ptr, CollisionPointsSize_t i);

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getWeight( CollisionPoints* ptr, CollisionPointsSize_t i);

#ifdef CUDA
__device__ __host__
#endif
unsigned getIndex( CollisionPoints* ptr, CollisionPointsSize_t i);

#ifdef CUDA
__device__ __host__
#endif
void clear(CollisionPoints* ptr );

#ifdef CUDA
__device__ __host__
#endif
gpuParticle_t pop(CollisionPoints* ptr );

#ifdef CUDA
__device__ __host__
#endif
gpuParticle_t getParticle(CollisionPoints* ptr, CollisionPointsSize_t i);


class CollisionPointsHost {
public:
    CollisionPointsHost( unsigned num);

    ~CollisionPointsHost();

    void CopyToGPU(void);
    void copyToGPU(void);

    CollisionPointsSize_t capacity(void) const { return MonteRay::capacity(ptrPoints); }
    CollisionPointsSize_t size(void) const { return MonteRay::size(ptrPoints); }

    CollisionPosition_t getPosition( unsigned i) const { return MonteRay::getPosition( ptrPoints, i); }
    CollisionDirection_t getDirection( unsigned i) const { return MonteRay::getDirection( ptrPoints, i); }

    gpuFloatType_t getX(unsigned i) const { return getPosition(i)[0]; }
    gpuFloatType_t getY(unsigned i) const { return getPosition(i)[1]; }
    gpuFloatType_t getZ(unsigned i) const { return getPosition(i)[2]; }
    gpuFloatType_t getU(unsigned i) const { return getDirection(i)[0]; }
    gpuFloatType_t getV(unsigned i) const { return getDirection(i)[1]; }
    gpuFloatType_t getW(unsigned i) const { return getDirection(i)[2]; }
    gpuFloatType_t getEnergy(unsigned i) const { return MonteRay::getEnergy( ptrPoints, i); }
    gpuFloatType_t getWeight(unsigned i) const { return MonteRay::getWeight( ptrPoints, i); }
    gpuFloatType_t getIndex(unsigned i) const { return MonteRay::getIndex( ptrPoints, i); }
    gpuParticle_t getParticle(unsigned i) { return MonteRay::getParticle( ptrPoints, i); }


    void add( const gpuParticle_t& );
    void add( const gpuParticle_t*, unsigned N=1 );
    void add( const void*, unsigned N=1 );

    void add( gpuFloatType_t x, gpuFloatType_t y, gpuFloatType_t z,
              gpuFloatType_t u, gpuFloatType_t v, gpuFloatType_t w,
              gpuFloatType_t energy, gpuFloatType_t weight, unsigned index);

    void clear(void) { MonteRay::clear( ptrPoints ); }

    gpuParticle_t pop(void) { return MonteRay::pop( ptrPoints ); }

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

    void writeParticle( const gpuParticle_t& );
    void writeBank();

    gpuParticle_t readParticle(void);
    void  read(std::fstream& infile);

    void readToMemory( const std::string& file );
    bool readToBank( const std::string& file, unsigned start );

    unsigned getNumCollisionsOnFile(void) const { return numCollisionOnFile; }
    unsigned getVersion(void) const { return currentVersion; }

    bool isCudaCopyMade(void) const { return cudaCopyMade; }

private:
    CollisionPoints* ptrPoints;
    unsigned numCollisionOnFile;
    unsigned currentVersion;
    unsigned long long position; ///< position in file (particle number, starting at 1)
    unsigned long long headerPos; ///< position in file in bytes immediately after header (starting position of the first particle record)
    unsigned currentParticlePos;

    CollisionPoints* temp;
    bool cudaCopyMade;

public:
    CollisionPoints* ptrPoints_device;

};

}
#endif /* COLLISIONPOINTS_HH_ */
