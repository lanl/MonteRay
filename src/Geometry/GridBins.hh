#ifndef GRIDBINS_H_
#define GRIDBINS_H_

#include <limits.h>
#include <stdio.h>        /* perror */
#include <errno.h>        /* errno */
#include <stdlib.h>
#include <vector>
#include <limits>
#include <string>
#include <iostream>
#include <fstream>

#include "MonteRayTypes.hh"
#include "MonteRayVector3D.hh"
#include "MonteRayCopyMemory.hh"

namespace MonteRay{

class HashBins;

typedef gpuFloatType_t float_t;
typedef MonteRay::Vector3D<gpuRayFloat_t> Position_t;
typedef MonteRay::Vector3D<gpuRayFloat_t> Direction_t;

class GridBins : public CopyMemoryBase<GridBins> {
public:
    unsigned numVertices = 0;
    unsigned numXY = 0;
    unsigned num[3] = { 0, 0, 0};
    unsigned offset[3] = { 0, 0, 0};
    int regular[3] = { 0, 0, 0};

    float_t delta[3]  = {0.0, 0.0, 0.0};
    float_t minMax[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    float_t* vertices = nullptr;
    HashBins* hash[3] = { nullptr, nullptr, nullptr};
    unsigned hashSize = 8000;

public:
    using Base = MonteRay::CopyMemoryBase<GridBins> ;

    GridBins();

    ~GridBins();

    CUDA_CALLABLE_MEMBER
    void initialize();

    std::string className(){ return std::string("GridBins");}

    void init() {
        initialize();
    }

    void copyToGPU(void);

    void copy(const GridBins* rhs);

    GridBins&
    operator=( GridBins& rhs );

    template<class READER_T>
    GridBins( READER_T& reader);

    CUDA_CALLABLE_MEMBER
    unsigned getMaxNumVertices() const {
        return MAXNUMVERTICES;
    }

    CUDA_CALLABLE_MEMBER
    unsigned getOffset( const unsigned dim ) const;

    void setVertices(unsigned dim, float_t min, float_t max, unsigned numBins);

    template<typename T>
    void setVertices(const unsigned dim, const std::vector<T>& vertices );

    CUDA_CALLABLE_MEMBER
    float_t getVertex(const unsigned dim, const unsigned index ) const {
        return vertices[ offset[dim] + index ];
    }

    CUDA_CALLABLE_MEMBER
    unsigned getNumVertices(const unsigned dim) const;

    CUDA_CALLABLE_MEMBER
    unsigned getNumBins(unsigned dim) const;

    CUDA_CALLABLE_MEMBER
    bool isRegular( unsigned dim) const;

    CUDA_CALLABLE_MEMBER
    void finalize();

    CUDA_CALLABLE_MEMBER
    unsigned getNumXY() const { return numXY; }

    CUDA_CALLABLE_MEMBER
    float_t min(const unsigned dim) const;

    CUDA_CALLABLE_MEMBER
    float_t max(const unsigned dim) const;

    CUDA_CALLABLE_MEMBER
    int getDimIndex(const unsigned dim, const gpuRayFloat_t pos ) const;

    CUDA_CALLABLE_MEMBER
    unsigned getIndex(const Position_t& particle_pos);

    CUDA_CALLABLE_MEMBER
    bool isIndexOutside(unsigned dim, int i) const;

    CUDA_CALLABLE_MEMBER
    unsigned calcIndex(const int* const indices ) const;

    CUDA_CALLABLE_MEMBER
    bool isOutside(const int* indices ) const;

    unsigned getNumCells() const { return num[0]*num[1]*num[2]; }

    Position_t getCenterPointByIndex(unsigned index ) const;
    Position_t getCenterPointByIndices( const unsigned* const indices ) const;

    void calcIJK(unsigned index, unsigned* indices ) const;

    CUDA_CALLABLE_MEMBER
    unsigned rayTrace(int* global_indices, gpuRayFloat_t* distances, const Position_t& pos, const Position_t& dir, float_t distance,  bool outsideDistances) const;

    CUDA_CALLABLE_MEMBER
    unsigned orderCrossings(int* global_indices, gpuRayFloat_t* distances, unsigned num, const int* const cells, const gpuRayFloat_t* const crossingDistances, unsigned* numCrossings, int* indices, float_t distance, bool outsideDistances ) const;

    CUDA_CALLABLE_MEMBER
    const HashBins* getHashPtr( unsigned dim ) { return hash[dim]; }

    CUDA_CALLABLE_MEMBER
    void getHashLowerUpperBins(unsigned dim, gpuFloatType_t value, unsigned& lower, unsigned& upper) const;
    CUDA_CALLABLE_MEMBER
    void setDefaultHashSize(unsigned n) { hashSize = n;}

    CUDA_CALLABLE_MEMBER
    unsigned getDefaultHashSize( void ) const { return hashSize; }

    void abort(const char* buffer);

    void writeToFile( const std::string& fileName ) const;
    void readFromFile( const std::string& fileName );

    void write(std::ostream& outfile) const;
    void  read(std::istream& infile);
};

//  static methods
float_t getDistance( Position_t& pos1, Position_t& pos2);

CUDA_CALLABLE_MEMBER
unsigned calcCrossings(const float_t* const vertices, unsigned nVertices, int* cells, gpuRayFloat_t* distances, float_t pos, float_t dir, float_t distance, int index );

CUDA_CALLABLE_KERNEL
void
kernelRayTrace(
        void* ptrNumCrossings,
        GridBins* ptrGrid,
        int* ptrCells,
        gpuRayFloat_t* ptrDistances,
        gpuFloatType_t x, gpuFloatType_t y, gpuFloatType_t z,
        gpuFloatType_t u, gpuFloatType_t v, gpuFloatType_t w,
        gpuFloatType_t distance,
        bool outsideDistances);

template<class READER_T>
GridBins::GridBins( READER_T& reader) : GridBins() {
    if( reader.getGeometryString() != "XYZ" )  {
        throw std::runtime_error( "Invalid Geometry type -- MonteRay::GridBins only supports XYZ");
    }
    for( unsigned d=0; d < 3; ++d) {
        std::vector<double> vertices = reader.getVertices(d);
        setVertices(d, vertices );
    }
    finalize();
}

}
#endif /* GRIDBINS_H_ */
