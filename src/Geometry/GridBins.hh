#ifndef GRIDBINS_H_
#define GRIDBINS_H_

#include <limits.h>
#include <stdio.h>        /* perror */
#include <errno.h>        /* errno */
#include <stdlib.h>
#include <vector>
#include <limits>

#define MAXNUMVERTICES 1001

#include "MonteRayDefinitions.hh"
#include "MonteRayVector3D.hh"
#include "GPUErrorCheck.hh"
#include "MonteRayCopyMemory.t.hh"
#include "HashBins.hh"

namespace MonteRay{

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

    GridBins() : CopyMemoryBase<GridBins>() {
        initialize();
        numVertices = MAXNUMVERTICES*3;
        vertices = (float_t*) MONTERAYHOSTALLOC( numVertices * sizeof( float_t ), false, std::string("GridBins::vertices") );

        for( unsigned i=0; i<numVertices; ++i) {
            vertices[i] = 0.0;
        }

    }

    ~GridBins() {
        if( Base::isCudaIntermediate ) {
            //printf( "GridBins::dtor -- intermediate -- freeing vertices\n" );
            MonteRayDeviceFree( vertices );
        } else {
            MonteRayHostFree( vertices, Base::isManagedMemory );
            vertices = NULL;
            for( unsigned i=0; i<3; ++i) {
                delete hash[i];
            }
        }
    }

    CUDA_CALLABLE_MEMBER
    void initialize() {
        numVertices = 0;
        offset[0] = 0;
        offset[1] = MAXNUMVERTICES;
        offset[2] = MAXNUMVERTICES*2;
        num[0] = 0;
        num[1] = 0;
        num[2] = 0;
        numXY = 0;
        regular[0] = true;
        regular[1] = true;
        regular[2] = true;

        delta[0] = 0.0;
        delta[1] = 0.0;
        delta[2] = 0.0;

        minMax[0] = 0.0;
        minMax[1] = 0.0;
        minMax[2] = 0.0;
        minMax[3] = 0.0;
        minMax[4] = 0.0;
        minMax[5] = 0.0;

        vertices = nullptr;

        hash[0] = nullptr;
        hash[1] = nullptr;
        hash[2] = nullptr;
        hashSize = 8000;
    }

    std::string className(){ return std::string("GridBins");}

    void init() {
        initialize();
    }

    void copyToGPU(void) {
        //std::cout << "Debug: GridBins::copyToGPU \n";
        for( unsigned i=0; i<3; ++i ) {
            if( hash[i] ) {
                hash[i]->copyToGPU();
            }
        }

        Base::copyToGPU();
    }

    void copy(const GridBins* rhs) {
#ifdef __CUDACC__
        if( numVertices != 0 && (numVertices != rhs->numVertices) ){
            std::cout << "Error: GridBins::copy -- can't change grid size after initialization.\n";
            std::cout << "Error: GridBins::copy -- isCudaIntermediate = " << isCudaIntermediate << " \n";
            std::cout << "Error: GridBins::copy -- rhs->isCudaIntermediate = " << rhs->isCudaIntermediate << " \n";
            throw std::runtime_error("GridBins::copy -- can't change grid size after initialization.");
        }

        if( isCudaIntermediate ) {
            // host to device
            //printf( "GridBins.hh::copy -- allocating vertices on device, numVertices = %d\n", rhs->numVertices);
            vertices = (float_t*) MONTERAYDEVICEALLOC( rhs->numVertices*sizeof(float_t), std::string("device - GridBins::vertices") );
            MonteRayMemcpy( vertices, rhs->vertices, rhs->numVertices*sizeof(float_t), cudaMemcpyHostToDevice );

            for( unsigned i=0; i<3; ++i ) {
                if( rhs->hash[i] ) {
                    hash[i] = rhs->hash[i]->devicePtr;
                }
            }

        } else {
            // device to host
            MonteRayMemcpy( vertices, rhs->vertices, rhs->numVertices*sizeof(float_t), cudaMemcpyDeviceToHost );
        }

        numVertices = rhs->numVertices;

        offset[0] = rhs->offset[0];
        offset[1] = rhs->offset[1];
        offset[2] = rhs->offset[2];

        num[0] = rhs->num[0];
        num[1] = rhs->num[1];
        num[2] = rhs->num[2];

        numXY = rhs->numXY;

        regular[0] = rhs->regular[0];
        regular[1] = rhs->regular[1];
        regular[2] = rhs->regular[2];

        delta[0] = rhs->delta[0];
        delta[1] = rhs->delta[1];
        delta[2] = rhs->delta[2];

        minMax[0] = rhs->minMax[0];
        minMax[1] = rhs->minMax[1];
        minMax[2] = rhs->minMax[2];
        minMax[3] = rhs->minMax[3];
        minMax[4] = rhs->minMax[4];
        minMax[5] = rhs->minMax[5];

        hashSize = rhs->hashSize;
#else
        throw std::runtime_error("GridBins::copy -- can NOT copy between host and device without CUDA.");
#endif
    }

    GridBins&
    operator=( GridBins& rhs ) {
        numVertices = rhs.numVertices;
        //vertices = rhs.vertices;
        offset[0] = rhs.offset[0];
        offset[1] = rhs.offset[1];
        offset[2] = rhs.offset[2];

        num[0] = rhs.num[0];
        num[1] = rhs.num[1];
        num[2] = rhs.num[2];

        numXY = rhs.numXY;

        regular[0] = rhs.regular[0];
        regular[1] = rhs.regular[1];
        regular[2] = rhs.regular[2];

        delta[0] = rhs.delta[0];
        delta[1] = rhs.delta[1];
        delta[2] = rhs.delta[2];

        minMax[0] = rhs.minMax[0];
        minMax[1] = rhs.minMax[1];
        minMax[2] = rhs.minMax[2];
        minMax[3] = rhs.minMax[3];
        minMax[4] = rhs.minMax[4];
        minMax[5] = rhs.minMax[5];

        hashSize = rhs.hashSize;
        return *this;
    }

    template<class READER_T>
    GridBins( READER_T& reader) : GridBins() {
        if( reader.getGeometryString() != "XYZ" )  {
            throw std::runtime_error( "Invalid Geometry type -- MonteRay::GridBins only supports XYZ");
        }
        for( unsigned d=0; d < 3; ++d) {
            std::vector<double> vertices = reader.getVertices(d);
            setVertices(d, vertices );
        }
        finalize();
    }

    CUDA_CALLABLE_MEMBER
    unsigned getMaxNumVertices() const {
        return MAXNUMVERTICES;
    }

    CUDA_CALLABLE_MEMBER
    unsigned getOffset( const unsigned dim ) const {
        MONTERAY_ASSERT( dim < 3);
        return offset[dim];
    }

    void setVertices(unsigned dim, float_t min, float_t max, unsigned numBins);

    template<typename T>
    void setVertices(const unsigned dim, const std::vector<T>& vertices );

    CUDA_CALLABLE_MEMBER
    float_t getVertex(const unsigned dim, const unsigned index ) const {
        return vertices[ offset[dim] + index ];
    }

    CUDA_CALLABLE_MEMBER
    unsigned getNumVertices(const unsigned dim) const {
        MONTERAY_ASSERT( dim < 3);
        return num[dim]+1;
    }

    CUDA_CALLABLE_MEMBER
    unsigned getNumBins(unsigned dim) const {
        MONTERAY_ASSERT( dim < 3);
        return num[dim];
    }

    CUDA_CALLABLE_MEMBER
    bool isRegular( unsigned dim) const {
        MONTERAY_ASSERT( dim < 3);
        return regular[dim];
    }

    CUDA_CALLABLE_MEMBER
    void finalize();

    CUDA_CALLABLE_MEMBER
    unsigned getNumXY() const { return numXY; }

    CUDA_CALLABLE_MEMBER
    float_t min(const unsigned dim) const {
        MONTERAY_ASSERT( dim < 3);
        return minMax[dim*2];
    }

    CUDA_CALLABLE_MEMBER
    float_t max(const unsigned dim) const {
        MONTERAY_ASSERT( dim < 3);
        return minMax[dim*2+1];
    }

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
    void getHashLowerUpperBins(unsigned dim, gpuFloatType_t value, unsigned& lower, unsigned& upper) const {
        hash[dim]->getLowerUpperBins(value, lower, upper );
    }

    CUDA_CALLABLE_MEMBER
    void setDefaultHashSize(unsigned n) { hashSize = n;}

    CUDA_CALLABLE_MEMBER
    unsigned getDefaultHashSize( void ) const { return hashSize; }


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

template<typename T>
void
GridBins::setVertices( const unsigned dim, const std::vector<T>& verts) {
    minMax[dim*2] = verts.front();
    minMax[dim*2+1] = verts.back();

    delta[dim] = -1.0;
    num[dim] = verts.size()-1;

    if( getNumBins(dim) > MAXNUMVERTICES ) {
        ABORT("GridBins::setVertices -- exceeding max number of vertices.");
    }

    unsigned counter = 0;
    for( auto itr = verts.cbegin(); itr != verts.cend(); ++itr) {
        vertices[offset[dim]+counter] = *itr;
        ++counter;
    }

    hash[dim] = new HashBins( vertices + offset[dim], num[dim]+1, hashSize );

    regular[dim] = false;
}

}
#endif /* GRIDBINS_H_ */
