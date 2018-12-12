#include "GridBins.hh"

#include <iostream>
#include <fstream>
#include <ostream>

#ifndef __CUDACC__
#include <cmath>
#endif

#include "GPUErrorCheck.hh"
#include "MonteRayDefinitions.hh"
#include "MonteRay_binaryIO.hh"
#include "BinarySearch.hh"
#include "MonteRayConstants.hh"
#include "MonteRayCopyMemory.t.hh"
#include "GPUErrorCheck.hh"
#include "HashBins.hh"

#ifdef MCATK_INLINED
#include "ReadLnk3dnt.hh"
#endif

namespace MonteRay{

GridBins::GridBins() : CopyMemoryBase<GridBins>() {
    initialize();
    numVertices = MAXNUMVERTICES*3;
    vertices = (float_t*) MONTERAYHOSTALLOC( numVertices * sizeof( float_t ), false, std::string("GridBins::vertices") );

    for( unsigned i=0; i<numVertices; ++i) {
        vertices[i] = 0.0;
    }
}

GridBins::~GridBins() {
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
void
GridBins::initialize() {
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

void
GridBins::copyToGPU(void) {
    //std::cout << "Debug: GridBins::copyToGPU \n";
    for( unsigned i=0; i<3; ++i ) {
        if( hash[i] ) {
            hash[i]->copyToGPU();
        }
    }

    Base::copyToGPU();
}

void
GridBins::copy(const GridBins* rhs) {
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
GridBins::operator=( GridBins& rhs ) {
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

void GridBins::setVertices( unsigned dim, float_t min, float_t max, unsigned numBins ) {

    minMax[dim*2] = min;
    minMax[dim*2+1] = max;

    delta[dim] = (max - min) / numBins;
    num[dim] = numBins;

    if( numBins+1 > MAXNUMVERTICES ) {
        ABORT("GridBins::setVertices -- exceeding max number of vertices.");
    }

    vertices[ offset[dim] ] = min;

    //global::float_t location;
    for( unsigned i = 1; i<numBins+1; ++i) {
        vertices[ i+ offset[dim]] = vertices[ i - 1 + offset[dim] ] + delta[dim];
    }

    hash[dim] = new HashBins( vertices + offset[dim], num[dim]+1, hashSize );

    regular[dim] = true;
}

CUDA_CALLABLE_MEMBER
void GridBins::finalize() {
    for( unsigned dim = 0; dim < 3; ++dim) {
        if( num[dim] == 0 ) {
            ABORT("GridBins::finalize -- vertices not set.");
        }
    }

    // move y data
    unsigned int pad = 1;

    unsigned new_offset = num[0] + pad;
    for( unsigned i = 0; i < num[1]+1; ++i) {
        vertices[i + new_offset] = vertices[i+offset[1]];
        vertices[i+offset[1]] = -1.0;
    }
    offset[1] = num[0] + pad;

    // move z data
    new_offset = num[0] + num[1] + pad + pad;
    for( unsigned i = 0; i < num[2]+1; ++i) {
        vertices[i + new_offset] = vertices[i+offset[2]];
        vertices[i+offset[2]] = -1.0;
    }
    offset[2] = num[0] + num[1] + pad + pad;

    numXY = num[0]*num[1];
}

CUDA_CALLABLE_MEMBER
unsigned
GridBins::getOffset( const unsigned dim ) const {
    MONTERAY_ASSERT( dim < 3);
    return offset[dim];
}

CUDA_CALLABLE_MEMBER
 unsigned
 GridBins::getNumVertices(const unsigned dim) const {
     MONTERAY_ASSERT( dim < 3);
     return num[dim]+1;
 }

 CUDA_CALLABLE_MEMBER
 unsigned
 GridBins::getNumBins(unsigned dim) const {
     MONTERAY_ASSERT( dim < 3);
     return num[dim];
 }

 CUDA_CALLABLE_MEMBER
 bool
 GridBins::isRegular( unsigned dim) const {
     MONTERAY_ASSERT( dim < 3);
     return regular[dim];
 }

 CUDA_CALLABLE_MEMBER
 float_t
 GridBins::min(const unsigned dim) const {
     MONTERAY_ASSERT( dim < 3);
     return minMax[dim*2];
 }

 CUDA_CALLABLE_MEMBER
 float_t
 GridBins::max(const unsigned dim) const {
     MONTERAY_ASSERT( dim < 3);
     return minMax[dim*2+1];
 }

 void
 GridBins::abort(const char* buffer) {
     ABORT( buffer );
 }

CUDA_CALLABLE_MEMBER
unsigned
GridBins::calcIndex(const int* const indices ) const {
    return indices[0] + indices[1]*getNumBins(0) + indices[2]*getNumBins(0)*getNumBins(1);
}

void GridBins::calcIJK(unsigned index, unsigned* indices ) const {
    indices[0] = 0; indices[1] = 0; indices[2] = 0;

    unsigned offsets[3];
    offsets[0] = 1;
    offsets[1] = getNumBins(0);
    offsets[2] = getNumBins(0)*getNumBins(1);

    for( int d = 2; d > -1; --d ) {
        unsigned current_offset = offsets[ d ];
        indices[d] = index / current_offset;
        index -= indices[d] * current_offset;
    }
}

CUDA_CALLABLE_MEMBER
int
GridBins::getDimIndex(const unsigned dim, const gpuRayFloat_t pos ) const {
    // returns -1 for one neg side of mesh
    // and number of bins on the pos side of the mesh
    // need to call isIndexOutside(dim, grid, index) to check if the
    // index is in the mesh
    int dim_index;
    gpuFloatType_t minimum = min(dim);
    unsigned numBins = getNumBins(dim);

    if( pos <= minimum ) {
        dim_index = -1;
    } else if( pos >= max(dim)  ) {
        dim_index = numBins;
    } else {
        if( regular[dim] ) {
            dim_index = ( pos -  minimum ) / delta[dim];
        } else {
            unsigned lower;
            unsigned upper;
            hash[dim]->getLowerUpperBins(pos, lower, upper);
            if( lower == upper ) { return lower; }
            dim_index = lower + LowerBoundIndex( vertices + offset[dim] + lower, upper-lower+1, pos  );
        }
    }
    return dim_index;
}

CUDA_CALLABLE_MEMBER
bool GridBins::isIndexOutside( unsigned dim, int i) const {
    return ( (i < 0 ||  i >= getNumBins(dim)) ? true : false);
}

CUDA_CALLABLE_MEMBER
bool GridBins::isOutside(const int* indices ) const {
    if( isIndexOutside(0, indices[0]) ) return true;
    if( isIndexOutside(1, indices[1]) ) return true;
    if( isIndexOutside(2, indices[2]) ) return true;
    return false;
}

CUDA_CALLABLE_MEMBER
unsigned
GridBins::getIndex(const Position_t& particle_pos) {

    int indices[3]= {0, 0, 0};
    for( unsigned d = 0; d < 3; ++d ) {
        indices[d] = getDimIndex(d, particle_pos[d] );

        // outside the grid
        if( isIndexOutside( d, indices[d] ) ) { return UINT_MAX; }
    }

    return calcIndex( indices );
}

Position_t GridBins::getCenterPointByIndices( const unsigned* const indices ) const{
    Position_t pos;
    for( unsigned i=0; i<3; ++i) {
        const unsigned vertexIndex = indices[i];
        pos[i] = (getVertex(i, vertexIndex) + getVertex(i, vertexIndex+1)) / 2.0f ;
    }
    return pos;
}

Position_t GridBins::getCenterPointByIndex(unsigned index ) const{
    unsigned indices[3];
    calcIJK(index, indices);

    return getCenterPointByIndices( indices );
}

float_t getDistance( Position_t& pos1, Position_t& pos2) {
    float_t deltaSq[3];
    deltaSq[0] = (pos1[0] - pos2[0])*(pos1[0] - pos2[0]);
    deltaSq[1] = (pos1[1] - pos2[1])*(pos1[1] - pos2[1]);
    deltaSq[2] = (pos1[2] - pos2[2])*(pos1[2] - pos2[2]);
    return sqrt( deltaSq[0] + deltaSq[1] + deltaSq[2]);
}

CUDA_CALLABLE_MEMBER
unsigned GridBins::rayTrace( int* global_indices, gpuRayFloat_t* distances, const Position_t& pos, const Position_t& dir, float_t distance,  bool outsideDistances) const {
    const bool debug = false;

    int current_indices[3] = {0, 0, 0}; // current position indices in the grid, must be int because can be outside

    if( debug ){
        printf( "GridBins::rayTrace --------------------------------\n");
    }

    int cells[3][MAXNUMVERTICES];
    gpuRayFloat_t crossingDistances[3][MAXNUMVERTICES];
    unsigned numCrossings[3];

    for( unsigned i=0; i<3; ++i){
        current_indices[i] = getDimIndex(i, pos[i] );

        numCrossings[i] = calcCrossings( vertices + offset[i], num[i]+1, cells[i], crossingDistances[i], pos[i], dir[i], distance, current_indices[i]);

        if( debug ){
            printf( "GridBins::rayTrace -- current_indices[i]=%d\n", current_indices[i] );
            printf( "GridBins::rayTrace -- numCrossings[i]=%d\n", numCrossings[i] );
        }

        // if outside and ray doesn't move inside then ray never enters the grid
        if( isIndexOutside(i,current_indices[i]) && numCrossings[i] == 0  ) {
            return 0U;
        }
    }

    if( debug ){
        printf( "GridBins::rayTrace -- numCrossings[0]=%d\n", numCrossings[0] );
        printf( "GridBins::rayTrace -- numCrossings[1]=%d\n", numCrossings[1] );
        printf( "GridBins::rayTrace -- numCrossings[2]=%d\n", numCrossings[2] );
    }

    return orderCrossings(global_indices, distances, MAXNUMVERTICES, cells[0], crossingDistances[0], numCrossings, current_indices, distance, outsideDistances);
}


CUDA_CALLABLE_MEMBER
unsigned calcCrossings(const float_t* const vertices, unsigned nVertices, int* cells, gpuRayFloat_t* distances, float_t pos, float_t dir, float_t distance, int index ){
    const bool debug = false;

    unsigned nDistances = 0;

    if( debug ) {
        printf( "GridBins::calcCrossings --------------------------------\n" );
        printf( "calcCrossings -- vertices[0]=%f\n", vertices[0] );
        printf( "calcCrossings -- vertices[nVertices-1]=%f\n", vertices[nVertices-1] );
        printf( "calcCrossings -- pos=%f\n", pos );
        printf( "calcCrossings -- dir=%f\n", dir );
    }

#ifdef __CUDACC__
    if( abs(dir) <= MonteRay::epsilon ) {
#else
    if( std::abs(dir) <= MonteRay::epsilon ) {
#endif
        return nDistances;
    }

    int start_index = index;
    int cell_index = start_index;

    if( start_index < 0 ) {
        if( dir < 0.0 ) {
            return nDistances;
        }
    }

    int nBins = nVertices - 1;
    if( start_index >= nBins ) {
        if( dir > 0.0 ) {
            return nDistances;
        }
    }

    unsigned offset = 0;
    if( dir > 0.0f ) {
        offset = 1;
    }
    //    unsigned offset = (unsigned) signbit(-dir);
    int end_index = offset*(nBins-1);;

    int dirIncrement = copysign( 1.0f, dir );

    unsigned num_indices = abs(end_index - start_index ) + 1;

    int current_index = start_index;

    // Calculate boundary crossing distances
    float_t invDir = 1/dir;
    bool rayTerminated = false;
    for( int i = 0; i < num_indices ; ++i ) {

        //        BOOST_ASSERT( (current_index + offset) >= 0 );
        //        BOOST_ASSERT( (current_index + offset) < nBins+1 );

        float_t minDistance = ( vertices[current_index + offset] - pos) * invDir;

        if( debug ) {
            printf( " calcCrossings -- current_index=%d\n", current_index );
            printf( " calcCrossings --        offset=%d\n", offset );
            printf( " calcCrossings -- vertices[current_index + offset]=%f\n", vertices[current_index + offset] );
        }

        //if( rayDistance == inf ) {
        //    // ray doesn't cross plane
        //    break;
        //}

        if( minDistance >= distance ) {
            cells[nDistances] = cell_index;
            distances[nDistances] = distance;
            ++nDistances;
            rayTerminated = true;
            break;
        }

        cells[nDistances] = cell_index;
        distances[nDistances] = minDistance;
        ++nDistances;

        current_index += dirIncrement;
        cell_index = current_index;
    }

    if( !rayTerminated ) {
        // finish with distance into area outside
        cells[nDistances] = cell_index;
        distances[nDistances] = distance;
        ++nDistances;
        rayTerminated = true;
    }

    if( debug ) {
        for( unsigned i=0; i<nDistances; ++i){
            printf( " calcCrossings -- i=%d  cell index=%d  distance=%f\n", i, cells[i], distances[i] );
        }
        printf( "-----------------------------------------------------------------------\n" );
    }

    return nDistances;
}

CUDA_CALLABLE_MEMBER
unsigned GridBins::orderCrossings(int* global_indices, gpuRayFloat_t* distances, unsigned num, const int* const cells, const gpuRayFloat_t* const crossingDistances, unsigned* numCrossings, int* indices, float_t distance, bool outsideDistances ) const {
    // Order the distance crossings to provide a rayTrace

    const bool debug = false;

    unsigned end[3] = {0, 0, 0}; //    last location in the distance[i] vector

    unsigned maxNumCrossings = 0;
    for( unsigned i=0; i<3; ++i){
        end[i] = numCrossings[i];
        maxNumCrossings += end[i];
    }

    if( debug ) {
        for( unsigned i=0; i<3; ++i){
            printf( "Debug: i=%d  numCrossings=%d\n", i, numCrossings[i]);
            for( unsigned j=0; j< numCrossings[i]; ++j ) {
                printf( "Debug: j=%d  index=%d  distance=%f", j, *((cells+i*num) + j),  *((crossingDistances+i*num)+j) );
            }
        }
    }

    float_t minDistances[3];

    bool outside;
    float_t priorDistance = 0.0;
    unsigned start[3] = {0, 0, 0}; // current location in the distance[i] vector

    unsigned numRayCrossings = 0;
    for( unsigned i=0; i<maxNumCrossings; ++i){

        unsigned minDim;
        float_t minimumDistance = MonteRay::inf;
        for( unsigned j = 0; j<3; ++j) {
            if( start[j] < end[j] ) {
                minDistances[j] = *((crossingDistances+j*num)+start[j]);
                if( minDistances[j] < minimumDistance ) {
                    minimumDistance = minDistances[j];
                    minDim = j;
                }
            } else {
                minDistances[j] = MonteRay::inf;
            }
        }

        indices[minDim] =  *((cells+minDim*num) + start[minDim]);
        if( debug ) {
            printf( "Debug: minDim=%d  index=%d   minimumDistance=%f\n", minDim, indices[minDim], minimumDistance);
        }

        // test for outside of the grid
        outside = isOutside( indices );

        if( debug ) {
            if( outside ) {
                printf( "Debug: ray is outside \n" );
            } else {
                printf( "Debug: ray is inside \n" );
            }
        }

        float_t currentDistance = minimumDistance;

        if( !outside || outsideDistances ) {
            float_t deltaDistance = currentDistance - priorDistance;

            if( deltaDistance > 0.0  ) {
                unsigned global_index;
                if( !outside ) {
                    global_index = calcIndex(indices );
                } else {
                    global_index = UINT_MAX;
                }
                global_indices[numRayCrossings] = global_index;
                distances[numRayCrossings] = deltaDistance;
                ++numRayCrossings;

                if( debug ) {
                    printf( "Debug: ******************\n" );
                    printf( "Debug:  Entry Num    = %d\n", numRayCrossings );
                    printf( "Debug:     index[0]  = %d\n", indices[0] );
                    printf( "Debug:     index[1]  = %d\n", indices[1] );
                    printf( "Debug:     index[2]  = %d\n", indices[2] );
                    printf( "Debug:     distance  = %f\n", deltaDistance );
                }
            }
        }

        if( currentDistance >= distance ) {
            break;
        }

        indices[minDim] = *((cells+minDim*num) + start[minDim]+1);

        if( ! outside ) {
            if( isIndexOutside(minDim, indices[minDim] ) ) {
                // ray has moved outside of grid
                break;
            }
        }

        ++start[minDim];
        priorDistance = currentDistance;
    }

    return numRayCrossings;
}

CUDA_CALLABLE_KERNEL
void
kernelRayTrace(void* ptrNumCrossings,
        GridBins* ptrGrid,
        int* ptrCells,
        gpuRayFloat_t* ptrDistances,
        gpuFloatType_t x, gpuFloatType_t y, gpuFloatType_t z,
        gpuFloatType_t u, gpuFloatType_t v, gpuFloatType_t w,
        gpuFloatType_t distance,
        bool outsideDistances) {

    const bool debug = false;

    if( debug ) {
        printf("kernelRayTrace(GridBins*):: Starting kernelRayTrace ******************\n");
    }

    unsigned* numCrossings = (unsigned*) ptrNumCrossings;

    Position_t pos( x, y, z );
    Direction_t dir( u, v, w );

    numCrossings[0] = ptrGrid->rayTrace( ptrCells, ptrDistances, pos, dir, distance, outsideDistances);

    if( debug ) {
        printf("kernelRayTrace(GridBins*):: numCrossings=%d\n",numCrossings[0]);
    }
}

CUDA_CALLABLE_MEMBER
void
GridBins::getHashLowerUpperBins(unsigned dim, gpuFloatType_t value, unsigned& lower, unsigned& upper) const {
    hash[dim]->getLowerUpperBins(value, lower, upper );
}

template<typename T>
void
GridBins::setVertices( const unsigned dim, const std::vector<T>& verts) {
    minMax[dim*2] = verts.front();
    minMax[dim*2+1] = verts.back();

    delta[dim] = -1.0;
    num[dim] = verts.size()-1;

    if( getNumBins(dim) > MAXNUMVERTICES ) {
        abort("GridBins::setVertices -- exceeding max number of vertices.");
    }

    unsigned counter = 0;
    for( auto itr = verts.cbegin(); itr != verts.cend(); ++itr) {
        vertices[offset[dim]+counter] = *itr;
        ++counter;
    }

    hash[dim] = new HashBins( vertices + offset[dim], num[dim]+1, hashSize );

    regular[dim] = false;
}

void
GridBins::writeToFile( const std::string& filename ) const {
    std::ofstream outfile;

    outfile.open( filename.c_str(), std::ios::binary | std::ios::out);
    if( ! outfile.is_open() ) {
        fprintf(stderr, "GridBins::writeToFile -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
        throw std::runtime_error("GridBins::writeToFile  -- Failure to open file" );
    }
    assert( outfile.good() );
    outfile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
    write( outfile );
    outfile.close();
}

void
GridBins::readFromFile( const std::string& filename ){
    std::ifstream infile;
    if( infile.is_open() ) {
        infile.close();
    }
    infile.open( filename.c_str(), std::ios::binary | std::ios::in);

    if( ! infile.is_open() ) {
        fprintf(stderr, "Error:  GridBins::readFromFile -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
        throw std::runtime_error("GridBins::readFromFile -- Failure to open file" );
    }
    assert( infile.good() );
    infile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
    read(infile);
    infile.close();
}

void
GridBins::write(std::ostream& outf) const {
    unsigned version = 0;
    binaryIO::write(outf, version );

    binaryIO::write(outf, numVertices);
    binaryIO::write(outf, numXY);
    binaryIO::write(outf, num);
    binaryIO::write(outf, offset);
    binaryIO::write(outf, regular);
    binaryIO::write(outf, delta);
    binaryIO::write(outf, minMax);

    for( unsigned i = 0; i<numVertices; ++i) {
        binaryIO::write(outf, vertices[i]);
    }
    for( unsigned i = 0; i<3; ++i) {
        hash[i]->write(outf);
    }
    binaryIO::write(outf, hashSize);
}

void
GridBins::read(std::istream& infile) {
    unsigned version;
    binaryIO::read(infile, version );

    binaryIO::read(infile, numVertices);
    binaryIO::read(infile, numXY);
    binaryIO::read(infile, num);
    binaryIO::read(infile, offset);
    binaryIO::read(infile, regular);
    binaryIO::read(infile, delta);
    binaryIO::read(infile, minMax);

    if( vertices ) {
        MonteRayHostFree( vertices, Base::isManagedMemory );
    }
    vertices = (float_t*) MONTERAYHOSTALLOC( numVertices * sizeof( float_t ), false, std::string("GridBins::vertices") );
    for( unsigned i = 0; i<numVertices; ++i) {
        binaryIO::read(infile, vertices[i]);
    }

    for( unsigned dim = 0; dim<3; ++dim) {
        if( hash[dim] ) {
            delete hash[dim];
        }
        hash[dim] = new HashBins();
        hash[dim]->read(infile);
    }
    binaryIO::read(infile, hashSize);
}

template void GridBins::setVertices<double>( const unsigned dim, const std::vector<double>& verts);
template void GridBins::setVertices<float>( const unsigned dim, const std::vector<float>& verts);

}

template class MonteRay::CopyMemoryBase<MonteRay::GridBins>;

