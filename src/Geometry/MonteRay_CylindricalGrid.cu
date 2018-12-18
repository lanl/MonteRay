#include "MonteRay_CylindricalGrid.t.hh"
#include "MonteRayDefinitions.hh"
#include "MonteRayConstants.hh"
#include "MonteRay_SingleValueCopyMemory.t.hh"
#include "MonteRayCopyMemory.t.hh"

#include <float.h>

namespace MonteRay {

using ptrCylindricalGrid_result_t = MonteRay_SingleValueCopyMemory<MonteRay_CylindricalGrid*>;

CUDA_CALLABLE_KERNEL
void createDeviceInstance(MonteRay_CylindricalGrid** pPtrInstance, ptrCylindricalGrid_result_t* pResult, MonteRay_GridBins* pGridR, MonteRay_GridBins* pGridZ ) {
    *pPtrInstance = new MonteRay_CylindricalGrid( 2, pGridR, pGridZ );
    pResult->v = *pPtrInstance;
    //if( debug ) printf( "Debug: createDeviceInstance -- pPtrInstance = %d\n", pPtrInstance );
}

CUDA_CALLABLE_KERNEL
void deleteDeviceInstance(MonteRay_CylindricalGrid** pPtrInstance) {
    delete *pPtrInstance;
}

CUDAHOST_CALLABLE_MEMBER
MonteRay_CylindricalGrid*
MonteRay_CylindricalGrid::getDeviceInstancePtr() {
    return devicePtr;
}

CUDA_CALLABLE_MEMBER
MonteRay_CylindricalGrid::MonteRay_CylindricalGrid(unsigned dim, pArrayOfpGridInfo_t pBins) :
MonteRay_GridSystemInterface(dim)
{
    MONTERAY_VERIFY( dim == DimMax, "MonteRay_CylindricalGrid::ctor -- only 2-D is allowed" ); // No greater than 2-D.

    DIM = dim;

    pRVertices = pBins[R];
    if( DIM > 1 ) { pZVertices = pBins[Z]; }
    //if( DIM > 2 ) { pThetaVertices = pBins[Theta]; }
    validate();
}

CUDA_CALLABLE_MEMBER
MonteRay_CylindricalGrid::MonteRay_CylindricalGrid(unsigned dim, GridBins_t* pGridR, GridBins_t* pGridZ ) :
MonteRay_GridSystemInterface(dim)
{
    MONTERAY_VERIFY( dim == DimMax, "MonteRay_CylindricalGrid::ctor -- only 2-D is allowed" ); // No greater than 2-D.

    DIM = 2;
    pRVertices = pGridR;
    pZVertices = pGridZ;
    validate();
}

CUDA_CALLABLE_MEMBER
MonteRay_CylindricalGrid::~MonteRay_CylindricalGrid(void){
#ifdef __CUDACC__
#ifndef __CUDA_ARCH__
    if( ptrDevicePtr ) {
        deleteDeviceInstance<<<1,1>>>( ptrDevicePtr );
        cudaDeviceSynchronize();
    }
    MonteRayDeviceFree( ptrDevicePtr );
#endif
#endif
}


CUDAHOST_CALLABLE_MEMBER
void
MonteRay_CylindricalGrid::copyToGPU(void) {
    if( debug ) std::cout << "Debug: MonteRay_CylindricalGrid::copyToGPU \n";
#ifdef __CUDACC__
    ptrDevicePtr = (MonteRay_CylindricalGrid**) MONTERAYDEVICEALLOC(sizeof(MonteRay_CylindricalGrid*), std::string("device - MonteRay_CylindricalGrid::ptrDevicePtr") );

    pRVertices->copyToGPU();
    pZVertices->copyToGPU();
    //pThetaVertices->copyToGPU();

    std::unique_ptr<ptrCylindricalGrid_result_t> ptrResult = std::unique_ptr<ptrCylindricalGrid_result_t>( new ptrCylindricalGrid_result_t() );
    ptrResult->copyToGPU();

    createDeviceInstance<<<1,1>>>( ptrDevicePtr, ptrResult->devicePtr, pRVertices->devicePtr, pZVertices->devicePtr );
    cudaDeviceSynchronize();
    ptrResult->copyToCPU();
    devicePtr = ptrResult->v;

#endif
}


CUDA_CALLABLE_MEMBER
void
MonteRay_CylindricalGrid::validate(void) {
    validateR();

    numRBins = pRVertices->getNumBins();
    if( DIM > 1 ) numZBins = pZVertices->getNumBins();
    //if( DIM > 2 ) { pThetaVertices = &(bins[Theta]); }
}

CUDA_CALLABLE_MEMBER
void
MonteRay_CylindricalGrid::validateR(void) {
    // Test for negative R
    for( int i=0; i<pRVertices->nVertices; ++i ){
        MONTERAY_VERIFY( pRVertices->vertices[i] >= 0.0, "MonteRay_CylindricalGrid::validateR -- Can't have negative values for radius!!!" );
    }

    pRVertices->modifyForRadial();
}

CUDA_CALLABLE_MEMBER
MonteRay_CylindricalGrid::Position_t
MonteRay_CylindricalGrid::convertFromCartesian( const Position_t& pos) const {
    Position_t particleMeshPosition = {0.0, 0.0, 0.0};

    gpuRayFloat_t r = sqrt(pos[x]*pos[x] + pos[y]*pos[y]);
    particleMeshPosition[R] = r;

    if(DIM > 1) {
        particleMeshPosition[Z] =  pos[z];
    }

    //     if(DIM > 2) {
    //         const double smallR = 1.0e-8;
    //         double theta = 0.0;
    //         if( r >= smallR ) {
    //             theta = std::acos( pos[x] / r );
    //         }
    //
    //         if( pos[y] < 0.0  ) {
    //             theta = 2.0*mcatk::Constants::pi - theta;
    //         }
    //         particleMeshPosition[Theta] =  theta / (2.0*mcatk::Constants::pi); // to get revolutions
    //     }

    return particleMeshPosition;
}


CUDA_CALLABLE_MEMBER
unsigned
MonteRay_CylindricalGrid::getIndex( const Position_t& particle_pos) const{
    if( debug ) { printf("Debug: MonteRay_CylindricalGrid::getIndex -- starting\n"); }

    int indices[3]= {0, 0, 0};
    Position_t pos = convertFromCartesian( particle_pos );

    if( debug ) { printf("%i\n", pRVertices->isRadial() ); }
    indices[0] = pRVertices->getRadialIndexFromR( pos[R] );

    if( DIM>1 ) { indices[1] = getAxialIndex( pos[Z] ); }

    for( auto d = 0; d < DIM; ++d ) {
        // outside the grid
        if( isIndexOutside(d, indices[d] ) ) { return UINT_MAX; }
    }

    return calcIndex( indices );;
}

CUDA_CALLABLE_MEMBER
bool
MonteRay_CylindricalGrid::isIndexOutside( unsigned d, int i) const {
    MONTERAY_ASSERT( d < 3 );

    if( d == R ) {
        MONTERAY_ASSERT( i >= 0 );
        if( i >= numRBins ) { return true; }
    }

    if( d == Z ) {
        if( i < 0 ||  i >= numZBins ){ return true; }
    }

    //    if( d == Theta ) {
    //        if( i < 0 ||  i >= numThetaBins ){ return true; }
    //    }
    return false;
}

CUDA_CALLABLE_MEMBER
unsigned
MonteRay_CylindricalGrid::calcIndex( const int indices[] ) const{
    unsigned index = indices[0];
    if( DIM > 1) {
        index += indices[1]*numRBins;
    }
    //    if( DIM > 2) {
    //        index += indices[2]*numZBins*numRBins;
    //    }

    return index;
}

CUDA_CALLABLE_MEMBER
uint3
MonteRay_CylindricalGrid::calcIJK( unsigned index ) const {
    uint3 indices;

    uint3 offsets;
    offsets.x = 1;

    offsets.y = numRBins;
    offsets.z = numRBins*numZBins;

    MONTERAY_ASSERT(offsets.z > 0 );
    MONTERAY_ASSERT(offsets.y > 0 );
    MONTERAY_ASSERT(offsets.x > 0 );

    indices.z = index / offsets.z;
    index -= indices.z * offsets.z;

    indices.y = index / offsets.y;
    index -= indices.y * offsets.y;

    indices.x = index / offsets.x;

    return indices;
}

CUDA_CALLABLE_MEMBER
bool
MonteRay_CylindricalGrid::isOutside( const int i[] ) const {
    for( unsigned d=0; d<DIM; ++d){
        if( isIndexOutside(d, i[d]) ) return true;
    }
    return false;
}

CUDA_CALLABLE_MEMBER
gpuRayFloat_t
MonteRay_CylindricalGrid::getVolume( unsigned index ) const {
    uint3 indices = calcIJK( index );

    gpuRayFloat_t innerRadiusSq = 0.0;
    if( indices.x > 0 ){
        innerRadiusSq = pRVertices->verticesSq[indices.x-1];
    }
    gpuRayFloat_t outerRadiusSq = pRVertices->verticesSq[indices.x];

    gpuRayFloat_t volume = MonteRay::pi * ( outerRadiusSq - innerRadiusSq );
    if( DIM > 1 ) volume *= pZVertices->vertices[ indices.y + 1 ] - pZVertices->vertices[ indices.y ];

    //DIM>2 here ?

    return volume;
}

CUDA_CALLABLE_MEMBER
void
MonteRay_CylindricalGrid::rayTrace( rayTraceList_t& rayTraceList, const GridBins_t::Position_t& pos, const GridBins_t::Position_t& dir, gpuRayFloat_t distance,  bool outsideDistances/*=false*/) const {
    if( debug ) printf( "Debug: MonteRay_CylindricalGrid::rayTrace -- \n");
    rayTraceList.reset();
    int indices[3] = {0, 0, 0}; // current position indices in the grid, must be int because can be outside

    multiDimRayTraceMap_t distances;

    // Crossing distance in R direction
    {
        distances[R].reset();
        gpuRayFloat_t particleRSq = calcParticleRSq( pos );
        indices[R] = pRVertices->getRadialIndexFromRSq(particleRSq);

        if( debug ) printf( "Debug: MonteRay_CylindricalGrid::rayTrace -- R Direction -  dimension=%d, index=%d\n", R, indices[R]);

        radialCrossingDistances( distances[R], pos, dir, indices[R], distance );

        if( debug ) printf( "Debug: MonteRay_CylindricalGrid::rayTrace -- R Direction -  dimension=%d, number of radial crossings = %d\n", R, distances[R].size() );

        // if outside and ray doesn't move inside then ray never enters the grid
        if( isIndexOutside(R,indices[R]) && distances[R].size() == 0   ) {
            return;
        }
    }

    // Crossing distance in Z direction
    if( DIM  > 1 ) {
        indices[Z] = pZVertices->getLinearIndex( pos[z] );

        if( debug ) printf( "Debug: MonteRay_CylindricalGrid::rayTrace -- Z Direction -  dimension=%d, index=%d\n", Z, indices[Z]);

        planarCrossingDistance(  distances[Z], *pZVertices, pos[z], dir[z], distance, indices[Z] );

        if( debug ) printf( "Debug: MonteRay_CylindricalGrid::rayTrace -- Z Direction -  dimension=%d, number of planar crossings = %d\n", Z, distances[Z].size() );

        // if outside and ray doesn't move inside then ray never enters the grid
        if( isIndexOutside(Z,indices[Z]) && distances[Z].size() == 0  ) {
            return;
        }
    }

    orderCrossings( rayTraceList, distances, indices, distance, outsideDistances );

    if( debug ) printf( "Debug: MonteRay_CylindricalGrid::rayTrace -- number of total crossings = %d\n", rayTraceList.size() );
    return;
}

CUDA_CALLABLE_MEMBER
void
MonteRay_CylindricalGrid::crossingDistance(singleDimRayTraceMap_t& rayTraceMap, unsigned d, const GridBins_t::Position_t& pos, const GridBins_t::Direction_t& dir, gpuRayFloat_t distance ) const {
    if( d == R ) {
        int index = pRVertices->getRadialIndexFromRSq(calcParticleRSq(pos));
        radialCrossingDistances( rayTraceMap, pos, dir, index, distance );
    }

    if( d == Z ) {
        int index = pZVertices->getLinearIndex( pos[z] );
        planarCrossingDistance( rayTraceMap, *pZVertices, pos[z], dir[z], distance, index );
    }
    return;
}

CUDA_CALLABLE_MEMBER
void
MonteRay_CylindricalGrid::radialCrossingDistances(singleDimRayTraceMap_t& rayTraceMap, const Position_t& pos, const Direction_t& dir, unsigned rIndex, gpuRayFloat_t distance ) const {
    //------ Distance to Sphere's Radial-boundary
    if( debug ) {
        printf("Debug: MonteRay_CylindricalGrid::radialCrossingDistances -- \n");
    }

    gpuRayFloat_t particleRSq = calcParticleRSq( pos );

    gpuRayFloat_t           A = calcQuadraticA( dir );
    gpuRayFloat_t           B = calcQuadraticB( pos, dir);

    // trace inward
    bool rayTerminated = radialCrossingDistanceSingleDirection<false>(rayTraceMap, *pRVertices, particleRSq, A, B, distance, rIndex);

    if( debug ) {
        printf("Debug: Inward ray trace size=%d\n",rayTraceMap.size());
        if( rayTerminated ) {
            printf("Debug: - ray terminated!\n");
        } else {
            printf("Debug: - ray not terminated!\n");
        }
    }

    // trace outward
    if( ! rayTerminated ) {
        if( !isIndexOutside(R, rIndex) ) {
            radialCrossingDistanceSingleDirection<true>(rayTraceMap, *pRVertices, particleRSq, A, B, distance, rIndex);
        } else {
            rayTraceMap.add(rIndex, distance);
        }
    }
}

CUDA_CALLABLE_MEMBER
void
MonteRay_CylindricalGrid::radialCrossingDistances( singleDimRayTraceMap_t& rayTraceMap, const Position_t& pos, const Direction_t& dir, gpuRayFloat_t distance ) const {
    gpuRayFloat_t particleRSq = calcParticleRSq( pos );
    unsigned rIndex = pRVertices->getRadialIndexFromRSq(particleRSq);
    radialCrossingDistances( rayTraceMap, pos, dir, rIndex, distance );
}

CUDA_CALLABLE_MEMBER
unsigned
MonteRay_CylindricalGrid::getNumBins( unsigned d) const {
    MONTERAY_ASSERT( d < 3 );
    if( debug ) printf("Debug: MonteRay_CylindricalGrid::getNumBins -- d= %d\n", d);
    if( debug ) printf("Debug: MonteRay_CylindricalGrid::getNumBins --calling pGridBins[d]->getNumBins()\n");
    if( d == 0 ) {
        return pRVertices->getNumBins();
    }
    if( d == 1 ) {
        return pZVertices->getNumBins();
    }
    if( d == 2 ) {
        return 1;
    }
    return 0;
}

template
CUDA_CALLABLE_MEMBER
void
MonteRay_CylindricalGrid::radialCrossingDistancesSingleDirection<true>( singleDimRayTraceMap_t& rayTraceMap, const Position_t& pos, const Direction_t& dir, gpuRayFloat_t distance) const;

template
CUDA_CALLABLE_MEMBER
void
MonteRay_CylindricalGrid::radialCrossingDistancesSingleDirection<false>( singleDimRayTraceMap_t& rayTraceMap, const Position_t& pos, const Direction_t& dir, gpuRayFloat_t distance) const;


} /* namespace MonteRay */
