#include "MonteRayDefinitions.hh"
#include "MonteRay_SphericalGrid.t.hh"
#include "MonteRayConstants.hh"
#include "GPUErrorCheck.hh"
#include "MonteRayParallelAssistant.hh"

#include <float.h>

namespace MonteRay {

/* using ptrSphericalGrid_result_t = Result<MonteRay_SphericalGrid*>; */ // fix this

CUDA_CALLABLE_KERNEL  createDeviceInstance(MonteRay_SphericalGrid** pPtrInstance, ptrSphericalGrid_result_t* pResult, MonteRay_GridBins* pGridR ) {
    *pPtrInstance = new MonteRay_SphericalGrid( 1, pGridR );
    pResult->v = *pPtrInstance;
    //if( debug ) printf( "Debug: createDeviceInstance -- pPtrInstance = %d\n", pPtrInstance );
}

CUDA_CALLABLE_KERNEL  deleteDeviceInstance(MonteRay_SphericalGrid** pPtrInstance) {
    delete *pPtrInstance;
}

CUDAHOST_CALLABLE_MEMBER
MonteRay_SphericalGrid*
MonteRay_SphericalGrid::getDeviceInstancePtr() {
    return devicePtr;
}

CUDA_CALLABLE_MEMBER
MonteRay_SphericalGrid::MonteRay_SphericalGrid(unsigned dim, pArrayOfpGridInfo_t pBins) :
MonteRay_GridSystemInterface(dim)
{
    MONTERAY_VERIFY( dim == DimMax, "MonteRay_SphericalGrid::ctor -- only 1-D is allowed" ); // No greater than 1-D.

    DIM = 1;
    pRVertices = pBins[0];
    validate();
}

CUDA_CALLABLE_MEMBER
MonteRay_SphericalGrid::MonteRay_SphericalGrid(unsigned dim, GridBins_t* pGridR ) :
MonteRay_GridSystemInterface(dim)
{
    MONTERAY_VERIFY( dim == DimMax, "MonteRay_SphericalGrid::ctor -- only 1-D is allowed" ); // No greater than 1-D.

    DIM = 1;
    pRVertices = pGridR;
    validate();
}

CUDA_CALLABLE_MEMBER
MonteRay_SphericalGrid::~MonteRay_SphericalGrid(void){
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
MonteRay_SphericalGrid::copyToGPU(void) {
    if( debug ) std::cout << "Debug: MonteRay_SphericalGrid::copyToGPU \n";
#ifdef __CUDACC__
    if( ! MonteRay::isWorkGroupMaster() ) return;
    ptrDevicePtr = (MonteRay_SphericalGrid**) MONTERAYDEVICEALLOC(sizeof(MonteRay_SphericalGrid*), std::string("device - MonteRay_SphericalGrid::ptrDevicePtr") );

    pRVertices->copyToGPU();

    std::unique_ptr<ptrSphericalGrid_result_t> ptrResult = std::unique_ptr<ptrSphericalGrid_result_t>( new ptrSphericalGrid_result_t() );
    ptrResult->copyToGPU();

    createDeviceInstance<<<1,1>>>( ptrDevicePtr, ptrResult->devicePtr, pRVertices->devicePtr );
    cudaDeviceSynchronize();
    ptrResult->copyToCPU();
    devicePtr = ptrResult->v;

#endif
}


CUDA_CALLABLE_MEMBER
void
MonteRay_SphericalGrid::validate(void) {
    validateR();
    numRBins = pRVertices->getNumBins();
}

CUDA_CALLABLE_MEMBER
void
MonteRay_SphericalGrid::validateR(void) {
    // Test for negative R
    for( int i=0; i<pRVertices->nVertices; ++i ){
        MONTERAY_VERIFY( pRVertices->vertices[i] >= 0.0, "MonteRay_SphericalGrid::validateR -- Can't have negative values for radius!!!" );
    }

    pRVertices->modifyForRadial();
}

CUDA_CALLABLE_MEMBER
MonteRay_SphericalGrid::Position_t
MonteRay_SphericalGrid::convertFromCartesian( const Position_t& pos) const {
    Position_t particleMeshPosition = {0.0, 0.0, 0.0};

    gpuRayFloat_t r = sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]);
    particleMeshPosition[R] = r;

    return particleMeshPosition;
}


CUDA_CALLABLE_MEMBER
unsigned
MonteRay_SphericalGrid::getIndex( const Position_t& particle_pos) const{
    if( debug ) printf("Debug: MonteRay_SphericalGrid::getIndex -- starting\n");

    int index = 0;
    Position_t pos = convertFromCartesian( particle_pos );

    if( debug )  printf("%i\n", pRVertices->isRadial() );
    index = pRVertices->getRadialIndexFromR( pos[R] );
    if( isIndexOutside(R, index ) ) { return UINT_MAX; }

    return index;
}

CUDA_CALLABLE_MEMBER
bool
MonteRay_SphericalGrid::isIndexOutside( unsigned d,  int i) const {
    MONTERAY_VERIFY( d == 0, "MonteRay_SphericalGrid::isIndexOutside -- Index i must not be negative." );
    MONTERAY_VERIFY( d == 0, "MonteRay_SphericalGrid::isIndexOutside -- Dimension d must be 0 because spherical geometry is 1-D." );
    return pRVertices->isIndexOutside(i);
}

CUDA_CALLABLE_MEMBER
bool
MonteRay_SphericalGrid::isOutside( const int i[] ) const {
    if( isIndexOutside(R, i[R]) ) { return true; }
    return false;
}

CUDA_CALLABLE_MEMBER
unsigned
MonteRay_SphericalGrid::calcIndex( const int indices[] ) const{
    return indices[R];
}


CUDA_CALLABLE_MEMBER
gpuRayFloat_t
MonteRay_SphericalGrid::getVolume( unsigned index ) const {
    gpuRayFloat_t innerRadius = 0.0;
    if( index > 0 ){
        innerRadius = pRVertices->vertices[index-1];
    }
    gpuRayFloat_t outerRadius = pRVertices->vertices[index];

    gpuRayFloat_t volume = 4.0 * MonteRay::pi * ( std::pow(outerRadius,3) - std::pow(innerRadius,3) ) / 3.0 ;

    return volume;
}

CUDA_CALLABLE_MEMBER
void
MonteRay_SphericalGrid::rayTrace(
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        const GridBins_t::Position_t& pos,
        const GridBins_t::Position_t& dir,
        const gpuRayFloat_t distance,
        const bool outsideDistances/*=false*/) const {

    if( debug ) printf( "Debug: MonteRay_SphericalGrid::rayTrace -- \n");

    int indices[3] = {0, 0, 0}; // current position indices in the grid, must be int because can be outside

    // Crossing distance in R direction
    {
        gpuRayFloat_t particleRSq = calcParticleRSq( pos );
        indices[R] = pRVertices->getRadialIndexFromRSq(particleRSq);

        if( debug ) printf( "Debug: MonteRay_SphericalGrid::rayTrace -- dimension=%d, index=%d\n", R, indices[R]);

        radialCrossingDistances( R, threadID, rayInfo, pos, dir, indices[R], distance );

        if( debug ) printf( "Debug: MonteRay_SphericalGrid::rayTrace -- dimension=%d, number of radial crossings = %d\n", R, rayInfo.getCrossingSize(R,threadID) );

        // if outside and ray doesn't move inside then ray never enters the grid
        if( isIndexOutside(R,indices[R]) && rayInfo.getCrossingSize(R,threadID) == 0   ) {
            return;
        }
    }

    orderCrossings<1>( threadID, rayInfo, indices, distance, outsideDistances );

    if( debug ) printf( "Debug: MonteRay_SphericalGrid::rayTrace -- number of total crossings = %d\n", rayInfo.getRayCastSize(threadID) );
    return;
}

CUDA_CALLABLE_MEMBER
void
MonteRay_SphericalGrid::crossingDistance(
        const unsigned dim,
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        const GridBins_t::Position_t& pos,
        const GridBins_t::Direction_t& dir,
        const gpuRayFloat_t distance ) const {

    int index = pRVertices->getRadialIndexFromRSq(calcParticleRSq(pos));
    radialCrossingDistances( dim, threadID, rayInfo, pos, dir, index, distance );
    return;
}

CUDA_CALLABLE_MEMBER
void
MonteRay_SphericalGrid::radialCrossingDistances(
        const unsigned dim,
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        const Position_t& pos,
        const Direction_t& dir,
        unsigned rIndex,
        const gpuRayFloat_t distance ) const {

    //------ Distance to Sphere's Radial-boundary
    if( debug ) {
        printf("Debug: MonteRay_SphericalGrid::radialCrossingDistances -- \n");
    }

    gpuRayFloat_t particleRSq = calcParticleRSq( pos );

    gpuRayFloat_t           A = calcQuadraticA( dir );
    gpuRayFloat_t           B = calcQuadraticB( pos, dir);

    // trace inward
    bool rayTerminated = radialCrossingDistanceSingleDirection<false>(
            dim,
            threadID,
            rayInfo,
            *pRVertices,
            particleRSq,
            A,
            B,
            distance,
            rIndex);

    if( debug ) {
        printf("Debug: Inward ray trace size=%d\n",rayInfo.getCrossingSize(dim, threadID));
        if( rayTerminated ) {
            printf("Debug: - ray terminated!\n");
        } else {
            printf("Debug: - ray not terminated!\n");
        }
    }

    // trace outward
    if( ! rayTerminated ) {
        if( !isIndexOutside(R, rIndex) ) {
            radialCrossingDistanceSingleDirection<true>(
                    dim,
                    threadID,
                    rayInfo,
                    *pRVertices,
                    particleRSq,
                    A,
                    B,
                    distance,
                    rIndex);
        } else {
            rayInfo.addCrossingCell( dim, threadID, rIndex, distance );
        }
    }
}

CUDA_CALLABLE_MEMBER
void
MonteRay_SphericalGrid::radialCrossingDistances(
        const unsigned dim,
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        const Position_t& pos,
        const Direction_t& dir,
        const gpuRayFloat_t distance ) const {

    gpuRayFloat_t particleRSq = calcParticleRSq( pos );
    unsigned rIndex = pRVertices->getRadialIndexFromRSq(particleRSq);
    radialCrossingDistances( dim, threadID, rayInfo, pos, dir, rIndex, distance );
}

template
CUDA_CALLABLE_MEMBER
void
MonteRay_SphericalGrid::radialCrossingDistancesSingleDirection<true>(
        const unsigned dim,
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        const Position_t& pos,
        const Direction_t& dir,
        const gpuRayFloat_t distance) const;

template
CUDA_CALLABLE_MEMBER
void
MonteRay_SphericalGrid::radialCrossingDistancesSingleDirection<false>(
        const unsigned dim,
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        const Position_t& pos,
        const Direction_t& dir,
        const gpuRayFloat_t distance) const;

} /* namespace MonteRay */
