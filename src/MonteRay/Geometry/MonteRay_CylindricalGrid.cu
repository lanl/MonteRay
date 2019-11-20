#include "MonteRay_CylindricalGrid.t.hh"
#include "MonteRayDefinitions.hh"
#include "MonteRayConstants.hh"
#include "MonteRay_SingleValueCopyMemory.t.hh"
#include "MonteRayCopyMemory.t.hh"
#include "MonteRayParallelAssistant.hh"

#include <float.h>

namespace MonteRay {

using ptrCylindricalGrid_result_t = MonteRay_SingleValueCopyMemory<MonteRay_CylindricalGrid*>;

CUDA_CALLABLE_KERNEL  createDeviceInstance(MonteRay_CylindricalGrid** pPtrInstance, ptrCylindricalGrid_result_t* pResult, MonteRay_GridBins* pGridR, MonteRay_GridBins* pGridCZ ) {
    *pPtrInstance = new MonteRay_CylindricalGrid( 2, pGridR, pGridCZ );
    pResult->v = *pPtrInstance;
    //if( debug ) printf( "Debug: createDeviceInstance -- pPtrInstance = %d\n", pPtrInstance );
}

CUDA_CALLABLE_KERNEL  deleteDeviceInstance(MonteRay_CylindricalGrid** pPtrInstance) {
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
    if( DIM > 1 ) { pZVertices = pBins[CZ]; }
    //if( DIM > 2 ) { pThetaVertices = pBins[Theta]; }
    validate();
}

CUDA_CALLABLE_MEMBER
MonteRay_CylindricalGrid::MonteRay_CylindricalGrid(unsigned dim, GridBins_t* pGridR, GridBins_t* pGridCZ ) :
MonteRay_GridSystemInterface(dim)
{
    MONTERAY_VERIFY( dim == DimMax, "MonteRay_CylindricalGrid::ctor -- only 2-D is allowed" ); // No greater than 2-D.

    DIM = 2;
    pRVertices = pGridR;
    pZVertices = pGridCZ;
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

#ifndef NDEBUG
    if( debug ) std::cout << "Debug: MonteRay_CylindricalGrid::copyToGPU \n";
#endif

#ifdef __CUDACC__
    if( ! MonteRay::isWorkGroupMaster() ) return;
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
Array<int, 3> MonteRay_CylindricalGrid::calcIndices(const GridBins_t::Position_t& pos) const {

  Position_t cyl_pos = convertFromCartesian( pos );
  Array<int, 3> indices;
  indices[R] = pRVertices->getRadialIndexFromR( cyl_pos[R] );

  if( DIM>1 ) { indices[CZ] = getAxialIndex( cyl_pos[CZ] ); }

  return indices;
}

CUDA_CALLABLE_MEMBER
MonteRay_CylindricalGrid::Position_t
MonteRay_CylindricalGrid::convertFromCartesian( const Position_t& pos) const {

    return { Math::sqrt(pos[x]*pos[x] + pos[y]*pos[y]), pos[z], 0.0 };

    // TPB: retain comment if we want to implement RZT geometries
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

    // return particleMeshPosition;
}


CUDA_CALLABLE_MEMBER
unsigned
MonteRay_CylindricalGrid::getIndex( const Position_t& particle_pos) const{
#ifndef NDEBUG
  if( debug ) { printf("Debug: MonteRay_CylindricalGrid::getIndex -- starting\n"); }
#endif
  const auto indices = calcIndices(particle_pos);
  // set to max if outside the grid
  for( auto d = 0; d < DIM; ++d ) {
    if( isIndexOutside(d, indices[d] ) ) {
      return std::numeric_limits<unsigned>::max(); 
    }
  }
  return calcIndex( indices.data() );;
}

CUDA_CALLABLE_MEMBER
bool
MonteRay_CylindricalGrid::isIndexOutside( unsigned d, int i) const {
    MONTERAY_ASSERT( d < 3 );

    if( d == R ) {
        MONTERAY_ASSERT( i >= 0 );
        if( i >= numRBins ) { return true; }
    }

    if( d == CZ ) {
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
MonteRay_CylindricalGrid::rayTrace(
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        const GridBins_t::Position_t& pos,
        const GridBins_t::Position_t& dir,
        const gpuRayFloat_t distance,
        const bool outsideDistances ) const{

#ifndef NDEBUG
    if( debug ) printf( "Debug: MonteRay_CylindricalGrid::rayTrace -- \n");
#endif

    int indices[COORD_DIM]; // current position indices in the grid, must be int because can be outside

    // Crossing distance in R direction
    {
        gpuRayFloat_t particleRSq = calcParticleRSq( pos );
        indices[R] = pRVertices->getRadialIndexFromRSq(particleRSq);

#ifndef NDEBUG
        if( debug ) printf( "Debug: MonteRay_CylindricalGrid::rayTrace -- R Direction -  dimension=%d, index=%d\n", R, indices[R]);
#endif

        radialCrossingDistances(R, threadID, rayInfo, pos, dir, particleRSq, indices[R], distance );

#ifndef NDEBUG
        if( debug ) printf( "Debug: MonteRay_CylindricalGrid::rayTrace -- R Direction -  dimension=%d, number of radial crossings = %d\n", R, rayInfo.getCrossingSize(R,threadID) );
#endif

        // if outside and ray doesn't move inside then ray never enters the grid
        if( isIndexOutside(R,indices[R]) && rayInfo.getCrossingSize(R,threadID) == 0   ) {
            return;
        }
    }

    // Crossing distance in CZ direction
    if( DIM  > 1 ) {
        indices[CZ] = pZVertices->getLinearIndex( pos[z] );

#ifndef NDEBUG
        if( debug ) printf( "Debug: MonteRay_CylindricalGrid::rayTrace -- CZ Direction -  dimension=%d, index=%d\n", CZ, indices[CZ]);
#endif

        planarCrossingDistance( CZ, threadID, rayInfo, *pZVertices, pos[z], dir[z], distance, indices[CZ] );

#ifndef NDEBUG
        if( debug ) printf( "Debug: MonteRay_CylindricalGrid::rayTrace -- CZ Direction -  dimension=%d, number of planar crossings = %d\n", CZ, rayInfo.getCrossingSize(CZ,threadID) );
#endif

        // if outside and ray doesn't move inside then ray never enters the grid
        if( isIndexOutside(CZ,indices[CZ]) && rayInfo.getCrossingSize(CZ,threadID) == 0  ) {
            return;
        }
    }

    orderCrossings<2>( threadID, rayInfo, indices, distance, outsideDistances );

#ifndef NDEBUG
    if( debug ) printf( "Debug: MonteRay_CylindricalGrid::rayTrace -- number of total crossings = %d\n", rayInfo.getRayCastSize(threadID) );
#endif
    return;
}

CUDA_CALLABLE_MEMBER
DirectionAndSpeed MonteRay_CylindricalGrid::convertToCellReferenceFrame(
    const Vector3D<gpuRayFloat_t>& cellVelocity,
    const GridBins_t::Position_t& pos,
    GridBins_t::Direction_t dir,
    gpuRayFloat_t speed) const {

  dir *= speed;
  dir[z] = dir[z] - cellVelocity[CZ];
  auto radiusSqr = pos[x]*pos[x] + pos[y]*pos[y];
  if(radiusSqr > 0.0){
    auto relativeRadialVelocity = cellVelocity[R] / Math::sqrt(radiusSqr);
    dir[x] = dir[x] - pos[x] * relativeRadialVelocity;
    dir[y] = dir[y] - pos[y] * relativeRadialVelocity;
  }
  speed = dir.magnitude();
  dir /= speed;
  return {dir, speed};
}

CUDA_CALLABLE_MEMBER 
DistAndDir MonteRay_CylindricalGrid::getMinRadialDistAndDir( 
    const GridBins_t::Position_t& pos, 
    const GridBins_t::Direction_t& dir, 
    const int radialIndex) const { 

  const auto A = calcQuadraticA( dir );
  const auto B = calcQuadraticB( pos, dir);
  const auto pRSq = calcParticleRSq(pos);
  // recalc this every time in case the cell velocity changes the ray's radial trajectory
  bool inwardTrajectory = (radialIndex == 0 ? false : isMovingInward(pos, dir));

  Roots roots = FindPositiveRoots(A,B, pRSq - pRVertices->verticesSq[radialIndex - inwardTrajectory]);
   // no-hit on surface, turn ray around
  if (roots.areInf( )) {
    // if point is outside mesh, then it just doesn't intersect the mesh
    if (radialIndex >= pRVertices->getNumBins()){
      return {std::numeric_limits<gpuRayFloat_t>::infinity(), R, not inwardTrajectory};
    }
    inwardTrajectory = false;
    roots = FindPositiveRoots(A,B, pRSq - pRVertices->verticesSq[radialIndex]);
  }
  return {roots.min(), R, not inwardTrajectory};
}
  

CUDA_CALLABLE_MEMBER 
DistAndDir MonteRay_CylindricalGrid::getMinDistToSurface( 
    const GridBins_t::Position_t& pos, 
    const GridBins_t::Direction_t& dir, 
    const int indices[]) const { 

  auto minRadialDistAndDir = getMinRadialDistAndDir(pos, dir, indices[R]);
  auto minCZDistAndDir = DistAndDir{(pZVertices->vertices[indices[CZ] + Math::signbit(-dir[z])] - pos[z])/dir[z], CZ, Math::signbit(-dir[z])};

  return minRadialDistAndDir.distance() < minCZDistAndDir.distance() ? 
    minRadialDistAndDir : minCZDistAndDir;
}

CUDA_CALLABLE_MEMBER
gpuRayFloat_t MonteRay_CylindricalGrid::getDistanceToInsideOfMesh(const GridBins_t::Position_t& pos, const GridBins_t::Direction_t& dir) const {
  gpuRayFloat_t radialDist = (calcParticleRSq(pos) >= pRVertices->verticesSq[pRVertices->getNumBins() - 1]) ? 
      (isMovingInward(pos, dir) ? 
         getMinRadialDistAndDir(pos, dir, pRVertices->getNumBins()).distance() + std::numeric_limits<gpuRayFloat_t>::epsilon() : 
         Roots::inf) :
      0.0;
  return Math::max(radialDist, pZVertices->distanceToGetInsideLinearMesh(pos, dir, z));
}

CUDA_CALLABLE_MEMBER
void
MonteRay_CylindricalGrid::rayTraceWithMovingMaterials( const unsigned threadID,
          RayWorkInfo& rayInfo,
          GridBins_t::Position_t pos,
          const GridBins_t::Direction_t& dir,
          gpuRayFloat_t distanceRemaining,
          const gpuRayFloat_t speed,
          const MaterialProperties& matProps,
          const bool outsideDistances ) const {

  auto distanceToInsideOfMesh = getDistanceToInsideOfMesh(pos, dir);
  if (distanceRemaining < distanceToInsideOfMesh){
    return;
  }
  pos += distanceToInsideOfMesh*dir;
  distanceRemaining -= distanceToInsideOfMesh;
  auto indices = calcIndices(pos);

  while(distanceRemaining > std::numeric_limits<gpuRayFloat_t>::epsilon()){
    // get "global" cell index, which is set to max if cell is outside mesh
    auto cellIndex = calcIndex(indices.data());

    // adjust dir and energy if moving materials
    auto dirAndSpeed = matProps.usingMaterialMotion() ?
      convertToCellReferenceFrame(matProps.velocity(cellIndex), pos, dir, speed) : 
      DirectionAndSpeed{dir, speed};

    auto newDir = dirAndSpeed.direction();
    auto distAndDir = getMinDistToSurface(pos, dirAndSpeed.direction(), indices.data());

    // min dist found, move ray and tally
    if ( distAndDir.distance() < distanceRemaining ) {
      rayInfo.addRayCastCell(threadID, cellIndex, distAndDir.distance());

      // update distance and position
      distanceRemaining -= distAndDir.distance();
      pos += distAndDir.distance()*dirAndSpeed.direction();

      // update indices
      distAndDir.isPositiveDir() ?
        indices[distAndDir.dimension()]++ : 
        indices[distAndDir.dimension()]-- ;

      // short-circuit if ray left the mesh
      if( isIndexOutside(distAndDir.dimension(), indices[distAndDir.dimension()] ) ) {
        return;
      }
    } else {
      rayInfo.addRayCastCell(threadID, cellIndex, distanceRemaining);
      return;
    }
  }
  return;
}

CUDA_CALLABLE_MEMBER
void
MonteRay_CylindricalGrid::crossingDistance(
        const unsigned dim,
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        const GridBins_t::Position_t& pos,
        const GridBins_t::Direction_t& dir,
        const gpuRayFloat_t distance ) const {

    if( dim == R ) {
        double rSq = calcParticleRSq(pos);
        int index = pRVertices->getRadialIndexFromRSq(rSq);
        radialCrossingDistances( dim, threadID, rayInfo, pos, dir, rSq, index, distance );
    }

    if( dim == CZ ) {
        int index = pZVertices->getLinearIndex( pos[z] );
        planarCrossingDistance( dim, threadID, rayInfo, *pZVertices, pos[z], dir[z], distance, index );
    }
    return;
}

CUDA_CALLABLE_MEMBER
void
MonteRay_CylindricalGrid::radialCrossingDistances(
        const unsigned dim,
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        const Position_t& pos,
        const Direction_t& dir,
        const double particleRSq,
        const unsigned rIndex,
        const gpuRayFloat_t distance ) const {
    //------ Distance to Cylinder's Radial-boundary

#ifndef NDEBUG
    if( debug ) {
        printf("Debug: MonteRay_CylindricalGrid::radialCrossingDistances -- \n");
    }
#endif

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

#ifndef NDEBUG
    if( debug ) {
        printf("Debug: Inward ray trace size=%d\n",rayInfo.getCrossingSize(dim, threadID));
        if( rayTerminated ) {
            printf("Debug: - ray terminated!\n");
        } else {
            printf("Debug: - ray not terminated!\n");
        }
    }
#endif

    // trace outward
    if( ! rayTerminated ) {
        if( !isIndexOutside(R, rIndex) ) {
            radialCrossingDistanceSingleDirection<true>(dim, threadID, rayInfo, *pRVertices, particleRSq, A, B, distance, rIndex);
        } else {
            rayInfo.addCrossingCell( dim, threadID, rIndex, distance );
        }
    }
}

CUDA_CALLABLE_MEMBER
void
MonteRay_CylindricalGrid::radialCrossingDistances(
        const unsigned dim,
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        const Position_t& pos,
        const Direction_t& dir,
        const gpuRayFloat_t distance ) const {

    gpuRayFloat_t particleRSq = calcParticleRSq( pos );
    unsigned rIndex = pRVertices->getRadialIndexFromRSq(particleRSq);
    radialCrossingDistances( dim, threadID, rayInfo, pos, dir, particleRSq, rIndex, distance );
}

CUDA_CALLABLE_MEMBER
unsigned
MonteRay_CylindricalGrid::getNumBins( unsigned d) const {
    MONTERAY_ASSERT( d < COORD_DIM );

#ifndef NDEBUG
    if( debug ) printf("Debug: MonteRay_CylindricalGrid::getNumBins -- d= %d\n", d);
    if( debug ) printf("Debug: MonteRay_CylindricalGrid::getNumBins --calling pGridBins[d]->getNumBins()\n");
#endif

    if( d == 0 ) {
        return pRVertices->getNumBins();
    }
    if( d == 1 ) {
        return pZVertices->getNumBins();
    }
    return 0;
}

template
CUDA_CALLABLE_MEMBER
void
MonteRay_CylindricalGrid::radialCrossingDistancesSingleDirection<true>(
        const unsigned dim,
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        const Position_t& pos,
        const Direction_t& dir,
        const gpuRayFloat_t distance) const;

template
CUDA_CALLABLE_MEMBER
void
MonteRay_CylindricalGrid::radialCrossingDistancesSingleDirection<false>(
        const unsigned dim,
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        const Position_t& pos,
        const Direction_t& dir,
        const gpuRayFloat_t distance) const;


} /* namespace MonteRay */
