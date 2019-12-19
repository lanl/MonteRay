#include "MonteRay_CylindricalGrid.t.hh"
#include "MonteRayDefinitions.hh"
#include "MonteRayConstants.hh"
#include "MonteRayParallelAssistant.hh"

#include <float.h>

namespace MonteRay {

// TPB TODO: see if constructor is necessary and appropriate
/* CUDA_CALLABLE_MEMBER */
/* MonteRay_CylindricalGrid::MonteRay_CylindricalGrid(unsigned dim, const GridBins_t* const pArrayOfpGridInfo_t pBins) : */
/* MonteRay_GridSystemInterface(dim) */
/* { */
/*     MONTERAY_VERIFY( dim == DimMax, "MonteRay_CylindricalGrid::ctor -- only 2-D is allowed" ); // No greater than 2-D. */

/*     DIM = dim; */

/*     gridBins[R] = pBins[R]; */
/*     if( DIM > 1 ) { gridBins[CZ] = pBins[CZ]; } */
/*     //if( DIM > 2 ) { pThetaVertices = pBins[Theta]; } */
/*     validate(); */
/* } */

MonteRay_CylindricalGrid::MonteRay_CylindricalGrid(int dim, GridBins_t gridR, GridBins_t gridCZ ) :
  MonteRay_GridSystemInterface({std::move(gridR), std::move(gridCZ)}, dim)
{
  MONTERAY_VERIFY( dim == DimMax, "MonteRay_CylindricalGrid::ctor -- only 2-D is allowed" ); // No greater than 2-D.

  if (not gridBins[CZ].isLinear()) { 
    throw std::runtime_error("In Constructor MonteRay_CylindricalGrid(dim, gridBins[R], gridBins[CZ]) -- gridBins[CZ] are not marked as linear !!!"); 
  }
  if (not gridBins[R].isRadial()) { 
    throw std::runtime_error("In Constructor MonteRay_CylindricalGrid(dim, gridBins[R], gridBins[CZ]) -- gridBins[R] are not marked as radial !!!"); 
  }
}

CUDA_CALLABLE_MEMBER
Array<int, 3> MonteRay_CylindricalGrid::calcIndices(const GridBins_t::Position_t& pos) const {
  Position_t cyl_pos = convertFromCartesian( pos );
  Array<int, 3> indices;
  indices[R] = gridBins[R].getRadialIndexFromR( cyl_pos[R] );

  if( DIM>1 ) { indices[CZ] = getAxialIndex( cyl_pos[CZ] ); }

  return indices;
}

CUDA_CALLABLE_MEMBER
MonteRay_CylindricalGrid::Position_t
MonteRay_CylindricalGrid::convertFromCartesian(const Position_t& pos) const {
    return { Math::sqrt(pos[x]*pos[x] + pos[y]*pos[y]), pos[z], 0.0 };
}

CUDA_CALLABLE_MEMBER
unsigned
MonteRay_CylindricalGrid::getIndex(const Position_t& particle_pos) const{
  const auto indices = calcIndices(particle_pos);
  // set to max if outside the grid
  for( int d = 0; d < DIM; ++d ) {
    if( isIndexOutside(d, indices[d] ) ) {
      return std::numeric_limits<unsigned>::max(); 
    }
  }
  return calcIndex( indices.data() );
}

CUDA_CALLABLE_MEMBER
bool
MonteRay_CylindricalGrid::isIndexOutside(unsigned d, int i) const {
  MONTERAY_ASSERT( d < 3 );
  
  if( d == R ) {
    MONTERAY_ASSERT( i >= 0 );
    if( i >= gridBins[R].getNumBins() ) { return true; }
  } else if( d == CZ ) {
    if( i < 0 ||  i >= gridBins[CZ].getNumBins() ){ return true; }
  }

  return false;
}

CUDA_CALLABLE_MEMBER
uint3 MonteRay_CylindricalGrid::calcIJK(unsigned index ) const {
  uint3 indices;

  indices.z = 0; // no theta support in this grid
  indices.y = index / gridBins[R].getNumBins();
  indices.x = index - indices.y * gridBins[R].getNumBins(); 

  return indices;
}

CUDA_CALLABLE_MEMBER
bool
MonteRay_CylindricalGrid::isOutside( const int i[] ) const {
  for( int d=0; d<DIM; ++d){
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
      innerRadiusSq = gridBins[R].verticesSq[indices.x-1];
  }
  gpuRayFloat_t outerRadiusSq = gridBins[R].verticesSq[indices.x];

  gpuRayFloat_t volume = MonteRay::pi * ( outerRadiusSq - innerRadiusSq );
  if( DIM > 1 ) volume *= gridBins[CZ].vertices[ indices.y + 1 ] - gridBins[CZ].vertices[ indices.y ];

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

  int indices[DimMax]; // current position indices in the grid, must be int because can be outside

  // Crossing distance in R direction
  {
    gpuRayFloat_t particleRSq = calcParticleRSq( pos );
    indices[R] = gridBins[R].getRadialIndexFromRSq(particleRSq);

    radialCrossingDistances(R, threadID, rayInfo, pos, dir, particleRSq, indices[R], distance );

    // if outside and ray doesn't move inside then ray never enters the grid
    if( isIndexOutside(R,indices[R]) && rayInfo.getCrossingSize(R,threadID) == 0   ) {
        return;
    }
  }

  // Crossing distance in CZ direction
  if( DIM  > 1 ) {
    indices[CZ] = gridBins[CZ].getLinearIndex( pos[z] );

    planarCrossingDistance( CZ, threadID, rayInfo, gridBins[CZ], pos[z], dir[z], distance, indices[CZ] );

    // if outside and ray doesn't move inside then ray never enters the grid
    if( isIndexOutside(CZ,indices[CZ]) && rayInfo.getCrossingSize(CZ,threadID) == 0  ) {
        return;
    }
  }

  orderCrossings<2>( threadID, rayInfo, indices, distance, outsideDistances );

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

  // this algorithm avoids calculating dir dot surface normal for every interaction by occasionally taking the max root 
  const auto A = calcQuadraticA( dir );
  const auto B = calcQuadraticB( pos, dir);
  const auto pRSq = calcParticleRSq(pos);
  // recalc this every time in case the cell velocity changes the ray's radial trajectory
  // special treatment for radialIndex == 0
  if (radialIndex == 0){
    auto root = FindMaxValidRoot(A,B, pRSq - gridBins[R].verticesSq[radialIndex]);
    return {root, R, true};
  }

  bool inwardTrajectory = isMovingInward(pos, dir);
  Roots roots = FindPositiveRoots(A,B, pRSq - gridBins[R].verticesSq[radialIndex - inwardTrajectory]);

  if (roots.areInf()) { // no-hit on surface, check other surface
    if (radialIndex >= gridBins[R].getNumBins()){
      return {std::numeric_limits<gpuRayFloat_t>::infinity(), R, not inwardTrajectory};
    }
    auto root = FindMaxValidRoot(A,B, pRSq - gridBins[R].verticesSq[radialIndex]);
    inwardTrajectory = false;
    // guaranteed to intersect only once, point is on edge or inside circle
    return {root, R, true};
  }

  return DistAndDir{roots.min(), R, not inwardTrajectory};
}
  

CUDA_CALLABLE_MEMBER 
DistAndDir MonteRay_CylindricalGrid::getMinDistToSurface( 
    const GridBins_t::Position_t& pos, 
    const GridBins_t::Direction_t& dir, 
    const int indices[]) const { 

  auto minRadialDistAndDir = getMinRadialDistAndDir(pos, dir, indices[R]);
  auto minCZDistAndDir = DistAndDir{(gridBins[CZ].vertices[indices[CZ] + Math::signbit(-dir[z])] - pos[z])/dir[z], CZ, Math::signbit(-dir[z])};

  // TESTING TEST TEST
  auto minDist = Math::min(minRadialDistAndDir.distance(), minCZDistAndDir.distance());
  if (minDist < 0){
    printf("neg dist pos (%f %f %f), dir (%f %f %f), indices (%d %d) \n", pos[x], pos[y], pos[z], dir[x], dir[y], dir[z], indices[R], indices[CZ]); 
  }
  if (Math::abs(minDist) == std::numeric_limits<gpuRayFloat_t>::infinity()) { 
    printf("inf dist pos (%f %f %f), dir (%f %f %f), indices (%d %d) \n", pos[x], pos[y], pos[z], dir[x], dir[y], dir[z], indices[R], indices[CZ]); 
  }
  return minRadialDistAndDir.distance() < minCZDistAndDir.distance() ? 
    minRadialDistAndDir : minCZDistAndDir;
}

CUDA_CALLABLE_MEMBER
gpuRayFloat_t MonteRay_CylindricalGrid::getDistanceToInsideOfMesh(const GridBins_t::Position_t& pos, const GridBins_t::Direction_t& dir) const {
  gpuRayFloat_t radialDist = (calcParticleRSq(pos) >= gridBins[R].verticesSq[gridBins[R].getNumVerticesSq() - 1]) ? 
      (isMovingInward(pos, dir) ? 
         getMinRadialDistAndDir(pos, dir, gridBins[R].getNumVertices()).distance() + std::numeric_limits<gpuRayFloat_t>::epsilon() : 
         Roots::inf) :
      0.0;
  return Math::max(radialDist, gridBins[CZ].distanceToGetInsideLinearMesh(pos[z], dir[z]));
}

CUDA_CALLABLE_MEMBER
void MonteRay_CylindricalGrid::crossingDistance(
        const unsigned dim,
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        const GridBins_t::Position_t& pos,
        const GridBins_t::Direction_t& dir,
        const gpuRayFloat_t distance ) const {

  if( dim == R ) {
    double rSq = calcParticleRSq(pos);
    int index = gridBins[R].getRadialIndexFromRSq(rSq);
    radialCrossingDistances( dim, threadID, rayInfo, pos, dir, rSq, index, distance );
  }

  if( dim == CZ ) {
    int index = gridBins[CZ].getLinearIndex( pos[z] );
    planarCrossingDistance( dim, threadID, rayInfo, gridBins[CZ], pos[z], dir[z], distance, index );
  }
  return;
}

CUDA_CALLABLE_MEMBER
void MonteRay_CylindricalGrid::radialCrossingDistances(
        const unsigned dim,
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        const Position_t& pos,
        const Direction_t& dir,
        const double particleRSq,
        const unsigned rIndex,
        const gpuRayFloat_t distance ) const {
    //------ Distance to Cylinder's Radial-boundary

  gpuRayFloat_t           A = calcQuadraticA( dir );
  gpuRayFloat_t           B = calcQuadraticB( pos, dir);

  // trace inward
  bool rayTerminated = radialCrossingDistanceSingleDirection<false>(
          dim,
          threadID,
          rayInfo,
          gridBins[R],
          particleRSq,
          A,
          B,
          distance,
          rIndex);

  // trace outward
  if( ! rayTerminated ) {
    if( !isIndexOutside(R, rIndex) ) {
      radialCrossingDistanceSingleDirection<true>(dim, threadID, rayInfo, gridBins[R], particleRSq, A, B, distance, rIndex);
    } else {
      rayInfo.addCrossingCell( dim, threadID, rIndex, distance );
    }
  }
}

CUDA_CALLABLE_MEMBER
void MonteRay_CylindricalGrid::radialCrossingDistances(
        const unsigned dim,
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        const Position_t& pos,
        const Direction_t& dir,
        const gpuRayFloat_t distance ) const {

    gpuRayFloat_t particleRSq = calcParticleRSq( pos );
    auto rIndex = gridBins[R].getRadialIndexFromRSq(particleRSq);
    radialCrossingDistances( dim, threadID, rayInfo, pos, dir, particleRSq, rIndex, distance );
}

template
CUDA_CALLABLE_MEMBER
void MonteRay_CylindricalGrid::radialCrossingDistancesSingleDirection<true>(
        const unsigned dim,
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        const Position_t& pos,
        const Direction_t& dir,
        const gpuRayFloat_t distance) const;

template
CUDA_CALLABLE_MEMBER
void MonteRay_CylindricalGrid::radialCrossingDistancesSingleDirection<false>(
        const unsigned dim,
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        const Position_t& pos,
        const Direction_t& dir,
        const gpuRayFloat_t distance) const;


} /* namespace MonteRay */
