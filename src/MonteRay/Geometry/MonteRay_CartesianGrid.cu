#include "MonteRay_CartesianGrid.hh"
#include "MonteRayDefinitions.hh"
#include "RayWorkInfo.hh"
#include "MonteRayParallelAssistant.hh"
#include <limits>

#include <float.h>

namespace MonteRay {

MonteRay_CartesianGrid::MonteRay_CartesianGrid(int dim, GridBins_t gridX, GridBins_t gridY, GridBins_t gridZ ) :
  MonteRay_GridSystemInterface({std::move(gridX), std::move(gridY), std::move(gridZ)}, dim)
{
  MONTERAY_VERIFY( dim == DimMax, "MonteRay_CartesianGrid::ctor -- only 3-D is allowed" ); // No greater than 3-D.
}

CUDA_CALLABLE_MEMBER
unsigned MonteRay_CartesianGrid::getIndex( const GridBins_t::Position_t& particle_pos) const{
  int indices[3]= {0,0,0};
  for( int d = 0; d < DIM; ++d ) {
    indices[d] = getDimIndex(d, particle_pos[d] );

    // outside the grid
    if( isIndexOutside(d, indices[d] ) ) {
      return std::numeric_limits<unsigned>::max();
    }
  }
  return calcIndex( indices );
}

CUDA_CALLABLE_MEMBER
gpuRayFloat_t
MonteRay_CartesianGrid::getVolume(unsigned index ) const {
  gpuRayFloat_t volume=1.0;
  uint3 indices = calcIJK( index );
  volume *= gridBins[0].vertices[ indices.x + 1 ] - gridBins[0].vertices[ indices.x ];
  volume *= gridBins[1].vertices[ indices.y + 1 ] - gridBins[1].vertices[ indices.y ];
  volume *= gridBins[2].vertices[ indices.z + 1 ] - gridBins[2].vertices[ indices.z ];
  return volume;
}

CUDA_CALLABLE_MEMBER
unsigned
MonteRay_CartesianGrid::getNumBins( unsigned d) const {
  return gridBins[d].getNumBins();
}

CUDA_CALLABLE_MEMBER
uint3
MonteRay_CartesianGrid::calcIJK( unsigned index ) const {
  uint3 indices;

  uint3 offsets;
  offsets.x = 1;

  offsets.y = getNumBins(0);
  offsets.z = getNumBins(0)*getNumBins(1);

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
void
MonteRay_CartesianGrid::rayTrace(
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        const GridBins_t::Position_t& particle_pos,
        const GridBins_t::Position_t& particle_dir,
        const gpuRayFloat_t distance,
        const bool outsideDistances ) const{

  int indices[3] = {0, 0, 0}; // current position indices in the grid, must be int because can be outside

  for( int d=0; d<DIM; ++d){

    indices[d] = getDimIndex(d, particle_pos[d] );

    planarCrossingDistance( d, threadID, rayInfo, gridBins[d], particle_pos[d], particle_dir[d], distance,indices[d]);

    // if outside and ray doesn't move inside then ray never enters the grid
    if( isIndexOutside(d,indices[d]) && rayInfo.getCrossingSize(d,threadID) == 0  ) {
        return;
    }
  }

  orderCrossings<3>( threadID, rayInfo, indices, distance, outsideDistances );

  return;
}


CUDA_CALLABLE_MEMBER
DirectionAndSpeed MonteRay_CartesianGrid::convertToCellReferenceFrame(
    const Vector3D<gpuRayFloat_t>& cellVelocity,
    const GridBins_t::Position_t&, // here to maintain same API as other grid types
    GridBins_t::Direction_t dir,
    gpuRayFloat_t speed) const
{
  dir = dir*speed - cellVelocity;
  speed = dir.magnitude();
  dir /= speed;
  return {dir, speed};
}

CUDA_CALLABLE_MEMBER DistAndDir
MonteRay_CartesianGrid::getMinDistToSurface(
       const GridBins_t::Position_t& pos,
       const GridBins_t::Direction_t& dir,
       const int indices[] 
       ) const {
 int d = 0;
 gpuRayFloat_t minDistToSurf = std::numeric_limits<gpuRayFloat_t>::max();
 unsigned minDistIndex = 0;
 for (d=0; d<DIM; ++d){
   if( Math::abs(dir[d]) >= std::numeric_limits<gpuRayFloat_t>::epsilon() ) {
     auto distToSurface = (gridBins[d].vertices[indices[d] + Math::signbit(-dir[d])] - pos[d])/dir[d];
     if (distToSurface < minDistToSurf){
       minDistToSurf = distToSurface;
       minDistIndex = d;
     }
   }
  }
 if (minDistToSurf < 0) {
   minDistToSurf = 0;
 }
 return {minDistToSurf, minDistIndex, std::signbit(-dir[minDistIndex])};
}

CUDA_CALLABLE_MEMBER
gpuRayFloat_t MonteRay_CartesianGrid::getDistanceToInsideOfMesh(const GridBins_t::Position_t& pos, const GridBins_t::Direction_t& dir) const {
  gpuRayFloat_t dist = 0.0;
  for (int d = 0; d < DIM; d++){
    dist = Math::max(dist, gridBins[d].distanceToGetInsideLinearMesh(pos[d], dir[d]));
  }
  return dist;
}

CUDA_CALLABLE_MEMBER
void
MonteRay_CartesianGrid::crossingDistance(
        const unsigned dim,
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        const GridBins_t::Position_t& pos,
        const GridBins_t::Direction_t& dir,
        const gpuRayFloat_t distance ) const {

  crossingDistance( dim, threadID, rayInfo, pos[dim], dir[dim], distance);
}

CUDA_CALLABLE_MEMBER
void
MonteRay_CartesianGrid::crossingDistance(
        const unsigned dim,
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        const gpuRayFloat_t pos,
        const gpuRayFloat_t dir,
        const gpuRayFloat_t distance ) const {

  crossingDistance(dim, threadID, rayInfo, gridBins[dim], pos, dir, distance, false);
  return;
}

CUDA_CALLABLE_MEMBER
void
MonteRay_CartesianGrid::crossingDistance(
        const unsigned dim,
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        const GridBins_t& Bins,
        const gpuRayFloat_t pos,
        const gpuRayFloat_t dir,
        const gpuRayFloat_t distance,
        const bool equal_spacing) const {

  int index = Bins.getLinearIndex(pos);
  planarCrossingDistance( dim, threadID, rayInfo, Bins, pos, dir, distance, index);
  return;
}


} /* namespace MonteRay */
