#ifndef EXPECTEDPATHLENGTH_T_HH_
#define EXPECTEDPATHLENGTH_T_HH_

#include <algorithm>

#ifdef __CUDACC__
#include <thrust/for_each.h>
#endif

#include "MaterialList.hh"
#include "Tally.hh"
#include "RayWorkInfo.hh"
#include "GPUAtomicAdd.hh"
#include "GPUUtilityFunctions.hh"
#include "ExpectedPathLength.hh"
// TPB TODO: Enable photons with multiple weights/energies
// TPB TODO: Enable accurate time-binning - currently uses particle or ray's time - it does not break a ray into multiple time bins

namespace MonteRay{

template<typename Ray, typename Geometry, typename MaterialList>
CUDA_CALLABLE_MEMBER
void ExpectedPathLengthTally::tallyCollision(
        int particleID,
        const Geometry& geometry,
        const MaterialList& matList,
        const MaterialProperties& matProps,
        const Ray& p,
        RayWorkInfo& rayInfo) {
  gpuTallyType_t opticalPathLength = 0.0;

  if( p.energy[0] < 1e-20 ) {
    return;
  }

  geometry.rayTrace(particleID, rayInfo, p.pos, p.dir, static_cast<gpuRayFloat_t>(1.0e6), false);

  gpuFloatType_t materialXS[MAXNUMMATERIALS];
  for( unsigned i=0; i < matList.numMaterials(); ++i ){
    materialXS[i] = matList.material(i).getTotalXS(p.energy[0], 1.0);
  }

  for( unsigned i=0; i < rayInfo.getRayCastSize(particleID); ++i ){
    int cell = rayInfo.getRayCastCell(particleID,i);
    gpuRayFloat_t distance = rayInfo.getRayCastDist(particleID,i);
    if( cell == std::numeric_limits<unsigned>::max() ) continue;

    opticalPathLength += tallyCellSegment(matList, matProps, materialXS, 
            cell, distance, p, opticalPathLength);

    if( opticalPathLength > 5.0 ) {
      // cut off at 5 mean free paths
      return;
    }
  }
}

template<typename Ray, typename MaterialList>
CUDA_CALLABLE_MEMBER
ExpectedPathLengthTally::TallyFloat 
ExpectedPathLengthTally::tallyCellSegment( const MaterialList& matList,
        const MaterialProperties& matProps,
        const gpuFloatType_t* materialXS,
        unsigned cell,
        gpuRayFloat_t distance,
        const Ray& p,
        ExpectedPathLengthTally::TallyFloat opticalPathLength ) {

  typedef gpuTallyType_t xs_t;
  typedef gpuTallyType_t attenuation_t;
  typedef gpuTallyType_t score_t;

  xs_t totalXS = 0.0;
  unsigned numMaterials = matProps.numMats(cell);
  for( unsigned i=0; i<numMaterials; ++i ) {
    unsigned matID = matProps.getMaterialID(cell, i);
    gpuFloatType_t density = matProps.getDensity(cell, i );
    if( density > 1e-5 ) {
        totalXS += materialXS[matID]*density;
    }
  }

  attenuation_t attenuation = 1.0;
  score_t score = distance;
  gpuTallyType_t cellOpticalPathLength = totalXS*distance;

  if( totalXS >  1e-5 ) {
    attenuation =  exp( - cellOpticalPathLength ); // TPB: this is a double-precision exponent and this math is done in double precision
    score = ( 1.0 / totalXS ) * ( 1.0 - attenuation );
  }
  score *= exp( -opticalPathLength ) * p.weight[0];

  this->score(score, cell, p.energy[0], p.getTime());

  return cellOpticalPathLength;
}

template<typename Ray, typename Geometry, typename MaterialList>
CUDA_CALLABLE_MEMBER void ExpectedPathLengthTally::rayTraceOnGridWithMovingMaterials( 
          Ray ray,
          gpuRayFloat_t timeRemaining,
          const Geometry& geometry,
          const MaterialProperties& matProps,
          const MaterialList& matList){

  if( ray.energy[0] < std::numeric_limits<gpuRayFloat_t>::epsilon() ) {
    return;
  }
  auto distanceToInsideOfMesh = geometry.getDistanceToInsideOfMesh(ray.position(), ray.direction());
  if (distanceToInsideOfMesh/ray.speed() > timeRemaining){
    return;
  }

  ray.position() += distanceToInsideOfMesh*ray.direction();
  timeRemaining -= distanceToInsideOfMesh/ray.speed();
  auto indices = geometry.calcIndices(ray.position());

  gpuRayFloat_t opticalPathLength = 0.0;
  while(timeRemaining > std::numeric_limits<gpuRayFloat_t>::epsilon()){
    auto cellID = geometry.calcIndex(indices.data());

    // adjust dir and energy if moving materials
    // TODO: remove decltype ?
    using DirectionAndSpeed = decltype( geometry.convertToCellReferenceFrame(matProps.velocity(cellID), ray.position(), ray.direction(), ray.speed()) );
    auto dirAndSpeed = matProps.usingMaterialMotion() ?
      geometry.convertToCellReferenceFrame(matProps.velocity(cellID), ray.position(), ray.direction(), ray.speed()) : 
      DirectionAndSpeed{ray.direction(), ray.speed()};

    auto distAndDir = geometry.getMinDistToSurface(ray.position(), dirAndSpeed.direction(), indices.data());

    // min dist found, move ray and tally
    distAndDir.setDistance( Math::min(distAndDir.distance(), timeRemaining*dirAndSpeed.speed()) );

    // update time and position since these values are probs in registers still
    timeRemaining -= distAndDir.distance()/dirAndSpeed.speed();
    ray.position() += distAndDir.distance()*dirAndSpeed.direction();

    auto energy = dirAndSpeed.speed()*dirAndSpeed.speed()*
      inv_neutron_speed_from_energy_const()*inv_neutron_speed_from_energy_const();
    auto matIDs = matProps.cellMaterialIDs(cellID);
    // TODO: make this a function call 
    gpuFloatType_t totalXS = 0.0;
    const int numMaterials = static_cast<int>(matProps.getNumMats(cellID));
    for (int i = 0; i < numMaterials; i++) {
      totalXS += matList.material(matProps.matID(cellID, i)).getTotalXS(energy, matProps.materialDensity(cellID, i));
    }

    // calc attenuation, add score to tally, update ray's weight
    gpuFloatType_t attenuation = 1.0;
    auto score = ray.getWeight();
    if( totalXS > static_cast<gpuFloatType_t>(1e-5) ) {
      attenuation = Math::exp( -totalXS*distAndDir.distance() );
      opticalPathLength += distAndDir.distance() * totalXS;
      score *= ( static_cast<gpuRayFloat_t>(1.0) - attenuation )/totalXS;
    } else {
      score *= distAndDir.distance();
    }

    this->score(score, cellID, energy, ray.time);
    // adjust ray's weight
    ray.setWeight(ray.getWeight()*attenuation);

    if (opticalPathLength > 5.0) { return; } // TODO: make this configurable, e-5 is approx 0.7%

    // update indices
    distAndDir.isPositiveDir() ?
      indices[distAndDir.dimension()]++ : 
      indices[distAndDir.dimension()]-- ;

    // short-circuit if ray left the mesh
    if( geometry.isIndexOutside(distAndDir.dimension(), indices[distAndDir.dimension()] ) ) { return; }
  }
  return;
}

template<unsigned N, typename Geometry, typename MaterialList>
void ExpectedPathLengthTally::rayTraceTallyWithMovingMaterials(
          const RayList_t<N>* const collisionPoints,
          gpuFloatType_t timeRemaining,
          const Geometry* const geometry,
          const MaterialProperties* const matProps,
          const MaterialList* const matList,
          cudaStream_t* stream) {

#ifdef __CUDACC__
  thrust::for_each(thrust::cuda::par.on(*stream), collisionPoints->begin(), collisionPoints->end(),
    [=] __device__ (const auto& ray){
      this->rayTraceOnGridWithMovingMaterials(ray, timeRemaining, *geometry, *matProps, *matList);
    }
  );
#else 
  std::for_each(collisionPoints->begin(), collisionPoints->end(),
    [=] (const auto& ray){
      this->rayTraceOnGridWithMovingMaterials(ray, timeRemaining, *geometry, *matProps, *matList);
    }
  );
#endif
}

template<unsigned N, typename Geometry, typename MaterialList>
CUDA_CALLABLE_KERNEL 
rayTraceTally(const Geometry* const pGeometry,
        const RayList_t<N>* const pCP,
        const MaterialList* const pMatList,
        const MaterialProperties* const pMatProps,
        RayWorkInfo* const pRayInfo,
        ExpectedPathLengthTally* const tally) {
#ifdef __CUDACC__
    int threadID = threadIdx.x + blockIdx.x*blockDim.x;
#else
    int threadID = 0;
#endif
    int particleID = threadID;

    int num = pCP->size();

    while( particleID < num ) {
       const auto& p = pCP->getParticle(particleID);
       pRayInfo->clear( threadID );

       tally->tallyCollision(threadID, *pGeometry, *pMatList, *pMatProps, p, *pRayInfo);

#ifdef __CUDACC__
       particleID += blockDim.x*gridDim.x;
#else
        ++particleID;
#endif
    }
    return;
}


} /* end namespace */

#endif
