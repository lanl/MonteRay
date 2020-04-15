#ifndef MONTERAYNEXTEVENTESTIMATOR_T_HH_
#define MONTERAYNEXTEVENTESTIMATOR_T_HH_

#include "NextEventEstimator.hh"
#include "MonteRayParallelAssistant.hh"
#include "GPUUtilityFunctions.hh"
#include "StreamAndEvent.hh"

namespace MonteRay{
// TPB: consider breaking this up into multiple functions.
template<unsigned N, typename Geometry, typename MaterialProperties, typename MaterialList>
CUDA_CALLABLE_MEMBER
void NextEventEstimator::calcScore( const int threadID, const Ray_t<N>& ray, RayWorkInfo& rayInfo, 
                               const Geometry& geometry, const MaterialProperties& matProps, 
                               const MaterialList& matList){
  // Neutrons are not yet supported 
  // TPB: but this algorithm doesn't care what particle we're ray tracing.  This should be a filter somewhere.
  if( ray.particleType == neutron ) {
#ifndef NDEBUG
    printf("MonteRay::NextEventEstimator::calcScore - neutrons not supported.\n");
#endif
    return;
  }

  MONTERAY_ASSERT( ray.detectorIndex < tallyPoints_.size());

  auto distAndDir = getDistanceDirection(ray.pos, tallyPoints_[ray.detectorIndex]);
  auto& dist = std::get<0>(distAndDir);
  auto& dir = std::get<1>(distAndDir);


  geometry.rayTrace( threadID, rayInfo, ray.pos, dir, dist, false);

  for( int energyIndex=0; energyIndex < N; ++energyIndex) {
    gpuFloatType_t weight = ray.weight[energyIndex];
    if( weight == 0.0 ) continue;

    gpuFloatType_t energy = ray.energy[energyIndex];
    if( energy <  1.0e-11 ) continue;

    gpuFloatType_t materialXS[MAXNUMMATERIALS]; // TPB TODO: investigate this - could be replaced by something like rayWorkInfo or done on-the-fly.
    // TPB: this could be removed entirely and done in a different kernel, then this function could take a list of XS
    // this could relieve register pressure, potentially.  Reduce monolithic kernels into smaller ones.  Also better for determining kernel timing.
    for( int i=0; i < matList.numMaterials(); ++i ){
      materialXS[i] = matList.material(i).getTotalXS(energy, 1.0);
    }

    TallyFloat opticalThickness = 0.0;
    for( int i=0; i < rayInfo.getRayCastSize( threadID ); ++i ){
      int cell = rayInfo.getRayCastCell( threadID, i);
      gpuRayFloat_t cellDistance = rayInfo.getRayCastDist( threadID, i);
      if( cell == std::numeric_limits<unsigned int>::max() ) continue; // TPB: this is geometry-dependent

      gpuFloatType_t totalXS = 0.0;
      int numMaterials = matProps.numMats(cell);

      for( int matIndex=0; matIndex<numMaterials; ++matIndex ) {
        int matID = matProps.getMaterialID(cell, matIndex);
        gpuFloatType_t density = matProps.getDensity(cell, matIndex);
        gpuFloatType_t xs = materialXS[matID]*density;
        totalXS += xs;
      }
      opticalThickness += totalXS * cellDistance;
    }
    this->score( ( weight / (2.0 * MonteRay::pi * dist*dist)  ) * exp( - opticalThickness),
      ray.detectorIndex, ray.energy[energyIndex], ray.time + dist / ray.speed() );
  }
}

#ifdef __CUDACC__
template<unsigned N, typename Geometry, typename MaterialProperties, typename MaterialList>
CUDA_CALLABLE_KERNEL  kernel_ScoreRayList(NextEventEstimator* ptr, const RayList_t<N>* pRayList, RayWorkInfo* pRayInfo, 
      const Geometry* const pGeometry, const MaterialProperties* const pMatProps, const MaterialList* const pMatList){

  int threadID = threadIdx.x + blockIdx.x*blockDim.x;
  int particleID = threadID;
  int num = pRayList->size();
  while( particleID < num ) {
    pRayInfo->clear( threadID );
    auto& ray = pRayList->points[particleID];
    ptr->calcScore(threadID, ray, *pRayInfo, *pGeometry, *pMatProps, *pMatList);
    particleID += blockDim.x*gridDim.x;
  }
}
#endif

template<unsigned N, typename Geometry, typename MaterialProperties, typename MaterialList>
void launch_ScoreRayList( NextEventEstimator* const pNextEventEstimator, int nBlocksArg, int nThreadsArg, const RayList_t<N>* pRayList, 
  RayWorkInfo* pRayInfo, const Geometry* const pGeometry, const MaterialProperties* const pMatProps, 
  const MaterialList* const pMatList, const cuda::StreamPointer& pStream = {}){
  // negative nBlocks and nThreads forces to specified value,
  // otherwise reasonable values are used based on the specified ones

  const MonteRayParallelAssistant& PA( MonteRayParallelAssistant::getInstance() );
  if( PA.getWorkGroupRank() != 0 ) return;
  
#ifdef __CUDACC__
  auto launchBounds = setLaunchBounds( nThreadsArg, nBlocksArg, pRayList->size() );
  int nBlocks = launchBounds.first;
  int nThreads = launchBounds.second;
  kernel_ScoreRayList<<<nBlocks, nThreads, 0, *pStream>>>( pNextEventEstimator, pRayList, pRayInfo, 
      pGeometry, pMatProps, pMatList );
#else
  cpuScoreRayList( pNextEventEstimator, pRayList, pRayInfo, pGeometry, pMatProps, pMatList );
#endif
}

#endif
} // end namespace MonteRay
