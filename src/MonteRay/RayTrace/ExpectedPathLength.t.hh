#ifndef EXPECTEDPATHLENGTH_T_HH_
#define EXPECTEDPATHLENGTH_T_HH_

#include "ExpectedPathLength.hh"
#include <algorithm>

#ifdef __CUDACC__
#include <thrust/for_each.h>
#endif

#include "MaterialList.hh"
#include "MonteRay_timer.hh"
#include "BasicTally.hh"
#include "RayWorkInfo.hh"
#include "GPUAtomicAdd.hh"
#include "GPUUtilityFunctions.hh"

namespace MonteRay{

template<unsigned N, typename Geometry, typename MaterialList>
CUDA_CALLABLE_MEMBER void
tallyCollision(
        unsigned particleID,
        const Geometry* pGeometry,
        const MaterialList* pMatList,
        const MaterialProperties* pMatProps,
        const Ray_t<N>* p,
        RayWorkInfo* pRayInfo,
        gpuTallyType_t* pTally
) {
    gpuTallyType_t opticalPathLength = 0.0;
    gpuFloatType_t energy = p->energy[0];

    if( energy < 1e-20 ) {
        return;
    }

    Position_t pos( p->pos[0], p->pos[1], p->pos[2] );
    Direction_t dir( p->dir[0], p->dir[1], p->dir[2] );

    pGeometry->rayTrace(particleID, *pRayInfo, pos, dir, static_cast<gpuRayFloat_t>(1.0e6), false);

    gpuFloatType_t materialXS[MAXNUMMATERIALS];
    for( unsigned i=0; i < pMatList->numMaterials(); ++i ){
      materialXS[i] = pMatList->material(i).getTotalXS(energy, 1.0);
    }

    for( unsigned i=0; i < pRayInfo->getRayCastSize(particleID); ++i ){
        int cell = pRayInfo->getRayCastCell(particleID,i);
        gpuRayFloat_t distance = pRayInfo->getRayCastDist(particleID,i);
        if( cell == std::numeric_limits<unsigned>::max() ) continue;

        opticalPathLength += tallyCellSegment(pMatList, pMatProps, materialXS, pTally,
                cell, distance, energy, p->weight[0], opticalPathLength);

        if( opticalPathLength > 5.0 ) {
            // cut off at 5 mean free paths
            return;
        }
    }
}

template<unsigned N, typename Geometry, typename MaterialList>
CUDA_CALLABLE_KERNEL 
rayTraceTally(const Geometry* pGeometry,
        const RayList_t<N>* pCP,
        const MaterialList* pMatList,
        const MaterialProperties* pMatProps,
        RayWorkInfo* pRayInfo,
        gpuTallyType_t* tally) {
#ifdef __CUDACC__
    int threadID = threadIdx.x + blockIdx.x*blockDim.x;
#else
    int threadID = 0;
#endif
    int particleID = threadID;

    int num = pCP->size();


    while( particleID < num ) {
       Ray_t<N> p = pCP->getParticle(particleID);
       pRayInfo->clear( threadID );

       MonteRay::tallyCollision<N>(threadID, pGeometry, pMatList, pMatProps, &p, pRayInfo, tally);

#ifdef __CUDACC__
       particleID += blockDim.x*gridDim.x;
#else
        ++particleID;
#endif
    }
    return;
}


template<unsigned N, typename Geometry, typename MaterialList>
MonteRay::tripleTime launchRayTraceTally(
        std::function<void (void)> cpuWork,
        int nBlocks,
        int nThreads,
        const Geometry* pGeometry,
        const RayListInterface<N>* pCP,
        const MaterialList* pMatList,
        const MaterialProperties* pMatProps,
        BasicTally* const pTally
) {
    MonteRay::tripleTime time;

    auto launchParams = setLaunchBounds( nThreads, nBlocks, pCP->getPtrPoints()->size() );
    nBlocks = launchParams.first;
    nThreads = launchParams.second;

#ifdef __CUDACC__
    auto pRayInfo = std::make_unique<RayWorkInfo>(nBlocks*nThreads);

    cudaEvent_t startGPU, stopGPU, start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);

    cudaStream_t stream;
    cudaStreamCreate( &stream );

    cudaEventRecord(start,0);
    cudaEventRecord(startGPU,stream);

    rayTraceTally<<<nBlocks,nThreads,0,stream>>>(
            pGeometry,
            pCP->getPtrPoints(),
            pMatList,
            pMatProps,
            pRayInfo.get(),
            pTally->data() );

    cudaEventRecord(stopGPU,stream);
    cudaStreamWaitEvent(stream, stopGPU, 0);

    {
        MonteRay::cpuTimer timer;
        timer.start();
        cpuWork();
        timer.stop();
        time.cpuTime = timer.getTime();
    }

    cudaStreamSynchronize( stream );
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaStreamDestroy(stream);

    float_t gpuTime;
    cudaEventElapsedTime(&gpuTime, startGPU, stopGPU );
    time.gpuTime = gpuTime / 1000.0;

    float_t totalTime;
    cudaEventElapsedTime(&totalTime, start, stop );
    time.totalTime = totalTime/1000.0;
#else
    auto pRayInfo = std::make_unique<RayWorkInfo>(1);

    MonteRay::cpuTimer timer1, timer2;
    timer1.start();

    rayTraceTally( pGeometry,
            pCP->getPtrPoints(),
            pMatList,
            pMatProps,
            pRayInfo.get(),
            pTally->data() );
    timer1.stop();
    timer2.start();
    cpuWork();
    timer2.stop();

    time.gpuTime = timer1.getTime();
    time.cpuTime = timer2.getTime();
    time.totalTime = timer1.getTime() + timer2.getTime();
#endif

    return time;
}

CUDA_CALLABLE_MEMBER
gpuTallyType_t
inline tallyCellSegment( const MaterialList* pMatList,
        const MaterialProperties* pMatProps,
        const gpuFloatType_t* materialXS,
        gpuTallyType_t* tally,
        unsigned cell,
        gpuRayFloat_t distance,
        gpuFloatType_t energy,
        gpuFloatType_t weight,
        gpuTallyType_t opticalPathLength ) {

    typedef gpuTallyType_t xs_t;
    typedef gpuTallyType_t attenuation_t;
    typedef gpuTallyType_t score_t;

    xs_t totalXS = 0.0;
    unsigned numMaterials = pMatProps->numMats(cell);

    for( unsigned i=0; i<numMaterials; ++i ) {

        unsigned matID = pMatProps->getMaterialID(cell, i);
        gpuFloatType_t density = pMatProps->getDensity(cell, i );
        if( density > 1e-5 ) {
            totalXS +=   materialXS[matID]*density;
        }
    }

    attenuation_t attenuation = 1.0;
    score_t score = distance;
    gpuTallyType_t cellOpticalPathLength = totalXS*distance;

    if( totalXS >  1e-5 ) {
        attenuation =  exp( - cellOpticalPathLength ); // TPB: this is a double-precision exponent and this math is done in double precision
        score = ( 1.0 / totalXS ) * ( 1.0 - attenuation );
    }
    score *= exp( -opticalPathLength ) * weight;

    gpu_atomicAdd( &tally[cell], score);

    return cellOpticalPathLength;
}

template<unsigned N, typename Geometry, typename MaterialList>
CUDA_CALLABLE_MEMBER void rayTraceOnGridWithMovingMaterials( 
          Ray_t<N> ray,
          gpuRayFloat_t timeRemaining,
          const Geometry& geometry,
          const MaterialProperties& matProps,
          const MaterialList& matList,
          gpuTallyType_t* const tally) {

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

    gpu_atomicAdd(&tally[cellID], score);
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
void rayTraceTallyWithMovingMaterials(
          const RayList_t<N>* const collisionPoints,
          gpuFloatType_t timeRemaining,
          const Geometry* const geometry,
          const MaterialProperties* const matProps,
          const MaterialList* const matList,
          gpuTallyType_t* const tally,
          cudaStream_t* stream = nullptr) {

#ifdef __CUDACC__
  thrust::for_each(thrust::cuda::par.on(*stream), collisionPoints->begin(), collisionPoints->end(),
    [=] __device__ (const auto& ray){
      rayTraceOnGridWithMovingMaterials(ray, timeRemaining, *geometry, *matProps, *matList, tally);
    }
  );
#else 
  std::for_each(collisionPoints->begin(), collisionPoints->end(),
    [=] (const auto& ray){
      rayTraceOnGridWithMovingMaterials(ray, timeRemaining, *geometry, *matProps, *matList, tally);
    }
  );
#endif
}

} /* end namespace */

#endif
