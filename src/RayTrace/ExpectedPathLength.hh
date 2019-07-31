#ifndef EXPECTEDPATHLENGTH_HH_
#define EXPECTEDPATHLENGTH_HH_

#include <limits>

#include <functional>
#include <memory>

#include "RayListInterface.hh"
#include "MaterialProperties.hh"
#include "MonteRay_timer.hh"
#include "MonteRayMaterialList.hh"
#include "HashLookup.hh"
#include "gpuTally.hh"
#include "Geometry.hh"
#include "RayWorkInfo.hh"
#include "GPUAtomicAdd.hh"
#include "GPUUtilityFunctions.hh"

namespace MonteRay{

class gpuTimingHost;
class MonteRayMaterialListHost;
class HashLookup;
class gpuTallyHost;
class tripleTime;

template <typename MaterialList>
CUDA_CALLABLE_MEMBER
gpuTallyType_t
tallyCellSegment(const MaterialList* pMatList,
        const MaterialProperties* pMatProps,
        const gpuFloatType_t* materialXS,
        gpuTallyType_t* tally,
        unsigned cell,
        gpuRayFloat_t distance,
        gpuFloatType_t energy,
        gpuFloatType_t weight,
        gpuTallyType_t opticalPathLength) {
    using xs_t = gpuTallyType_t;
    using attenuation_t = gpuTallyType_t;
    using score_t = gpuTallyType_t;

    xs_t totalXS = 0.0;
    unsigned numMaterials = pMatProps->numMaterials(cell);
    for( unsigned i=0; i<numMaterials; ++i ) {
      gpuFloatType_t density = pMatProps->getDensity(cell, i);
      if( density > static_cast<xs_t>(1e-5) ) {
        unsigned matID = pMatProps->getMatID(cell, i);
        totalXS +=   materialXS[matID]*density;
      }
    }

    // TPB TODO: cast 1.0 to gpuFloatType_t
    //
    attenuation_t attenuation = static_cast<attenuation_t>(1.0);
    score_t score = distance;
    gpuTallyType_t cellOpticalPathLength = totalXS*distance;

    if( totalXS >  static_cast<xs_t>(1e-5) ) {
        attenuation =  exp( - cellOpticalPathLength );
        score = ( static_cast<xs_t>(1.0) / totalXS ) * ( static_cast<attenuation_t>(1.0) - attenuation );
    }
    score *= exp( -opticalPathLength ) * weight;

    gpu_atomicAdd( &tally[cell], score);

    return cellOpticalPathLength;
}

template<unsigned N, typename GRIDTYPE, typename MaterialList>
CUDA_CALLABLE_MEMBER void
tallyCollision(
        unsigned particleID,
        const GRIDTYPE* pGrid,
        const MaterialList* pMatList,
        const MaterialProperties* pMatProps,
        const HashLookup* pHash,
        const Ray_t<N>* p,
        RayWorkInfo* pRayInfo,
        gpuTallyType_t* pTally
) {
    using enteringFraction_t = gpuTallyType_t;
    gpuTallyType_t opticalPathLength = 0.0;
    gpuFloatType_t energy = p->energy[0];

    unsigned HashBin;
    if( p->particleType == neutron ) {
        HashBin = getHashBin(pHash, energy);
    }

    if( energy < 1e-20 ) {
        return;
    }

    Position_t pos( p->pos[0], p->pos[1], p->pos[2] );
    Direction_t dir( p->dir[0], p->dir[1], p->dir[2] );

    pGrid->rayTrace(particleID, *pRayInfo, pos, dir, 1.0e6f, false);

    gpuFloatType_t materialXS[MAXNUMMATERIALS];
    for( unsigned i=0; i < pMatList->numMaterials; ++i ){
        if( p->particleType == neutron ) {
            materialXS[i] = getTotalXS( pMatList, i, pHash, HashBin, energy, 1.0);
        } else {
            materialXS[i] = getTotalXS( pMatList, i, energy, 1.0);
        }
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

template<unsigned N, typename GRIDTYPE, typename MaterialList>
CUDA_CALLABLE_KERNEL 
rayTraceTally(const GRIDTYPE* pGrid,
        const RayList_t<N>* pCP,
        const MaterialList* pMatList,
        const MaterialProperties* pMatProps,
        const HashLookup* pHash,
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

       MonteRay::tallyCollision<N>(threadID, pGrid, pMatList, pMatProps, pHash, &p, pRayInfo, tally);

#ifdef __CUDACC__
       particleID += blockDim.x*gridDim.x;
#else
        ++particleID;
#endif
    }
    return;
}


template<unsigned N, typename GRIDTYPE, typename MaterialList>
MonteRay::tripleTime launchRayTraceTally(
        std::function<void (void)> cpuWork,
        int nBlocks,
        int nThreads,
        const GRIDTYPE* pGrid,
        const RayListInterface<N>* pCP,
        const MaterialList* pMatList,
        const MaterialProperties* pMatProps,
        gpuTallyHost* pTally
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
            pGrid->getDevicePtr(),
            pCP->getPtrPoints()->devicePtr,
            pMatList->ptr_device,
            pMatProps,
            pMatList->getHashPtr()->getPtrDevice(),
            pRayInfo.get(),
            pTally->temp->tally );

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

    rayTraceTally( pGrid->getPtr(),
            pCP->getPtrPoints(),
            pMatList->getPtr(),
            pMatProps,
            pMatList->getHashPtr()->getPtr(),
            pRayInfo.get(),
            pTally->getPtr()->tally
    );
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


} /* end namespace */

#endif /* EXPECTEDPATHLENGTH_HH_ */
