#ifndef EXPECTEDPATHLENGTH_T_HH_
#define EXPECTEDPATHLENGTH_T_HH_

#include "ExpectedPathLength.hh"

#include "MaterialList.hh"
#include "MonteRay_timer.hh"
#include "gpuTally.hh"
#include "GridBins.hh"
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
    using enteringFraction_t = gpuTallyType_t;
    gpuTallyType_t opticalPathLength = 0.0;
    gpuFloatType_t energy = p->energy[0];

    if( energy < 1e-20 ) {
        return;
    }

    Position_t pos( p->pos[0], p->pos[1], p->pos[2] );
    Direction_t dir( p->dir[0], p->dir[1], p->dir[2] );

    pGeometry->rayTrace(particleID, *pRayInfo, pos, dir, 1.0e6f, false);

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
            pGeometry->getDevicePtr(),
            pCP->getPtrPoints()->devicePtr,
            pMatList,
            pMatProps,
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

    rayTraceTally( pGeometry->getPtr(),
            pCP->getPtrPoints(),
            pMatList,
            pMatProps,
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

#ifdef DEBUG
    const bool debug = false;
#endif

    typedef gpuTallyType_t xs_t;
    typedef gpuTallyType_t attenuation_t;
    typedef gpuTallyType_t score_t;

    xs_t totalXS = 0.0;
    unsigned numMaterials = pMatProps->numMats(cell);

#ifdef DEBUG
    if( debug ) {
        printf("GPU::tallyCellSegment:: cell=%d, numMaterials=%d\n", cell, numMaterials);
    }
#endif

    for( unsigned i=0; i<numMaterials; ++i ) {

        unsigned matID = pMatProps->getMaterialID(cell, i);
        gpuFloatType_t density = pMatProps->getDensity(cell, i );
        if( density > 1e-5 ) {
            totalXS +=   materialXS[matID]*density;
        }
        //    if( debug ) {
        //      printf("GPU::tallyCellSegment::       material=%d, density=%f, xs=%f, totalxs=%f\n", i, density, xs, totalXS);
        //    }
    }

    attenuation_t attenuation = 1.0;
    score_t score = distance;
    gpuTallyType_t cellOpticalPathLength = totalXS*distance;

    if( totalXS >  1e-5 ) {
        attenuation =  exp( - cellOpticalPathLength );
        score = ( 1.0 / totalXS ) * ( 1.0 - attenuation );
    }
    score *= exp( -opticalPathLength ) * weight;

    gpu_atomicAdd( &tally[cell], score);

#ifdef DEBUG
    if( debug ) {
        printf("GPU::tallyCellSegment:: total score=%f\n", tally[cell] );
    }
#endif

    return cellOpticalPathLength;
}

} /* end namespace */

#endif
