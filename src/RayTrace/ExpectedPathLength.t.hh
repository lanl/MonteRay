#ifndef EXPECTEDPATHLENGTH_T_HH_
#define EXPECTEDPATHLENGTH_T_HH_

#include <limits>

#include "ExpectedPathLength.hh"

#include "MonteRay_MaterialProperties.hh"
#include "MonteRay_timer.hh"
#include "MonteRayMaterialList.hh"
#include "HashLookup.hh"
#include "gpuTally.hh"
#include "Geometry.hh"
#include "RayWorkInfo.hh"
#include "GPUAtomicAdd.hh"
#include "GPUUtilityFunctions.hh"

namespace MonteRay{

template <typename MaterialList>
CUDA_CALLABLE_MEMBER
gpuTallyType_t
tallyCellSegment( const MaterialList* pMatList,
        const MonteRay_MaterialProperties_Data* pMatProps,
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
    unsigned numMaterials = getNumMats( pMatProps, cell);

#ifdef DEBUG
    if( debug ) {
        printf("GPU::tallyCellSegment:: cell=%d, numMaterials=%d\n", cell, numMaterials);
    }
#endif

    for( unsigned i=0; i<numMaterials; ++i ) {

        unsigned matID = getMatID(pMatProps, cell, i);
        gpuFloatType_t density = getDensity(pMatProps, cell, i );
        if( density > 1e-5 ) {
            totalXS +=   materialXS[matID]*density;
        }
        //		if( debug ) {
        //			printf("GPU::tallyCellSegment::       material=%d, density=%f, xs=%f, totalxs=%f\n", i, density, xs, totalXS);
        //		}
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

template<unsigned N, typename GRIDTYPE, typename MaterialList>
CUDA_CALLABLE_MEMBER void
tallyCollision(
        unsigned particleID,
        const GRIDTYPE* pGrid,
        const MaterialList* pMatList,
        const MonteRay_MaterialProperties_Data* pMatProps,
        const HashLookup* pHash,
        const Ray_t<N>* p,
        RayWorkInfo* pRayInfo,
        gpuTallyType_t* pTally )
{
#ifdef DEBUG
    const bool debug = false;

    if( debug ) {
        printf("--------------------------------------------------------------------------------------------------------\n");
        printf("GPU::tallyCollision:: nCollisions=%d, x=%f, y=%f, z=%f, u=%f, v=%f, w=%f, energy=%f, weight=%f, index=%d \n",
                particleID+1,
                p->pos[0],
                p->pos[1],
                p->pos[2],
                p->dir[0],
                p->dir[1],
                p->dir[2],
                p->energy[0],
                p->weight[0],
                p->index
        );
    }
#endif

    typedef gpuTallyType_t enteringFraction_t;

    gpuTallyType_t opticalPathLength = 0.0;

    gpuFloatType_t energy = p->energy[0];

    unsigned HashBin;
    if( p->particleType == neutron ) {
        HashBin = getHashBin(pHash, energy);
    }

    if( energy < 1e-20 ) {
        return;
    }


    //	gpuRayFloat_t pos = make_float3( p->pos[0], p->pos[1], p->pos[2]);
    //	gpuRayFloat_t dir = make_float3( p->dir[0], p->dir[1], p->dir[2]);
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
CUDA_CALLABLE_KERNEL  rayTraceTally(
        const GRIDTYPE* pGrid,
        const RayList_t<N>* pCP,
        const MaterialList* pMatList,
        const MonteRay_MaterialProperties_Data* pMatProps,
        const HashLookup* pHash,
        RayWorkInfo* pRayInfo,
        gpuTallyType_t* tally){

#ifdef DEBUG
    const bool debug = false;
#endif

#ifdef __CUDACC__
    int threadID = threadIdx.x + blockIdx.x*blockDim.x;
#else
    int threadID = 0;
#endif
    int particleID = threadID;

    int num = pCP->size();

#ifdef DEBUG
    if( debug ) printf("GPU::rayTraceTally:: starting threadID=%d  N=%d\n", threadID, N );
#endif

    while( particleID < num ) {
        Ray_t<N> p = pCP->getParticle(particleID);
        pRayInfo->clear( threadID );

#ifdef DEBUG
        if( debug ) {
            printf("--------------------------------------------------------------------------------------------------------\n");
            printf("GPU::rayTraceTally:: threadID=%d\n", threadID );
            printf("GPU::rayTraceTally:: x=%f\n", p.pos[0] );
            printf("GPU::rayTraceTally:: y=%f\n", p.pos[1] );
            printf("GPU::rayTraceTally:: z=%f\n", p.pos[2] );
            printf("GPU::rayTraceTally:: u=%f\n", p.dir[0] );
            printf("GPU::rayTraceTally:: v=%f\n", p.dir[1] );
            printf("GPU::rayTraceTally:: w=%f\n", p.dir[2] );
            printf("GPU::rayTraceTally:: energy=%f\n", p.energy[0] );
            printf("GPU::rayTraceTally:: weight=%f\n", p.weight[0] );
            printf("GPU::rayTraceTally:: index=%d\n", p.index );
        }
#endif

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
        const MonteRay_MaterialProperties* pMatProps,
        gpuTallyHost* pTally
)
{
    MonteRay::tripleTime time;

    auto launchParams = setLaunchBounds( nThreads, nBlocks, pCP->getPtrPoints()->size() );
    nBlocks = launchParams.first;
    nThreads = launchParams.second;

#ifdef __CUDACC__
    RayWorkInfo rayInfo(nBlocks*nThreads);
    rayInfo.copyToGPU();

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
            pMatProps->ptrData_device,
            pMatList->getHashPtr()->getPtrDevice(),
            rayInfo.devicePtr,
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
    RayWorkInfo rayInfo(1, true);

    MonteRay::cpuTimer timer1, timer2;
    timer1.start();

    rayTraceTally( pGrid->getPtr(),
            pCP->getPtrPoints(),
            pMatList->getPtr(),
            pMatProps->getPtr(),
            pMatList->getHashPtr()->getPtr(),
            &rayInfo,
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

#endif /* EXPECTEDPATHLENGTH_T_HH_ */
