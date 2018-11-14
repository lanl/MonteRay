#ifndef MONTERAYNEXTEVENTESTIMATOR_T_HH_
#define MONTERAYNEXTEVENTESTIMATOR_T_HH_

#include "MonteRayNextEventEstimator.hh"

#include "MonteRayDefinitions.hh"
#include "MonteRayCopyMemory.t.hh"
#include "MonteRay_SpatialGrid.hh"
#include "HashLookup.hh"
#include "GPUAtomicAdd.hh"
#include "GPUErrorCheck.hh"
#include "GridBins.hh"
#include "MonteRay_MaterialProperties.hh"
#include "MonteRayMaterialList.hh"
#include "ExpectedPathLength.hh"
#include "RayList.hh"
#include "MonteRayTally.hh"
#include "MonteRayParallelAssistant.hh"

namespace MonteRay {

template<typename GRID_T>
CUDAHOST_CALLABLE_MEMBER
MonteRayNextEventEstimator<GRID_T>::MonteRayNextEventEstimator(unsigned num) {
    if( num == 0 ) { num = 1; }
    if( Base::debug ) {
        std::cout << "RayList_t::RayList_t(n), n=" << num << " \n";
    }
    init();
    x = (position_t*) MONTERAYHOSTALLOC( num*sizeof( position_t ), Base::isManagedMemory, "x" );
    y = (position_t*) MONTERAYHOSTALLOC( num*sizeof( position_t ), Base::isManagedMemory, "y" );
    z = (position_t*) MONTERAYHOSTALLOC( num*sizeof( position_t ), Base::isManagedMemory, "z" );
    nAllocated = num;
}

template<typename GRID_T>
CUDAHOST_CALLABLE_MEMBER
MonteRayNextEventEstimator<GRID_T>::~MonteRayNextEventEstimator(){
    if( Base::isCudaIntermediate ) {
        if( x != NULL )     { MonteRayDeviceFree( x ); }
        if( y != NULL )     { MonteRayDeviceFree( y ); }
        if( z != NULL )     { MonteRayDeviceFree( z ); }
    } else {
        if( x != NULL )     { MonteRayHostFree( x, Base::isManagedMemory ); }
        if( y != NULL )     { MonteRayHostFree( y, Base::isManagedMemory ); }
        if( z != NULL )     { MonteRayHostFree( z, Base::isManagedMemory ); }
        if( pTally != NULL ) { delete pTally; }
        if( pTallyTimeBinEdges != NULL ) { delete pTallyTimeBinEdges; }
    }
}

template<typename GRID_T>
CUDAHOST_CALLABLE_MEMBER
void
MonteRayNextEventEstimator<GRID_T>::init() {
     nUsed = 0;
     nAllocated = 0;
     radius = 0.0;

     x = NULL;
     y = NULL;
     z = NULL;
     pTally = NULL;
     pTallyTimeBinEdges = NULL;

     pGridBins = NULL;
     pMatPropsHost = NULL;
     pMatProps = NULL;
     pMatListHost = NULL;
     pMatList = NULL;
     pHashHost = NULL;
     pHash = NULL;

     initialized = false;
     copiedToGPU = false;
 }

template<typename GRID_T>
void MonteRayNextEventEstimator<GRID_T>::initialize() {
    if( initialized ) return;
    //printf( "Debug: MonteRayNextEventEstimator<GRID_T>::initialize -- nUsed = %d\n ", nUsed );

    pTally = new MonteRayTally(nUsed);
    if( pTallyTimeBinEdges) { pTally->setTimeBinEdges( *pTallyTimeBinEdges ); }
    pTally->initialize();

    initialized = true;
}

template<typename GRID_T>
CUDAHOST_CALLABLE_MEMBER
unsigned
MonteRayNextEventEstimator<GRID_T>::add( position_t xarg, position_t yarg, position_t zarg) {

     MONTERAY_VERIFY( nUsed < nAllocated, "MonteRayNextEventEstimator::add -- Detector list is full.  Can't add any more detectors." );

     x[nUsed] = xarg;
     y[nUsed] = yarg;
     z[nUsed] = zarg;
     return nUsed++;
 }

template<typename GRID_T>
CUDAHOST_CALLABLE_MEMBER
void
MonteRayNextEventEstimator<GRID_T>::copy(const MonteRayNextEventEstimator* rhs) {
    if(  MonteRayParallelAssistant::getInstance().getWorkGroupRank() != 0 ) return;

    if( Base::isCudaIntermediate  and !rhs->initialized ) {
        throw std::runtime_error("MonteRayNextEventEstimator<GRID_T>::copy -- not initialized"  );
    }

#ifdef __CUDACC__
    if( Base::debug ) {
        std::cout << "Debug: MonteRayNextEventEstimator::copy (const MonteRayNextEventEstimator* rhs) \n";
    }

    if( Base::isCudaIntermediate && rhs->isCudaIntermediate ) {
        throw std::runtime_error(" MonteRayNextEventEstimator::copy -- can NOT copy CUDA intermediate to CUDA intermediate.");
    }

    if( !Base::isCudaIntermediate && !rhs->isCudaIntermediate ) {
        throw std::runtime_error(" MonteRayNextEventEstimator::copy -- can NOT copy CUDA non-intermediate to CUDA non-intermediate.");
    }

    if( nAllocated > 0 && nAllocated != rhs->nAllocated) {
        throw std::runtime_error(" MonteRayNextEventEstimator::copy -- can NOT change the number of allocated tally points.");
    }

    unsigned num = rhs->nAllocated;
    if( Base::isCudaIntermediate ) {
        // target is the intermediate, origin is the host
        if( x == NULL ) {
            x = (position_t*) MONTERAYDEVICEALLOC( num*sizeof( position_t ), "x" );
        }
        if( y == NULL ) {
            y = (position_t*) MONTERAYDEVICEALLOC( num*sizeof( position_t ), "y" );
        }
        if( z == NULL ) {
            z = (position_t*) MONTERAYDEVICEALLOC( num*sizeof( position_t ), "z" );
        }
//        if( tally == NULL ) {
//            tally = (tally_t*) MONTERAYDEVICEALLOC( num*sizeof( tally_t ), "tally" );
//        }

        MonteRayMemcpy(x, rhs->x, num*sizeof(position_t), cudaMemcpyHostToDevice);
        MonteRayMemcpy(y, rhs->y, num*sizeof(position_t), cudaMemcpyHostToDevice);
        MonteRayMemcpy(z, rhs->z, num*sizeof(position_t), cudaMemcpyHostToDevice);
        //MonteRayMemcpy( tally, rhs->tally, num*sizeof(tally_t), cudaMemcpyHostToDevice);

        pMatPropsHost = NULL;
        pMatListHost = NULL;
        pHashHost = NULL;

        pGridBins = rhs->pGridBins->getDevicePtr();
        pMatProps = rhs->pMatPropsHost->ptrData_device;
        pMatList = rhs->pMatListHost->ptr_device;
        pHash = rhs->pMatListHost->getHashPtr()->getPtrDevice();
        pTally = rhs->pTally->devicePtr;

        nAllocated = rhs->nAllocated;
        nUsed = rhs->nUsed;
        radius = rhs->radius;
        initialized = rhs->initialized;
        copiedToGPU = rhs->copiedToGPU;

    } else {
        // target is the host, origin is the intermediate

//        if( Base::debug ) std::cout << "Debug: MonteRayNextEventEstimator::copy - copying tally from device to host\n";
//        MonteRayMemcpy( tally, rhs->tally, num*sizeof(tally_t), cudaMemcpyDeviceToHost);
//        if( Base::debug ) std::cout << "Debug: MonteRayNextEventEstimator::copy - DONE copying tally from device to host\n";
    }



    if( Base::debug ) {
        std::cout << "Debug: MonteRayNextEventEstimator::copy -- exiting." << std::endl;
    }
#else

#endif
}

template<typename GRID_T>
CUDAHOST_CALLABLE_MEMBER
void
MonteRayNextEventEstimator<GRID_T>::copyToGPU(){
    if( MonteRayParallelAssistant::getInstance().getWorkGroupRank() != 0 ) return;

    copiedToGPU = true;
    pTally->copyToGPU();
    Base::copyToGPU();
}

template<typename GRID_T>
CUDAHOST_CALLABLE_MEMBER
void
MonteRayNextEventEstimator<GRID_T>::copyToCPU(){
    if( !copiedToGPU ) return;

    if( MonteRayParallelAssistant::getInstance().getWorkGroupRank() != 0 ) return;

    pTally->copyToCPU();
    Base::copyToCPU();

}

template<typename GRID_T>
CUDA_CALLABLE_MEMBER
typename MonteRayNextEventEstimator<GRID_T>::tally_t
MonteRayNextEventEstimator<GRID_T>::getTally(unsigned spatialIndex, unsigned timeIndex) const {
    return pTally->getTally(spatialIndex, timeIndex);
}


template<typename GRID_T>
CUDA_CALLABLE_MEMBER
typename MonteRayNextEventEstimator<GRID_T>::position_t
MonteRayNextEventEstimator<GRID_T>::distance(unsigned i, MonteRay::Vector3D<gpuRayFloat_t>& pos ) const {
    using namespace std;
    MONTERAY_ASSERT(i<nUsed);
    MonteRay::Vector3D<gpuRayFloat_t> pos2( getX(i), getY(i), getZ(i) );
    MonteRay::Vector3D<gpuRayFloat_t> dir = pos2 - pos;

    return dir.magnitude();
}

template<typename GRID_T>
CUDA_CALLABLE_MEMBER
typename MonteRayNextEventEstimator<GRID_T>::position_t
MonteRayNextEventEstimator<GRID_T>::getDistanceDirection(
        unsigned i,
        MonteRay::Vector3D<gpuRayFloat_t>& pos,
        MonteRay::Vector3D<gpuRayFloat_t>& dir ) const {
    using namespace std;
    MONTERAY_ASSERT(i<nUsed);

    MonteRay::Vector3D<gpuRayFloat_t> pos2( getX(i), getY(i), getZ(i) );
    dir = pos2 - pos;

    position_t dist = distance(i,pos);
    position_t invDistance = 1/ dist;
    dir *= invDistance;

    return dist;
}

template<typename GRID_T>
template<unsigned N>
CUDA_CALLABLE_MEMBER
typename MonteRayNextEventEstimator<GRID_T>::tally_t
MonteRayNextEventEstimator<GRID_T>::calcScore( Ray_t<N>& ray ) {

    const bool debug = false;

    if( ray.detectorIndex >= nUsed ) {
        printf("Debug: MonteRayNextEventEstimator::calcScore -- ray.detectorIndex < nUsed,  ray.detectorIndex = %d, nUsed  = %d\n", ray.detectorIndex, nUsed );
    }
    MONTERAY_ASSERT( ray.detectorIndex < nUsed);
    tally_t score = 0.0;

    int cells[2*MAXNUMVERTICES];
    gpuRayFloat_t crossingDistances[2*MAXNUMVERTICES];

    unsigned numberOfCells;

    MonteRay::Vector3D<gpuRayFloat_t> pos( ray.pos[0], ray.pos[1], ray.pos[2]);
    MonteRay::Vector3D<gpuRayFloat_t> dir( ray.dir[0], ray.dir[1], ray.dir[2]);

    gpuRayFloat_t dist = getDistanceDirection(
            ray.detectorIndex,
            pos,
            dir );
    if( debug ) printf("Debug: MonteRayNextEventEstimator::calcScore -- distance to detector = %20.12f\n",dist );

    gpuFloatType_t time = ray.time + dist / ray.speed();

    //      float3_t pos = make_float3( x, y, z);
    //      float3_t dir = make_float3( u, v, w);

    numberOfCells = pGridBins->rayTrace( cells, crossingDistances, pos, dir, dist, false);

    for( unsigned energyIndex=0; energyIndex < N; ++energyIndex) {
        gpuFloatType_t weight = ray.weight[energyIndex];
        if( weight == 0.0 ) continue;

        gpuFloatType_t energy = ray.energy[energyIndex];
        if( energy <  1.0e-11 ) continue;

        tally_t partialScore = 0.0;

        if( debug ) {
            printf("Debug: MonteRayNextEventEstimator::calcScore -- energyIndex=%d, energy=%f, weight=%f\n", energyIndex, energy,  weight);
        }
        gpuFloatType_t materialXS[MAXNUMMATERIALS];
        for( unsigned i=0; i < pMatList->numMaterials; ++i ){
            if( debug ) printf("Debug: MonteRayNextEventEstimator::calcScore -- materialIndex=%d\n", i);
            materialXS[i] = getTotalXS( pMatList, i, energy, 1.0);
            if( debug ) {
                printf("Debug: MonteRayNextEventEstimator::calcScore -- materialIndex=%d, materialXS=%f\n", i, materialXS[i]);
            }
        }

        tally_t opticalThickness = 0.0;
        for( unsigned i=0; i < numberOfCells; ++i ){
            int cell = cells[i];
            gpuRayFloat_t cellDistance = crossingDistances[i];
            if( cell == UINT_MAX ) continue;

            gpuFloatType_t totalXS = 0.0;
            unsigned numMaterials = getNumMats( pMatProps, cell);
            if( debug ) {
                printf("Debug: MonteRayNextEventEstimator::calcScore -- cell=%d, cellDistance=%f, numMaterials=%d\n", cell, cellDistance, numMaterials);
            }
            for( unsigned matIndex=0; matIndex<numMaterials; ++matIndex ) {
                unsigned matID = getMatID(pMatProps, cell, matIndex);
                gpuFloatType_t density = getDensity(pMatProps, cell, matIndex );
                gpuFloatType_t xs = materialXS[matID]*density;
                totalXS += xs;
                if( debug ) {
                    printf("Debug: MonteRayNextEventEstimator::calcScore -- materialID=%d, density=%f, materialXS=%f, xs=%f, totalxs=%f\n", matID, density, materialXS[matID], xs, totalXS);
                }
            }

            opticalThickness += totalXS * cellDistance;
            if( debug ) printf("Debug: MonteRayNextEventEstimator::calcScore -- optialThickness= %20.12f\n",opticalThickness );
        }

        partialScore = ( weight / (2.0 * MonteRay::pi * dist*dist)  ) * exp( - opticalThickness);
        score += partialScore;
    }

    if( debug ) {
        printf("Debug: MonteRayNextEventEstimator::calcScore -- value=%e .\n" , score);
    }
//    gpu_atomicAdd( &tally[ray.detectorIndex], score );
    pTally->score(score, ray.detectorIndex, time);

    return score;
}

template<typename GRID_T>
template<unsigned N>
CUDA_CALLABLE_MEMBER
void
MonteRayNextEventEstimator<GRID_T>::score( const RayList_t<N>* pRayList, unsigned tid ) {
    const bool debug = false;

    if( debug ) {
        printf("Debug: MonteRayNextEventEstimator::score -- tid=%d, particle=%d .\n",
                tid,  pRayList->points[tid].particleType);
    }

    // Neutrons are not yet supported
    if( pRayList->points[tid].particleType == neutron ) return;

    tally_t value = calcScore<N>( pRayList->points[tid] );
}

template<typename GRID_T>
CUDAHOST_CALLABLE_MEMBER
void
MonteRayNextEventEstimator<GRID_T>::printPointDets( const std::string& outputFile, unsigned nSamples, unsigned constantDimension) {
    if( MonteRayParallelAssistant::getInstance().getWorkGroupRank() != 0 ) return;

    if( nUsed == 0 ) {
        return;
    }

    std::ofstream out;
    out.open(outputFile.c_str(), std::ios::out );
    if( ! out.is_open() ) {
        throw std::runtime_error( "Failure opening output file.  File= " + outputFile );
    }
    outputTimeBinnedTotal( out, nSamples, constantDimension);
    out.close();
}

template<typename GRID_T>
CUDAHOST_CALLABLE_MEMBER
void
MonteRayNextEventEstimator<GRID_T>::outputTimeBinnedTotal(std::ostream& out,unsigned nSamples, unsigned constantDimension){
    out << "#  MonteRay results                                          \n";
    out << "#                                                            \n";
    out << "#                                      Score        Score    \n";
    out << "#        X          Y          Z       Average      Rel Err. \n";
    out << "# ________   ________   ________   ___________   ___________ \n";
    //        12345678   12345678   12345678   12345678901   12345678901
    //     "12        123        123        123           123
    //boost::format fmt( "  %1$8.3f   %2$8.3f   %3$8.3f   %4$11.4e   %5$11.4e\n" );

    gpuFloatType_t sum = 0.0;
    gpuFloatType_t min = std::numeric_limits<double>::infinity();
    gpuFloatType_t max = -std::numeric_limits<double>::infinity();

    // dim2 used to insert new-line when it decreases, indicating a new row.
    unsigned dim2;
    switch (constantDimension) {
    case 0:
        dim2 = 2; // z
        break;
    case 1:
        dim2 = 2; // z
        break;
    case 2:
        dim2 = 1; // y
        break;
    default:
        break;
    }

    Vector3D<gpuFloatType_t> pos = getPoint(0);

    // previousSecondDimPosition used to detect when to insert carriage return
    double previousSecondDimPosition = pos[dim2];

    for( unsigned i=0; i < nUsed; ++i ) {
        Vector3D<gpuFloatType_t> pos = getPoint(i);
        gpuFloatType_t value = getTally(i) / nSamples;

        if(  pos[dim2] < previousSecondDimPosition ) {
            out << "\n";
        }

        char buffer[200];
        snprintf( buffer, 200, "   %8.3f   %8.3f   %8.3f   %11.4e   %511.4e\n",
                                  pos[0], pos[1], pos[2],  value,       0.0 );
        out << buffer;

        previousSecondDimPosition = pos[dim2];

        if( value < min ) min = value;
        if( value > max ) max = value;
        sum += value;
    }
    out << "\n#\n";
    out << "# Min value = " << min << "\n";
    out << "# Max value = " << max << "\n";
    out << "# Average value = " << sum / nUsed << "\n";
}

template<typename GRID_T, unsigned N>
CUDA_CALLABLE_KERNEL
void kernel_ScoreRayList(MonteRayNextEventEstimator<GRID_T>* ptr, const RayList_t<N>* pRayList ) {
    const bool debug = false;

    if( debug ) {
        printf("Debug: MonteRayNextEventEstimator::kernel_ScoreRayList\n");
    }

#ifdef __CUDACC__
    unsigned tid = threadIdx.x + blockIdx.x*blockDim.x;
#else
    unsigned tid = 0;
#endif

    unsigned num = pRayList->size();
    while( tid < num ) {
        if( debug ) {
            printf("Debug: MonteRayNextEventEstimator::kernel_ScoreRayList -- tid=%d\n", tid);
        }
        ptr->score(pRayList,tid);

#ifdef __CUDACC__
        tid += blockDim.x*gridDim.x;
#else
        ++tid;
#endif
    }
}

template<typename GRID_T>
template<unsigned N>
void MonteRayNextEventEstimator<GRID_T>::launch_ScoreRayList( unsigned nBlocks, unsigned nThreads, const RayList_t<N>* pRayList, cudaStream_t* stream )
{
    const bool debug = false;
    if( !initialized ) { initialize(); }

    if( MonteRayParallelAssistant::getInstance().getWorkGroupRank() != 0 ) return;

    const unsigned nRays = pRayList->size();
    if( nThreads > nRays ) {
        nThreads = nRays;
    }
    nThreads = (( nThreads + 32 -1 ) / 32 ) *32;

    const unsigned numThreadOverload = nBlocks;
    nBlocks = std::min(( nRays + numThreadOverload*nThreads -1 ) / (numThreadOverload*nThreads), 65535U);

    if( debug ) {
        printf("Debug: MonteRayNextEventEstimator::launch_ScoreRayList -- launching kernel_ScoreRayList with %d blocks, %d threads, to process %d rays\n", nBlocks, nThreads, nRays);
    }
#ifdef __CUDACC__
    if( stream ) {
        kernel_ScoreRayList<<<nBlocks, nThreads, 0, *stream>>>( Base::devicePtr, pRayList->devicePtr );
    } else {
        kernel_ScoreRayList<<<nBlocks, nThreads, 0, 0>>>( Base::devicePtr, pRayList->devicePtr );
    }
    if( debug ) {
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if( cudaerr != cudaSuccess ) {
            printf("MonteRayNextEventEstimator::launch_ScoreRayList -- kernel_ScoreRayList launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
        }
    }
#else
    kernel_ScoreRayList( this, pRayList );
#endif
}

template<typename GRID_T>
CUDAHOST_CALLABLE_MEMBER
void
MonteRayNextEventEstimator<GRID_T>::setGeometry(const GRID_T* pGrid, const MonteRay_MaterialProperties* pMPs) {
     pGridBins = pGrid;
     pMatPropsHost = pMPs;
     pMatProps = pMPs->getPtr();
}

template<typename GRID_T>
CUDAHOST_CALLABLE_MEMBER
void
MonteRayNextEventEstimator<GRID_T>::setMaterialList(const MonteRayMaterialListHost* ptr) {
     pMatListHost = ptr;
     pMatList = ptr->getPtr();
     pHashHost = ptr->getHashPtr();
     pHash = pHashHost->getPtr();
}

template<typename GRID_T>
CUDAHOST_CALLABLE_MEMBER
void
MonteRayNextEventEstimator<GRID_T>::gather() {
    if( !pTally ) {
        throw std::runtime_error("Error: MonteRayNextEventEstimator<GRID_T>::gather() -- Tally not allocated!!");
    }
    pTally->gather();
}

template<typename GRID_T>
CUDAHOST_CALLABLE_MEMBER
void
MonteRayNextEventEstimator<GRID_T>::gatherWorkGroup() {
    // mainly for testing
    if( !pTally ) {
        throw std::runtime_error("Error: MonteRayNextEventEstimator<GRID_T>::gatherWorkGroup() -- Tally not allocated!!");
    }
    pTally->gatherWorkGroup();
}

}

#endif /* MONTERAYNEXTEVENTESTIMATOR_T_HH_ */


