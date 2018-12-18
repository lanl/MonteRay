#ifndef MONTERAYNEXTEVENTESTIMATOR_T_HH_
#define MONTERAYNEXTEVENTESTIMATOR_T_HH_

#include <sys/types.h>
#include <unistd.h>

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

#ifdef DEBUG
    if( Base::debug ) {
        std::cout << "MonteRayNextEventEstimator::MonteRayNextEventEstimator(n), n=" << num << " \n";
    }
#endif

    reallocate(num);
}

template<typename GRID_T>
void
MonteRayNextEventEstimator<GRID_T>::reallocate(unsigned num) {
    if( x != NULL )     { MonteRayHostFree( x, Base::isManagedMemory ); }
    if( y != NULL )     { MonteRayHostFree( y, Base::isManagedMemory ); }
    if( z != NULL )     { MonteRayHostFree( z, Base::isManagedMemory ); }
    if( pTally != NULL ) { delete pTally; }
    if( pTallyTimeBinEdges != NULL ) { delete pTallyTimeBinEdges; }

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

#ifdef DEBUG
    if( Base::debug ) {
        std::cout << "Debug: MonteRayNextEventEstimator::copy (const MonteRayNextEventEstimator* rhs) \n";
    }
#endif

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


#ifdef DEBUG
    if( Base::debug ) {
        std::cout << "Debug: MonteRayNextEventEstimator::copy -- exiting." << std::endl;
    }
#endif

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

#ifdef DEBUG
    const bool debug = false;
#endif

    if( ray.detectorIndex >= nUsed ) {
        printf("ERROR: MonteRayNextEventEstimator::calcScore -- ray.detectorIndex < nUsed,  ray.detectorIndex = %d, nUsed  = %d\n", ray.detectorIndex, nUsed );
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

#ifdef DEBUG
    if( debug ) printf("Debug: MonteRayNextEventEstimator::calcScore -- distance to detector = %20.12f\n",dist );
#endif

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

#ifdef DEBUG
        if( debug ) {
            printf("Debug: MonteRayNextEventEstimator::calcScore -- energyIndex=%d, energy=%f, weight=%f\n", energyIndex, energy,  weight);
        }
#endif

        gpuFloatType_t materialXS[MAXNUMMATERIALS];
        for( unsigned i=0; i < pMatList->numMaterials; ++i ){
#ifdef DEBUG
            if( debug ) printf("Debug: MonteRayNextEventEstimator::calcScore -- materialIndex=%d\n", i);
#endif
            materialXS[i] = getTotalXS( pMatList, i, energy, 1.0);
#ifdef DEBUG
            if( debug ) {
                printf("Debug: MonteRayNextEventEstimator::calcScore -- materialIndex=%d, materialXS=%f\n", i, materialXS[i]);
            }
#endif
        }

        tally_t opticalThickness = 0.0;
        for( unsigned i=0; i < numberOfCells; ++i ){
            int cell = cells[i];
            gpuRayFloat_t cellDistance = crossingDistances[i];
            if( cell == UINT_MAX ) continue;

            gpuFloatType_t totalXS = 0.0;
            unsigned numMaterials = getNumMats( pMatProps, cell);

#ifdef DEBUG
            if( debug ) {
                printf("Debug: MonteRayNextEventEstimator::calcScore -- cell=%d, cellDistance=%f, numMaterials=%d\n", cell, cellDistance, numMaterials);
            }
#endif

            for( unsigned matIndex=0; matIndex<numMaterials; ++matIndex ) {
                unsigned matID = getMatID(pMatProps, cell, matIndex);
                gpuFloatType_t density = getDensity(pMatProps, cell, matIndex );
                gpuFloatType_t xs = materialXS[matID]*density;
                totalXS += xs;
#ifdef DEBUG
                if( debug ) {
                    printf("Debug: MonteRayNextEventEstimator::calcScore -- materialID=%d, density=%f, materialXS=%f, xs=%f, totalxs=%f\n", matID, density, materialXS[matID], xs, totalXS);
                }
#endif
            }

            opticalThickness += totalXS * cellDistance;
#ifdef DEBUG
            if( debug ) printf("Debug: MonteRayNextEventEstimator::calcScore -- optialThickness= %20.12f\n",opticalThickness );
#endif
        }

        partialScore = ( weight / (2.0 * MonteRay::pi * dist*dist)  ) * exp( - opticalThickness);
        score += partialScore;
    }

#ifdef DEBUG
    if( debug ) {
        printf("Debug: MonteRayNextEventEstimator::calcScore -- value=%e .\n" , score);
    }
#endif

//    gpu_atomicAdd( &tally[ray.detectorIndex], score );
    pTally->score(score, ray.detectorIndex, time);

    return score;
}

template<typename GRID_T>
template<unsigned N>
CUDA_CALLABLE_MEMBER
void
MonteRayNextEventEstimator<GRID_T>::score( const RayList_t<N>* pRayList, unsigned tid ) {
#ifdef DEBUG
    const bool debug = false;

    if( debug ) {
        printf("Debug: MonteRayNextEventEstimator::score -- tid=%d, particle=%d .\n",
                tid,  pRayList->points[tid].particleType);
    }
#endif

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
    out << "#  MonteRay results                                                    \n";
    out << "#                                                                      \n";
    out << "#       X          Y          Z      Time        Score        Score    \n";
    out << "#     (cm)       (cm)       (cm)   (shakes)      Average      Rel Err. \n";
    out << "# ________   ________   ________   ________   ___________   ___________ \n";
    //        12345678   12345678   12345678   12345678   12345678901   12345678901
    //     "12        123        123        123        123           123
    //boost::format fmt( "  %1$8.3f   %2$8.3f   %3$8.3f   %4$8.3f   %5$11.4e   %6$11.4e\n" );

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
        for( int j=0; j < pTally->getNumTimeBins(); ++j ) {

            double time = pTally->getTimeBinEdge(j);
            Vector3D<gpuFloatType_t> pos = getPoint(i);
            gpuFloatType_t value = getTally(i,j) / nSamples;

            if(  pos[dim2] < previousSecondDimPosition ) {
                out << "\n";
            }

            char buffer[200];
            snprintf( buffer, 200, "  %8.3f   %8.3f   %8.3f   %8.3f   %11.4e   %11.4e\n",
                                     pos[0], pos[1], pos[2],   time,   value,     0.0 );
            out << buffer;

            previousSecondDimPosition = pos[dim2];

            if( value < min ) min = value;
            if( value > max ) max = value;
            sum += value;
        }
    }
    out << "\n#\n";
    out << "# Min value = " << min << "\n";
    out << "# Max value = " << max << "\n";
    out << "# Average value = " << sum / nUsed << "\n";
}

template<typename GRID_T, unsigned N>
CUDA_CALLABLE_KERNEL
void kernel_ScoreRayList(MonteRayNextEventEstimator<GRID_T>* ptr, const RayList_t<N>* pRayList ) {
#ifdef DEBUG
    const bool debug = false;

    if( debug ) {
        printf("Debug: MonteRayNextEventEstimator::kernel_ScoreRayList\n");
    }
#endif

#ifdef __CUDACC__
    unsigned tid = threadIdx.x + blockIdx.x*blockDim.x;
#else
    unsigned tid = 0;
#endif

    unsigned num = pRayList->size();
    while( tid < num ) {

#ifdef DEBUG
        if( debug ) {
            printf("Debug: MonteRayNextEventEstimator::kernel_ScoreRayList -- tid=%d\n", tid);
        }
#endif

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
void MonteRayNextEventEstimator<GRID_T>::launch_ScoreRayList( int nBlocksArg, int nThreadsArg, const RayList_t<N>* pRayList, cudaStream_t* stream, bool dumpOnFailure )
{
    // negative nBlocks and nThreads forces to specified value,
    // otherwise reasonable values are used based on the specified ones

    //const bool debug = false;
    if( !initialized ) { initialize(); }
    const MonteRayParallelAssistant& PA( MonteRayParallelAssistant::getInstance() );

    if( PA.getWorkGroupRank() != 0 ) return;

    unsigned nThreads = std::abs(nThreadsArg);
    unsigned nBlocks = std::abs(nBlocksArg);

    const unsigned nRays = pRayList->size();
    if( nThreadsArg > 0 ) {
        if( nThreads > nRays ) {
            nThreads = nRays;
        }
        nThreads = (( nThreads + 32 -1 ) / 32 ) *32;
    }

    if( nBlocksArg > 0 ) {
        const unsigned numThreadOverload = nBlocks;
        nBlocks = std::min(( nRays + numThreadOverload*nThreads -1 ) / (numThreadOverload*nThreads), 65535U);
    }

#ifdef DEBUG
        std::cout << "Debug: MonteRayNextEventEstimator::launch_ScoreRayList -- launching kernel_ScoreRayList on " <<
                     PA.info() << " with " << nBlocks << " blocks, " << nThreads <<
                     " threads, to process " << nRays << " rays\n";
#endif

#ifdef __CUDACC__
#ifndef DEBUG
    // Release compile
    if( stream ) {
        kernel_ScoreRayList<<<nBlocks, nThreads, 0, *stream>>>( Base::devicePtr, pRayList->devicePtr );
    } else {
        kernel_ScoreRayList<<<nBlocks, nThreads, 0, 0>>>( Base::devicePtr, pRayList->devicePtr );
    }
#else
    // Debug compile
    kernel_ScoreRayList<<<nBlocks, nThreads, 0, 0>>>( Base::devicePtr, pRayList->devicePtr );

    bool failure = false;
    std::stringstream msg;

    // first check launch failure
    cudaError_t error = cudaPeekAtLastError();
    if( error != cudaSuccess ) {
        failure = true;
        msg << "ERROR:  MonteRayNextEventEstimator::launch_ScoreRayList -- kernel_ScoreRayList launch on " <<
                PA.info() << " failed with error " << cudaGetErrorString(error) << ".\n";
    }

    // second check successful kernel completion.
    error = cudaDeviceSynchronize();
    if( !failure and error != cudaSuccess ) {
        failure = true;
        msg << "ERROR:  MonteRayNextEventEstimator::launch_ScoreRayList -- kernel_ScoreRayList execution on " <<
                PA.info() << " failed with error " << cudaGetErrorString(error) << ".\n";
    }

    if( failure ) {
        std::cout << msg.str();
        if( dumpOnFailure ) dumpState(pRayList);
        throw std::runtime_error( msg.str() );
    }

#endif
#else
    kernel_ScoreRayList( this, pRayList );
#endif
}

template<typename GRID_T>
void
MonteRayNextEventEstimator<GRID_T>::writeToFile( const std::string& filename) {
    std::ofstream state;
    state.open( filename.c_str(), std::ios::binary | std::ios::out);
    write( state );
    state.close();
}


template<typename GRID_T>
void
MonteRayNextEventEstimator<GRID_T>::readFromFile( const std::string& filename) {
    std::ifstream state;
    state.open( filename.c_str(), std::ios::binary | std::ios::in);
    if( ! state.good() ) {
        throw std::runtime_error( "MonteRayNextEventEstimator::read -- can't open file for reading" );
    }
    read( state );
    state.close();
}

template<typename GRID_T>
template<unsigned N>
void
MonteRayNextEventEstimator<GRID_T>::dumpState( const RayList_t<N>* pRayList, const std::string& optBaseName ) {
    const MonteRayParallelAssistant& PA( MonteRayParallelAssistant::getInstance() );

    std::string baseName;

    if( optBaseName.empty() ) {
        pid_t processID = getpid();

        baseName =  PA.hostname() + std::string("_") +
                    std::to_string( PA.getDeviceID() ) + std::string("_") +
                    std::to_string( processID ) + std::string(".bin");
    } else {
        baseName = optBaseName + std::string(".bin");
    }

    // write out state of MonteRayNextEventEstimator class
    std::string filename = std::string("nee_state_") + baseName;
    writeToFile( filename );

    // write out particle list
    filename = std::string("raylist_") + baseName;
    pRayList->writeToFile( filename );

    // write out geometry
    filename = std::string("geometry_") + baseName;
    pGridBins->writeToFile( filename );

    // write out material properties
    filename = std::string("matProps_") + baseName;
    pMatPropsHost->writeToFile( filename );

    // write out materials
    filename = std::string("materialList_") + baseName;
    pMatListHost->writeToFile( filename );

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

template<typename GRID_T>
template<typename IOTYPE>
void
MonteRayNextEventEstimator<GRID_T>::write(IOTYPE& out) {
    unsigned version = 0;
    binaryIO::write( out, version );
    binaryIO::write( out, nAllocated );
    binaryIO::write( out, nUsed );
    binaryIO::write( out, radius );
    for( unsigned i=0; i<nUsed; ++i ){ binaryIO::write( out, x[i] ); }
    for( unsigned i=0; i<nUsed; ++i ){ binaryIO::write( out, y[i] ); }
    for( unsigned i=0; i<nUsed; ++i ){ binaryIO::write( out, z[i] ); }

    bool hasTimeBinEdges = false;
    if( pTallyTimeBinEdges ) {
        hasTimeBinEdges = true;
    }
    binaryIO::write( out, hasTimeBinEdges );
    if( hasTimeBinEdges ) {
        binaryIO::write( out, *pTallyTimeBinEdges );
    }

}

template<typename GRID_T>
template<typename IOTYPE>
void
MonteRayNextEventEstimator<GRID_T>::read(IOTYPE& in) {
    initialized = false;
    copiedToGPU = false;

    unsigned version = 0;
    binaryIO::read( in, version );
    binaryIO::read( in, nAllocated );
    reallocate(nAllocated);
    binaryIO::read( in, nUsed );
    binaryIO::read( in, radius );
    for( unsigned i=0; i<nUsed; ++i ){ binaryIO::read( in, x[i] ); }
    for( unsigned i=0; i<nUsed; ++i ){ binaryIO::read( in, y[i] ); }
    for( unsigned i=0; i<nUsed; ++i ){ binaryIO::read( in, z[i] ); }

    bool hasTimeBinEdges = true;
    binaryIO::read( in, hasTimeBinEdges );
    if( hasTimeBinEdges ) {
        pTallyTimeBinEdges = new std::vector<gpuFloatType_t>;
        binaryIO::read( in, *pTallyTimeBinEdges );
    }

    initialize();

}

} // end namespace

#endif /* MONTERAYNEXTEVENTESTIMATOR_T_HH_ */


