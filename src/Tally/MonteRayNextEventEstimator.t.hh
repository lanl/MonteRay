#ifndef MONTERAYNEXTEVENTESTIMATOR_T_HH_
#define MONTERAYNEXTEVENTESTIMATOR_T_HH_

#include <sys/types.h>
#include <unistd.h>
#include <limits>
#include <tuple>

#include "MonteRayNextEventEstimator.hh"

#include "MonteRayDefinitions.hh"
#include "MonteRayCopyMemory.t.hh"
#include "HashLookup.hh"
#include "MonteRay_MaterialProperties.hh"
#include "MonteRayMaterialList.hh"
#include "ExpectedPathLength.hh"
#include "RayList.hh"
#include "RayWorkInfo.hh"
#include "MonteRayTally.hh"
#include "MonteRayParallelAssistant.hh"
#include "GPUUtilityFunctions.hh"
#include "RayWorkInfo.hh"

namespace MonteRay {

template<typename Geometry>
CUDAHOST_CALLABLE_MEMBER
MonteRayNextEventEstimator<Geometry>::MonteRayNextEventEstimator(unsigned num) {
    if( num == 0 ) { num = 1; }
    reallocate(num);
}

template<typename Geometry>
void
MonteRayNextEventEstimator<Geometry>::reallocate(unsigned num) {
    if( tallyPoints != NULL )     { MonteRayHostFree( tallyPoints, Base::isManagedMemory ); }
    if( pTally != NULL ) { delete pTally; }
    if( pTallyTimeBinEdges != NULL ) { delete pTallyTimeBinEdges; }

    init();

    tallyPoints = (decltype(tallyPoints)) MONTERAYHOSTALLOC( num*sizeof( decltype(*tallyPoints) ), Base::isManagedMemory, "tallyPoints" );
    nAllocated = num;
}


template<typename Geometry>
CUDAHOST_CALLABLE_MEMBER
MonteRayNextEventEstimator<Geometry>::~MonteRayNextEventEstimator(){
    if( Base::isCudaIntermediate ) {
        if( tallyPoints != NULL )     { MonteRayDeviceFree( tallyPoints ); }
    } else {
        if( tallyPoints != NULL )     { MonteRayHostFree( tallyPoints, Base::isManagedMemory ); }
        if( pTally != NULL ) { delete pTally; }
        if( pTallyTimeBinEdges != NULL ) { delete pTallyTimeBinEdges; }
    }
}

template<typename Geometry>
CUDAHOST_CALLABLE_MEMBER
void
MonteRayNextEventEstimator<Geometry>::init() {
     nUsed = 0;
     nAllocated = 0;
     radius = 0.0;

     tallyPoints = NULL;
     pTally = NULL;
     pTallyTimeBinEdges = NULL;

     pGeometry = NULL;
     pMatPropsHost = NULL;
     pMatProps = NULL;
     pMatListHost = NULL;
     pMatList = NULL;
     pHashHost = NULL;
     pHash = NULL;

     initialized = false;
     copiedToGPU = false;
 }

template<typename Geometry>
void MonteRayNextEventEstimator<Geometry>::initialize() {
    if( initialized ) return;

    pTally = new MonteRayTally(nUsed);
    if( pTallyTimeBinEdges) { pTally->setTimeBinEdges( *pTallyTimeBinEdges ); }
    pTally->initialize();

    initialized = true;
}

template<typename Geometry>
CUDAHOST_CALLABLE_MEMBER
unsigned
MonteRayNextEventEstimator<Geometry>::add( position_t xarg, position_t yarg, position_t zarg) {

     MONTERAY_VERIFY( nUsed < nAllocated, "MonteRayNextEventEstimator::add -- Detector list is full.  Can't add any more detectors." );

     tallyPoints[nUsed][0] = xarg;
     tallyPoints[nUsed][1] = yarg;
     tallyPoints[nUsed][2] = zarg;
     return nUsed++;
 }

template<typename Geometry>
CUDAHOST_CALLABLE_MEMBER
void
MonteRayNextEventEstimator<Geometry>::copy(const MonteRayNextEventEstimator* rhs) {
    if( ! MonteRay::isWorkGroupMaster() ) return;

    if( Base::isCudaIntermediate  and !rhs->initialized ) {
        throw std::runtime_error("MonteRayNextEventEstimator<Geometry>::copy -- not initialized"  );
    }

#ifdef __CUDACC__

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
        if( tallyPoints == NULL ) {
            tallyPoints = tallyPoints = (decltype(tallyPoints)) MONTERAYDEVICEALLOC( num*sizeof( decltype(*tallyPoints) ), "tallyPoints" );
        }
        MonteRayMemcpy(tallyPoints, rhs->tallyPoints, num*sizeof( decltype(*tallyPoints) ), cudaMemcpyHostToDevice);

        pMatPropsHost = NULL;
        pMatListHost = NULL;
        pHashHost = NULL;

        pGeometry = rhs->pGeometry->getDevicePtr();
        pMatProps = rhs->pMatPropsHost->ptrData_device;
        pMatList = rhs->pMatListHost->ptr_device;
        pHash = rhs->pMatListHost->getHashPtr()->getPtrDevice();
        pTally = rhs->pTally->devicePtr;

        nAllocated = rhs->nAllocated;
        nUsed = rhs->nUsed;
        radius = rhs->radius;
        initialized = rhs->initialized;
        copiedToGPU = rhs->copiedToGPU;

    } 

#else

#endif
}

template<typename Geometry>
CUDAHOST_CALLABLE_MEMBER
void
MonteRayNextEventEstimator<Geometry>::copyToGPU(){
    if( ! MonteRay::isWorkGroupMaster() ) return;

    copiedToGPU = true;
    pTally->copyToGPU();
    Base::copyToGPU();
}

template<typename Geometry>
CUDAHOST_CALLABLE_MEMBER
void
MonteRayNextEventEstimator<Geometry>::copyToCPU(){
    if( !copiedToGPU ) return;

    if( ! MonteRay::isWorkGroupMaster() ) return;

    pTally->copyToCPU();
    Base::copyToCPU();

}

template<typename Geometry>
CUDA_CALLABLE_MEMBER
typename MonteRayNextEventEstimator<Geometry>::tally_t
MonteRayNextEventEstimator<Geometry>::getTally(unsigned spatialIndex, unsigned timeIndex) const {
    return pTally->getTally(spatialIndex, timeIndex);
}

template<typename Geometry>
template<unsigned N>
CUDA_CALLABLE_MEMBER
typename MonteRayNextEventEstimator<Geometry>::tally_t
MonteRayNextEventEstimator<Geometry>::calcScore( unsigned threadID, Ray_t<N>& ray, RayWorkInfo& rayInfo ) {

    MONTERAY_ASSERT( ray.detectorIndex < nUsed);
    tally_t score = 0.0;
    

    // TPB TODO: remove construction of pos2 and just have a points be a list of Vector3Ds
    MonteRay::Vector3D<gpuRayFloat_t> pos( ray.pos[0], ray.pos[1], ray.pos[2]);
    auto distAndDir = getDistanceDirection(pos, tallyPoints[ray.detectorIndex]);
    auto& dist = std::get<0>(distAndDir);
    auto& dir = std::get<1>(distAndDir);

    gpuFloatType_t time = ray.time + dist / ray.speed();

    pGeometry->rayTrace( threadID, rayInfo, pos, dir, dist, false);

    for( unsigned energyIndex=0; energyIndex < N; ++energyIndex) {
        gpuFloatType_t weight = ray.weight[energyIndex];
        if( weight == 0.0 ) continue;

        gpuFloatType_t energy = ray.energy[energyIndex];
        if( energy <  1.0e-11 ) continue;

        tally_t partialScore = 0.0;

        gpuFloatType_t materialXS[MAXNUMMATERIALS];
        for( unsigned i=0; i < pMatList->numMaterials; ++i ){
            materialXS[i] = getTotalXS( pMatList, i, energy, 1.0);
        }

        tally_t opticalThickness = 0.0;
        for( unsigned i=0; i < rayInfo.getRayCastSize( threadID ); ++i ){
            int cell = rayInfo.getRayCastCell( threadID, i);
            gpuRayFloat_t cellDistance = rayInfo.getRayCastDist( threadID, i);
            if( cell == std::numeric_limits<unsigned int>::max() ) continue;

            gpuFloatType_t totalXS = 0.0;
            unsigned numMaterials = getNumMats( pMatProps, cell);


            for( unsigned matIndex=0; matIndex<numMaterials; ++matIndex ) {
                unsigned matID = getMatID(pMatProps, cell, matIndex);
                gpuFloatType_t density = getDensity(pMatProps, cell, matIndex );
                gpuFloatType_t xs = materialXS[matID]*density;
                totalXS += xs;
            }

            opticalThickness += totalXS * cellDistance;
        }

        partialScore = ( weight / (2.0 * MonteRay::pi * dist*dist)  ) * exp( - opticalThickness);
        score += partialScore;
    }


    pTally->score(score, ray.detectorIndex, time);

    return score;
}

template<typename Geometry>
template<unsigned N>
CUDA_CALLABLE_MEMBER
void
MonteRayNextEventEstimator<Geometry>::score( const RayList_t<N>* pRayList, RayWorkInfo* pRayInfo, unsigned tid, unsigned pid ) {

    // Neutrons are not yet supported
    // TPB: TODO remove this. Neutrons shouldn't have made it in the bank if they're not scorable.
    auto& ray = pRayList->points[pid];
    if( ray.particleType == neutron ) return;

    tally_t value = calcScore<N>( tid, ray, *pRayInfo );
}

template<typename Geometry>
CUDAHOST_CALLABLE_MEMBER
void
MonteRayNextEventEstimator<Geometry>::printPointDets( const std::string& outputFile, unsigned nSamples, unsigned constantDimension) {
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

template<typename Geometry>
CUDAHOST_CALLABLE_MEMBER
void
MonteRayNextEventEstimator<Geometry>::outputTimeBinnedTotal(std::ostream& out,unsigned nSamples, unsigned constantDimension){
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

template<typename Geometry, unsigned N>
CUDA_CALLABLE_KERNEL  kernel_ScoreRayList(MonteRayNextEventEstimator<Geometry>* ptr, const RayList_t<N>* pRayList, RayWorkInfo* pRayInfo ) {

#ifdef __CUDACC__
    unsigned threadID = threadIdx.x + blockIdx.x*blockDim.x;
#else
    unsigned threadID = 0;
#endif

    unsigned particleID = threadID;

    unsigned num = pRayList->size();
    while( particleID < num ) {
        pRayInfo->clear( threadID );


        ptr->score(pRayList, pRayInfo, threadID, particleID);

#ifdef __CUDACC__
        particleID += blockDim.x*gridDim.x;
#else
        ++particleID;
#endif
    }
}

template<typename Geometry>
template<unsigned N>
void MonteRayNextEventEstimator<Geometry>::launch_ScoreRayList( int nBlocksArg, int nThreadsArg, const RayList_t<N>* pRayList, RayWorkInfo* pRayInfo, cudaStream_t* stream, bool dumpOnFailure )
{
    // negative nBlocks and nThreads forces to specified value,
    // otherwise reasonable values are used based on the specified ones

    if( !initialized ) { initialize(); }
    const MonteRayParallelAssistant& PA( MonteRayParallelAssistant::getInstance() );

    if( PA.getWorkGroupRank() != 0 ) return;

    auto launchBounds = setLaunchBounds( nThreadsArg, nBlocksArg, pRayList->size() );
    unsigned nBlocks = launchBounds.first;
    unsigned nThreads = launchBounds.second;

#ifdef __CUDACC__
    if( stream ) {
        kernel_ScoreRayList<<<nBlocks, nThreads, 0, *stream>>>( Base::devicePtr, pRayList->devicePtr, pRayInfo->devicePtr );
    } else {
        kernel_ScoreRayList<<<nBlocks, nThreads, 0, 0>>>( Base::devicePtr, pRayList->devicePtr, pRayInfo->devicePtr );
    }
#else
    kernel_ScoreRayList( this, pRayList, pRayInfo );
#endif
}

template<typename Geometry>
void
MonteRayNextEventEstimator<Geometry>::writeToFile( const std::string& filename) {
    std::ofstream state;
    state.open( filename.c_str(), std::ios::binary | std::ios::out);
    write( state );
    state.close();
}


template<typename Geometry>
void
MonteRayNextEventEstimator<Geometry>::readFromFile( const std::string& filename) {
    std::ifstream state;
    state.open( filename.c_str(), std::ios::binary | std::ios::in);
    if( ! state.good() ) {
        throw std::runtime_error( "MonteRayNextEventEstimator::read -- can't open file for reading" );
    }
    read( state );
    state.close();
}

template<typename Geometry>
template<unsigned N>
void
MonteRayNextEventEstimator<Geometry>::dumpState( const RayList_t<N>* pRayList, const std::string& optBaseName ) {
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
    pGeometry->writeToFile( filename );

    // write out material properties
    filename = std::string("matProps_") + baseName;
    pMatPropsHost->writeToFile( filename );

    // write out materials
    filename = std::string("materialList_") + baseName;
    pMatListHost->writeToFile( filename );

}

template<typename Geometry>
CUDAHOST_CALLABLE_MEMBER
void
MonteRayNextEventEstimator<Geometry>::setGeometry(const Geometry* pGeometryIn, const MonteRay_MaterialProperties* pMPs) {
     pGeometry = pGeometryIn;
     pMatPropsHost = pMPs;
     pMatProps = pMPs->getPtr();
}

template<typename Geometry>
CUDAHOST_CALLABLE_MEMBER
void
MonteRayNextEventEstimator<Geometry>::updateMaterialProperties( MonteRay_MaterialProperties* pMPs) {
     pMatPropsHost = pMPs;
     pMatProps = pMPs->getPtr();

     if( copiedToGPU ) {
         Base::copyToGPU();
     }
}

template<typename Geometry>
CUDAHOST_CALLABLE_MEMBER
void
MonteRayNextEventEstimator<Geometry>::setMaterialList(const MonteRayMaterialListHost* ptr) {
     pMatListHost = ptr;
     pMatList = ptr->getPtr();
     pHashHost = ptr->getHashPtr();
     pHash = pHashHost->getPtr();
}

template<typename Geometry>
CUDAHOST_CALLABLE_MEMBER
void
MonteRayNextEventEstimator<Geometry>::gather() {
    if( !pTally ) {
        throw std::runtime_error("Error: MonteRayNextEventEstimator<Geometry>::gather() -- Tally not allocated!!");
    }
    pTally->gather();
}

template<typename Geometry>
CUDAHOST_CALLABLE_MEMBER
void
MonteRayNextEventEstimator<Geometry>::gatherWorkGroup() {
    // mainly for testing
    if( !pTally ) {
        throw std::runtime_error("Error: MonteRayNextEventEstimator<Geometry>::gatherWorkGroup() -- Tally not allocated!!");
    }
    pTally->gatherWorkGroup();
}

template<typename Geometry>
template<typename IOTYPE>
void
MonteRayNextEventEstimator<Geometry>::write(IOTYPE& out) {
    unsigned version = 0;
    binaryIO::write( out, version );
    binaryIO::write( out, nAllocated );
    binaryIO::write( out, nUsed );
    binaryIO::write( out, radius );
    for( unsigned i=0; i<nUsed; ++i ){ binaryIO::write( out, tallyPoints[i][0] ); }
    for( unsigned i=0; i<nUsed; ++i ){ binaryIO::write( out, tallyPoints[i][1] ); }
    for( unsigned i=0; i<nUsed; ++i ){ binaryIO::write( out, tallyPoints[i][2] ); }

    bool hasTimeBinEdges = false;
    if( pTallyTimeBinEdges ) {
        hasTimeBinEdges = true;
    }
    binaryIO::write( out, hasTimeBinEdges );
    if( hasTimeBinEdges ) {
        binaryIO::write( out, *pTallyTimeBinEdges );
    }

}

template<typename Geometry>
template<typename IOTYPE>
void
MonteRayNextEventEstimator<Geometry>::read(IOTYPE& in) {
    initialized = false;
    copiedToGPU = false;

    unsigned version = 0;
    binaryIO::read( in, version );
    binaryIO::read( in, nAllocated );
    reallocate(nAllocated);
    binaryIO::read( in, nUsed );
    binaryIO::read( in, radius );
    for( unsigned i=0; i<nUsed; ++i ){ binaryIO::read( in, tallyPoints[i][0] ); }
    for( unsigned i=0; i<nUsed; ++i ){ binaryIO::read( in, tallyPoints[i][1] ); }
    for( unsigned i=0; i<nUsed; ++i ){ binaryIO::read( in, tallyPoints[i][2] ); }

    bool hasTimeBinEdges = true;
    binaryIO::read( in, hasTimeBinEdges );
    if( hasTimeBinEdges ) {
        pTallyTimeBinEdges = new std::vector<gpuFloatType_t>;
        binaryIO::read( in, *pTallyTimeBinEdges );
    }

    initialize();

}

template<typename Geometry>
template<unsigned N>
void
MonteRayNextEventEstimator<Geometry>::cpuScoreRayList( const RayList_t<N>* pRayList, RayWorkInfo* pRayInfo ) {
    for( auto i=0; i<pRayList->size(); ++i ) {
        pRayInfo->clear(0);
        score(pRayList, pRayInfo, 0, i);
    }
}

} // end namespace

#endif /* MONTERAYNEXTEVENTESTIMATOR_T_HH_ */


