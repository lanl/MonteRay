#ifndef RAYLISTCONTROLLER_T_HH_
#define RAYLISTCONTROLLER_T_HH_

#include "RayListController.hh"

#include "ReadAndWriteFiles.hh"
#include "ExpectedPathLength.t.hh"
#include "NextEventEstimator.t.hh"

namespace MonteRay {

template<typename Geometry, unsigned N>
RayListController<Geometry,N>::RayListController(
        int blocks,
        int threads,
        Geometry* pGB,
        MaterialList* pML,
        MaterialProperties* pMP,
        BasicTally* pT
) :
nBlocks(blocks),
nThreads(threads),
pGeometry( pGB ),
pMatList( pML ),
pMatProps( pMP ),
pTally(pT),
PA( MonteRayParallelAssistant::getInstance() )
{
    initialize();
    kernel = [&, blocks, threads ] ( void ) {
      if( PA.getWorkGroupRank() != 0 ) { return; }

      auto launchBounds = setLaunchBounds( threads, blocks,  currentBank->getPtrPoints()->size() );

#ifndef NDEBUG
      size_t freeMemory = 0;
      size_t totalMemory = 0;
#ifdef __CUDACC__
      cudaError_t memError = cudaMemGetInfo( &freeMemory, &totalMemory);
      freeMemory = freeMemory/1000000;
      totalMemory = totalMemory/1000000;
#endif
      std::cout << "Debug: RayListController -- launching kernel on " <<
                   PA.info() << " with " << launchBounds.first << " blocks, " << launchBounds.second  <<
                   " threads, to process " << currentBank->getPtrPoints()->size() << " rays," <<
                   " free GPU memory= " << freeMemory << "MB, total GPU memory= " << totalMemory << "MB \n";
#endif

      /* if (pMatProps->usingMaterialMotion()){ */
        /* constexpr gpuFloatType_t timeRemaining = 10.0E6; */
        /* rayTraceTallyWithMovingMaterials( */
        /*     currentBank->getPtrPoints(), */
        /*     timeRemaining, */
        /*     pGeometry, */
        /*     pMatProps, */
        /*     pMatList, */
        /*     pTally->data(), */
        /*     stream1.get()); */
      /* } else */
      {
#ifdef __CUDACC__
      rayTraceTally<<<launchBounds.first,launchBounds.second,0, *stream1>>>(
              pGeometry,
              currentBank->getPtrPoints(),
              pMatList,
              pMatProps,
              rayInfo.get(),
              pTally->data() );
#else
      rayTraceTally( 
              pGeometry,
              currentBank->getPtrPoints(),
              pMatList,
              pMatProps,
              rayInfo.get(),
              pTally->data() );
#endif

      }
  };

}

template<typename Geometry, unsigned N>
RayListController<Geometry,N>::RayListController(
        int blocks,
        int threads,
        Geometry* pGB,
        MaterialList* pML,
        MaterialProperties* pMP,
        unsigned numPointDets
) :
nBlocks(blocks),
nThreads(threads),
pGeometry( pGB ),
pMatList( pML ),
pMatProps( pMP ),
pTally(NULL),
PA( MonteRayParallelAssistant::getInstance() )
{
    pNextEventEstimatorBuilder = std::make_unique<NextEventEstimator::Builder>();
    pNextEventEstimator = std::make_unique<NextEventEstimator>( );
    usingNextEventEstimator = true;
    initialize();
    kernel = [&, blocks, threads] ( void ) {
      if( currentBank->size() > 0 ) {
        launch_ScoreRayList(pNextEventEstimator.get(), blocks,threads, currentBank->getPtrPoints(), rayInfo.get(), 
            pGeometry, pMatProps, pMatList, stream1.get() );
      }
    };
}

template<typename Geometry, unsigned N>
RayListController<Geometry,N>::RayListController( unsigned numPointDets, const std::string& filename ) :
PA( MonteRayParallelAssistant::getInstance() )
{
    initialize();
    pNextEventEstimatorBuilder = std::make_unique<NextEventEstimator::Builder>();
    pNextEventEstimator = std::make_unique<NextEventEstimator>( );
    setOutputFileName( filename );
    usingNextEventEstimator = true;
    kernel = [&] ( void ) {
        // do nothing
        return;
    };
}

template<typename Geometry, unsigned N>
void
RayListController<Geometry, N>::initialize(){
    if( PA.getWorkGroupRank() != 0 ) { return; }

    setCapacity( 100000 ); // default buffer capacity to 0.1 million.

    pTimer.reset( new cpuTimer );

#ifdef __CUDACC__
    stream1.reset( new cudaStream_t);
    startGPU.reset( new cudaEvent_t);
    stopGPU.reset( new cudaEvent_t);
    start.reset( new cudaEvent_t);
    stop.reset( new cudaEvent_t);
    copySync1.reset( new cudaEvent_t);
    copySync2.reset( new cudaEvent_t);

    cudaStreamCreate( stream1.get() );
    cudaEventCreate(start.get());
    cudaEventCreate(stop.get());
    cudaEventCreate(startGPU.get());
    cudaEventCreate(stopGPU.get());
    cudaEventCreate(copySync1.get());
    cudaEventCreate(copySync2.get());
    currentCopySync = copySync1.get();
#endif
}

template<typename Geometry, unsigned N>
RayListController<Geometry,N>::~RayListController(){

#ifdef __CUDACC__
    cudaDeviceSynchronize();
    if( stream1 ) cudaStreamDestroy( *stream1 );
#endif
}

template<typename Geometry, unsigned N>
unsigned
RayListController<Geometry,N>::capacity(void) const {
    if(currentBank) return currentBank->capacity();
    return 0;
}

template<typename Geometry, unsigned N>
unsigned
RayListController<Geometry,N>::size(void) const {
    if(currentBank) return currentBank->size();
    return 0;
}

template<typename Geometry, unsigned N>
void
RayListController<Geometry,N>::setCapacity(unsigned n) {
//    if( n > 100000 ) {
//        std::cout << "Debug: WARNING: MonteRay -- RayListController::setCapacity, limiting capacity to 100000\n";
//        n = 100000;
//    }

    if( PA.getWorkGroupRank() != 0 ) { return; }
    bank1.reset( new RayListInterface<N>(n) );
    bank2.reset( new RayListInterface<N>(n) );
    currentBank = bank1.get();

    auto launchBounds = setLaunchBounds( nThreads, nBlocks, n );
    unsigned totalNumThreads  = launchBounds.first * launchBounds.second;

    // size rayInfo to total number of threads
    rayInfo.reset( new RayWorkInfo( totalNumThreads ) );
}

template<typename Geometry, unsigned N>
unsigned
RayListController<Geometry,N>::getWorldRank() {
    return PA.getWorldRank();
}

template<typename Geometry, unsigned N>
void
RayListController<Geometry,N>::flush(bool final){
    if( PA.getWorkGroupRank() != 0 ) { return; }

    if( isSendingToFile() ) { flushToFile(final); }

    if( currentBank->size() == 0 ) {
        if( final ) {
            cudaDeviceSynchronize();
            printTotalTime();
            currentBank->clear();
        }
        return;
    }

    if( nFlushs > 0 ) {
        /* std::cout << "Debug: flush nFlushs =" <<nFlushs-1 << " -- stopping timers\n"; */
        stopTimers();
    }
    /* std::cout << "Debug: flush nFlushs =" <<nFlushs << " -- starting timers\n"; */

    startTimers();

    ++nFlushs;

#ifdef __CUDACC__
    gpuErrchk( cudaPeekAtLastError() );
    currentBank->copyToGPU();
    gpuErrchk( cudaEventRecord(*currentCopySync, 0) );
    gpuErrchk( cudaEventSynchronize(*currentCopySync) );
#endif

    // launch kernel
    kernel();

    // only uncomment for testing, forces the cpu and gpu to sync
#ifndef NDEBUG
#ifdef __CUDACC__
    gpuErrchk( cudaPeekAtLastError() );
#endif
#endif

#ifdef __CUDACC__
    gpuErrchk( cudaEventRecord( *stopGPU, *stream1) );
    gpuErrchk( cudaStreamWaitEvent( *stream1, *stopGPU, 0) );
#endif

    if( final ) {
        std::cout << "Debug: final flush nFlushs =" <<nFlushs-1 << " -- stopping timers\n";
        stopTimers();
        printTotalTime();
        currentBank->clear();
        return;
    }

    swapBanks();
}

template<typename Geometry, unsigned N>
void
RayListController<Geometry,N>::flushToFile(bool final){
    if( PA.getWorldRank() != 0 ) { return; }

#ifndef NDEBUG
    const bool debug = false;

    if( debug ) {
        if( final ) {
            std::cout << "Debug: RayListController::flushToFile - starting -- final = true \n";
        } else {
            std::cout << "Debug: RayListController::flushToFile - starting -- final = false \n";
        }
    }
#endif

    if( ! fileIsOpen ) {
        try {
#ifndef NDEBUG
            if( debug ) std::cout << "Debug: RayListController::flushToFile - opening file, filename=" << outputFileName << "\n";
#endif
            currentBank->openOutput( outputFileName );
        } catch ( ... ) {
            std::stringstream msg;
            msg << "Failure opening file for collision writing!\n";
            msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "RayListController::flushToFile" << "\n\n";
            std::cout << "MonteRay Error: " << msg.str();
            throw std::runtime_error( msg.str() );
        }

        fileIsOpen = true;
    }

    try {
#ifndef NDEBUG
        if( debug )  std::cout << "Debug: RayListController::flushToFile - writing bank -- bank size = "<< currentBank->size() << "\n";
#endif
        currentBank->writeBank();
    } catch ( ... ) {
        std::stringstream msg;
        msg << "Failure writing collisions to file!\n";
        msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "RayListController::flushToFile" << "\n\n";
        std::cout << "MonteRay Error: " << msg.str();
        throw std::runtime_error( msg.str() );
    }

    currentBank->clear();

    if( final ) {
        try {
#ifndef NDEBUG
            if( debug ) std::cout << "Debug: RayListController::flushToFile - file flush, closing collision file\n";
#endif
            currentBank->closeOutput();
        } catch ( ... ) {
            std::stringstream msg;
            msg << "Failure closing collision file!\n";
            msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " <<"RayListController::flushToFile" << "\n\n";
            std::cout << "MonteRay Error: " << msg.str();
            throw std::runtime_error( msg.str() );
        }

        fileIsOpen = false;
    }
}

template<typename Geometry, unsigned N>
size_t
RayListController<Geometry,N>::readCollisionsFromFile(std::string name) {
    if( PA.getWorldRank() != 0 ) { return 0; }

    bool end = false;
    unsigned numParticles = 0;

    do  {
        end = currentBank->readToBank(name, numParticles);
        numParticles += currentBank->size();
        flush(end);
    } while ( ! end );
    return numParticles;
}

template<typename Geometry, unsigned N>
size_t
RayListController<Geometry,N>::readCollisionsFromFileToBuffer(std::string name){
    if( PA.getWorldRank() != 0 ) { return 0; }

    unsigned numParticles = 0;
    currentBank->readToBank(name, numParticles);
    numParticles += currentBank->size();
    return numParticles;
}

template<typename Geometry, unsigned N>
void
RayListController<Geometry,N>::startTimers(){
    // start timers
    if( PA.getWorkGroupRank() != 0 ) { return; }

    pTimer->start();
#ifdef __CUDACC__
    gpuErrchk( cudaEventRecord( *start,0) );
    gpuErrchk( cudaEventRecord( *startGPU, *stream1) );
#endif
}

template<typename Geometry, unsigned N>
void
RayListController<Geometry,N>::stopTimers(){
    // stop timers and sync
    if( PA.getWorkGroupRank() != 0 ) { return; }

    pTimer->stop();
    float_t cpuCycleTime = pTimer->getTime();
    cpuTime += cpuCycleTime;

#ifdef __CUDACC__
    gpuErrchk( cudaStreamSynchronize( *stream1 ) );
    gpuErrchk( cudaEventRecord(*stop, 0) );
    gpuErrchk( cudaEventSynchronize( *stop ) );

    float_t gpuCycleTime;
    gpuErrchk( cudaEventElapsedTime(&gpuCycleTime, *startGPU, *stopGPU ) );
    gpuCycleTime /= 1000.0;
    if( gpuCycleTime < 0.0 ) {
        gpuCycleTime = 0.0;
    }
    gpuTime += gpuCycleTime;

    float totalCycleTime;
    gpuErrchk( cudaEventElapsedTime(&totalCycleTime, *start, *stop ) );
    totalCycleTime /= 1000.0;
    wallTime += totalCycleTime;
    printCycleTime(cpuCycleTime, gpuCycleTime , totalCycleTime);
#else
    printCycleTime(cpuCycleTime, cpuCycleTime , cpuCycleTime);
#endif

}

template<typename Geometry, unsigned N>
void
RayListController<Geometry,N>::swapBanks(){
    if( PA.getWorkGroupRank() != 0 ) { return; }

    // Swap banks
    if( currentBank == bank1.get() ) {
        currentBank = bank2.get();
#ifdef __CUDACC__
        currentCopySync = copySync2.get();
#endif
    } else {
        currentBank = bank1.get();
#ifdef __CUDACC__
        currentCopySync = copySync1.get();
#endif
    }

#ifdef __CUDACC__
    cudaEventSynchronize( *currentCopySync );
#endif
    currentBank->clear();
}

template<typename Geometry, unsigned N>
void
RayListController<Geometry,N>::sync(void){
    if( PA.getWorkGroupRank() != 0 ) { return; }
    defaultStreamSync();
}

template<typename Geometry, unsigned N>
void
RayListController<Geometry,N>::clearTally(void) {
    if( PA.getWorkGroupRank() != 0 ) { return; }

    std::cout << "Debug: clearTally called \n";

    if( nFlushs > 0 ) {
        stopTimers();
    }
    //	std::cout << "Debug: clearTally nFlushs =" <<nFlushs << " -- starting timers\n";
    //	startTimers();
    //
    //	++nFlushs;
    //
    //	cudaEventRecord( stopGPU.get(), stream1.get());
    //	cudaStreamWaitEvent(stream1.get(), stopGPU.get(), 0);

    defaultStreamSync();
    if( pTally ) pTally->clear();
    if( bank1 ) bank1->clear();
    if( bank2) bank2->clear();
    defaultStreamSync();
}

template<typename Geometry, unsigned N>
void
RayListController<Geometry,N>::printTotalTime() const{
    std::cout << "Debug: \n";
    std::cout << "Debug: total gpuTime = " << gpuTime << "\n";
    std::cout << "Debug: total cpuTime = " << cpuTime << "\n";
    std::cout << "Debug: total wallTime = " << wallTime << "\n";
}

template<typename Geometry, unsigned N>
void
RayListController<Geometry,N>::printCycleTime(float_t cpu, float_t gpu, float_t wall) const{
    std::cout << "Debug: \n";
    std::cout << "Debug: cycle gpuTime = " << gpu << "\n";
    std::cout << "Debug: cycle cpuTime = " << cpu << "\n";
    std::cout << "Debug: cycle wallTime = " << wall << "\n";
}

template<typename Geometry, unsigned N>
void
RayListController<Geometry,N>::printPointDets( const std::string& outputFile, unsigned nSamples, unsigned constantDimension) {
    if( PA.getWorldRank() != 0 ) { return; }

    if( ! usingNextEventEstimator ) {
        throw std::runtime_error( "RayListController::printPointDets  -- only supports printing of Next-Event Estimators." );
    }

    pNextEventEstimator->printPointDets(outputFile, nSamples, constantDimension );
}

template<typename Geometry, unsigned N>
void
RayListController<Geometry,N>::outputTimeBinnedTotal(std::ostream& out,unsigned nSamples, unsigned constantDimension){
    if( PA.getWorldRank() != 0 ) { return; }

    if( ! usingNextEventEstimator ) {
        throw std::runtime_error( "RayListController::outputTimeBinnedTotal  -- only supports outputting Next-Event Estimator results." );
    }
    pNextEventEstimator->outputTimeBinnedTotal(out, nSamples, constantDimension );
}

template<typename Geometry, unsigned N>
void
RayListController<Geometry,N>::updateMaterialProperties( MaterialProperties* pMPs) {
    if( PA.getWorkGroupRank() != 0 ) { return; }

    pMatProps = pMPs;
}

template<typename Geometry, unsigned N>
unsigned
RayListController<Geometry,N>::addPointDet( gpuFloatType_t x, gpuFloatType_t y, gpuFloatType_t z ){
    if( PA.getWorkGroupRank() != 0 ) { return 0; }
    if( ! isUsingNextEventEstimator() ) {
        throw std::runtime_error( "RayListController::addPointDet - Next-Event Estimator not enabled." );
    }
    return pNextEventEstimatorBuilder->add( x, y, z );
}

template<typename Geometry, unsigned N>
void
RayListController<Geometry,N>::setPointDetExclusionRadius(gpuFloatType_t r){
    if( PA.getWorkGroupRank() != 0 ) { return; }
    if( ! isUsingNextEventEstimator() ) {
        throw std::runtime_error( "RayListController::setPointDetExclusionRadius - Next-Event Estimator not enabled." );
    }
    pNextEventEstimatorBuilder->setExclusionRadius( r );
}

template<typename Geometry, unsigned N>
void
RayListController<Geometry,N>::copyPointDetTallyToCPU(void) {
    if( PA.getWorkGroupRank() != 0 ) { return; }

    if( ! isUsingNextEventEstimator() ) {
        throw std::runtime_error( "RayListController::copyPointDetTallyToCPU - Next-Event Estimator not enabled." );
    }
#ifdef __CUDACC__
    cudaDeviceSynchronize();
#endif
}

template<typename Geometry, unsigned N>
gpuTallyType_t
RayListController<Geometry,N>::getPointDetTally(unsigned spatialIndex, unsigned timeIndex ) const {
    if( PA.getWorldRank() != 0 ) { return 0.0; }

    if( ! isUsingNextEventEstimator() ) {
        throw std::runtime_error( "RayListController::getPointDetTally - Next-Event Estimator not enabled." );
    }
    return pNextEventEstimator->getTally(spatialIndex, timeIndex);
}

template<typename Geometry, unsigned N>
void
RayListController<Geometry,N>::copyPointDetToGPU(void) {

    if( ! isUsingNextEventEstimator() ) {
        throw std::runtime_error( "RayListController::getPointDetTally - Next-Event Estimator not enabled." );
    }

    if( ! tallyInitialized ) {
        tallyInitialized = true;
    } else {
        return;
    }

    pNextEventEstimatorBuilder->setTimeBinEdges( TallyTimeBinEdges );
    pNextEventEstimator = std::make_unique<NextEventEstimator>(pNextEventEstimatorBuilder->build());
#ifdef __CUDACC__
    cudaDeviceSynchronize();
#endif
}

template<typename Geometry, unsigned N>
void RayListController<Geometry,N>::dumpPointDetForDebug(const std::string& fileName ) {
    // ensures setup - the copy to the GPU is not used
    copyPointDetToGPU();
    writeToFile(fileName, *pNextEventEstimator);
}

template<typename Geometry, unsigned N>
void
RayListController<Geometry,N>::gather() {
    if( pNextEventEstimator ) {
        pNextEventEstimator->gather();
    }
}

} // end namespace MonteRay

#endif // RAYLISTCONTROLLER_T_HH
