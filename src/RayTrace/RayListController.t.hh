#ifndef RAYLISTCONTROLLER_T_HH_
#define RAYLISTCONTROLLER_T_HH_

#include <algorithm>

#include "RayListController.hh"

#include "MonteRayMaterialList.hh"
#include "MonteRay_MaterialProperties.hh"
#include "gpuTally.hh"
#include "RayListInterface.hh"
#include "ExpectedPathLength.t.hh"
#include "GPUErrorCheck.hh"
#include "GPUSync.hh"
#include "MonteRayNextEventEstimator.t.hh"
#include "MonteRay_timer.hh"

namespace MonteRay {

template<typename GRID_T, unsigned N>
RayListController<GRID_T,N>::RayListController(
        unsigned blocks,
        unsigned threads,
        GRID_T* pGB,
        MonteRayMaterialListHost* pML,
        MonteRay_MaterialProperties* pMP,
        gpuTallyHost* pT
) :
nBlocks(blocks),
nThreads(threads),
pGrid( pGB ),
pMatList( pML ),
pMatProps( pMP ),
pTally(pT),
PA( MonteRayParallelAssistant::getInstance() )
{
    pNextEventEstimator.reset();
    initialize();
    kernel = [&] ( void ) {
        if( PA.getWorkGroupRank() != 0 ) { return; }
#ifdef __CUDACC__
        rayTraceTally<<<nBlocks,nThreads,0, *stream1>>>(
                pGrid->getDevicePtr(),
                currentBank->getPtrPoints()->devicePtr, pMatList->ptr_device,
                pMatProps->ptrData_device, pMatList->getHashPtr()->getPtrDevice(),
                pTally->temp->tally );
#else
        rayTraceTally( pGrid->getPtr(),
                       currentBank->getPtrPoints(), pMatList->getPtr(),
                       pMatProps->getPtr(), pMatList->getHashPtr()->getPtr(),
                       pTally->getPtr()->tally );
#endif
    };

}

template<typename GRID_T, unsigned N>
RayListController<GRID_T,N>::RayListController(
        unsigned blocks,
        unsigned threads,
        GRID_T* pGB,
        MonteRayMaterialListHost* pML,
        MonteRay_MaterialProperties* pMP,
        unsigned numPointDets
) :
nBlocks(blocks),
nThreads(threads),
pGrid( pGB ),
pMatList( pML ),
pMatProps( pMP ),
pTally(NULL),
PA( MonteRayParallelAssistant::getInstance() )
{
    pNextEventEstimator = std::make_shared<MonteRayNextEventEstimator<GRID_T>>( numPointDets );
    usingNextEventEstimator = true;
    initialize();
    kernel = [&] ( void ) {
#ifdef DEBUG
        const bool debug = false;
#endif

        //copyPointDetToGPU();

        if( currentBank->size() > 0 ) {
            //if( PA.getWorkGroupRank() != 0 ) { return; }
#ifdef DEBUG
            if( debug ) std::cout << "Debug: RayListController::kernel() -- Next Event Estimator kernel. Calling pNextEventEstimator->launch_ScoreRayList.\n";
#endif
            pNextEventEstimator->launch_ScoreRayList(nBlocks,nThreads, currentBank->getPtrPoints(), stream1.get() );
        }
    };
}

template<typename GRID_T, unsigned N>
RayListController<GRID_T,N>::RayListController( unsigned numPointDets, const std::string& filename ) :
PA( MonteRayParallelAssistant::getInstance() )
{
    initialize();
    pNextEventEstimator = std::make_shared<MonteRayNextEventEstimator<GRID_T>>( numPointDets );
    setOutputFileName( filename );
    usingNextEventEstimator = true;
    kernel = [&] ( void ) {
        // do nothing
        return;
    };
}

template<typename GRID_T, unsigned N>
void
RayListController<GRID_T, N>::initialize(){
    if( PA.getWorkGroupRank() != 0 ) { return; }

    setCapacity( 1000000 ); // default buffer capacity to 1 million.

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

template<typename GRID_T, unsigned N>
RayListController<GRID_T,N>::~RayListController(){

#ifdef __CUDACC__
    if( stream1 ) cudaStreamDestroy( *stream1 );
#endif
}

template<typename GRID_T, unsigned N>
unsigned
RayListController<GRID_T,N>::capacity(void) const {
    if(currentBank) return currentBank->capacity();
    return 0;
}

template<typename GRID_T, unsigned N>
unsigned
RayListController<GRID_T,N>::size(void) const {
    if(currentBank) return currentBank->size();
    return 0;
}

template<typename GRID_T, unsigned N>
void
RayListController<GRID_T,N>::setCapacity(unsigned n) {
    if( PA.getWorkGroupRank() != 0 ) { return; }
    bank1.reset( new RayListInterface<N>(n) );
    bank2.reset( new RayListInterface<N>(n) );
    currentBank = bank1.get();
}


template<typename GRID_T, unsigned N>
void
RayListController<GRID_T,N>::add( const Ray_t<N>& ray){
    if( PA.getWorkGroupRank() != 0 ) { return; }

    currentBank->add( ray );
    if( size() == capacity() ) {
        std::cout << "Debug: bank full, flushing.\n";
        flush();
    }
}

template<typename GRID_T, unsigned N>
unsigned
RayListController<GRID_T,N>::getWorldRank() {
    return PA.getWorldRank();
}

template<typename GRID_T, unsigned N>
void
RayListController<GRID_T,N>::add( const Ray_t<N>* rayArray, unsigned num){
    if( PA.getWorkGroupRank() != 0 ) { return; }

    int NSpaces = capacity() - size();

    int NAdding = std::min(NSpaces, int(num));
    int NRemaining = num - NAdding;
    currentBank->add( rayArray, NAdding );
    if( size() == capacity() ) {
        std::cout << "Debug: bank full, flushing.\n";
        flush();
    }
    if( NRemaining > 0 ) {
        add( rayArray + NAdding, NRemaining );
    }
}

template<typename GRID_T, unsigned N>
void
RayListController<GRID_T,N>::flush(bool final){
    if( PA.getWorkGroupRank() != 0 ) { return; }

#ifdef DEBUG
    const bool debug = false;
    if( debug ) std::cout << "Debug: RayListController<N>::flush\n";
#endif

    if( isSendingToFile() ) { flushToFile(final); }

    if( currentBank->size() == 0 ) {
        if( final ) {
            printTotalTime();
            currentBank->clear();
        }
        return;
    }

    if( nFlushs > 0 ) {
        std::cout << "Debug: flush nFlushs =" <<nFlushs-1 << " -- stopping timers\n";
        stopTimers();
    }
    std::cout << "Debug: flush nFlushs =" <<nFlushs << " -- starting timers\n";

    startTimers();

    ++nFlushs;

#ifdef __CUDACC__
    currentBank->copyToGPU();
    gpuErrchk( cudaEventRecord(*currentCopySync, 0) );
    gpuErrchk( cudaEventSynchronize(*currentCopySync) );
#endif

    // launch kernel
    kernel();

    // only uncomment for testing, forces the cpu and gpu to sync
    //gpuErrchk( cudaPeekAtLastError() );

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

template<typename GRID_T, unsigned N>
void
RayListController<GRID_T,N>::flushToFile(bool final){
    if( PA.getWorldRank() != 0 ) { return; }

#ifdef DEBUG
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
#ifdef DEBUG
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
#ifdef DEBUG
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
#ifdef DEBUG
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

template<typename GRID_T, unsigned N>
size_t
RayListController<GRID_T,N>::readCollisionsFromFile(std::string name) {
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

template<typename GRID_T, unsigned N>
size_t
RayListController<GRID_T,N>::readCollisionsFromFileToBuffer(std::string name){
    if( PA.getWorldRank() != 0 ) { return 0; }

    unsigned numParticles = 0;
    currentBank->readToBank(name, numParticles);
    numParticles += currentBank->size();
    return numParticles;
}

template<typename GRID_T, unsigned N>
void
RayListController<GRID_T,N>::startTimers(){
    // start timers
    if( PA.getWorkGroupRank() != 0 ) { return; }

    pTimer->start();
#ifdef __CUDACC__
    gpuErrchk( cudaEventRecord( *start,0) );
    gpuErrchk( cudaEventRecord( *startGPU, *stream1) );
#endif
}

template<typename GRID_T, unsigned N>
void
RayListController<GRID_T,N>::stopTimers(){
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

template<typename GRID_T, unsigned N>
void
RayListController<GRID_T,N>::swapBanks(){
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

template<typename GRID_T, unsigned N>
void
RayListController<GRID_T,N>::sync(void){
    if( PA.getWorkGroupRank() != 0 ) { return; }

    GPUSync sync;
    sync.sync();
}

template<typename GRID_T, unsigned N>
void
RayListController<GRID_T,N>::clearTally(void) {
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

    GPUSync sync;
    if( pTally ) pTally->clear();
    if( bank1 ) bank1->clear();
    if( bank2) bank2->clear();
    sync.sync();
}

template<typename GRID_T, unsigned N>
void
RayListController<GRID_T,N>::printTotalTime() const{
    std::cout << "Debug: \n";
    std::cout << "Debug: total gpuTime = " << gpuTime << "\n";
    std::cout << "Debug: total cpuTime = " << cpuTime << "\n";
    std::cout << "Debug: total wallTime = " << wallTime << "\n";
}

template<typename GRID_T, unsigned N>
void
RayListController<GRID_T,N>::printCycleTime(float_t cpu, float_t gpu, float_t wall) const{
    std::cout << "Debug: \n";
    std::cout << "Debug: cycle gpuTime = " << gpu << "\n";
    std::cout << "Debug: cycle cpuTime = " << cpu << "\n";
    std::cout << "Debug: cycle wallTime = " << wall << "\n";
}

template<typename GRID_T, unsigned N>
void
RayListController<GRID_T,N>::printPointDets( const std::string& outputFile, unsigned nSamples, unsigned constantDimension) {
    if( PA.getWorldRank() != 0 ) { return; }

    if( ! usingNextEventEstimator ) {
        throw std::runtime_error( "RayListController::printPointDets  -- only supports printing of Next-Event Estimators." );
    }

    pNextEventEstimator->printPointDets(outputFile, nSamples, constantDimension );
}

template<typename GRID_T, unsigned N>
void
RayListController<GRID_T,N>::outputTimeBinnedTotal(std::ostream& out,unsigned nSamples, unsigned constantDimension){
    if( PA.getWorldRank() != 0 ) { return; }

    if( ! usingNextEventEstimator ) {
        throw std::runtime_error( "RayListController::outputTimeBinnedTotal  -- only supports outputting Next-Event Estimator results." );
    }
    pNextEventEstimator->outputTimeBinnedTotal(out, nSamples, constantDimension );
}

template<typename GRID_T, unsigned N>
unsigned
RayListController<GRID_T,N>::addPointDet( gpuFloatType_t x, gpuFloatType_t y, gpuFloatType_t z ){
    if( PA.getWorkGroupRank() != 0 ) { return 0; }
    if( ! isUsingNextEventEstimator() ) {
        throw std::runtime_error( "RayListController::addPointDet - Next-Event Estimator not enabled." );
    }
    return pNextEventEstimator->add( x, y, z );
}

template<typename GRID_T, unsigned N>
void
RayListController<GRID_T,N>::setPointDetExclusionRadius(gpuFloatType_t r){
    if( PA.getWorkGroupRank() != 0 ) { return; }
    if( ! isUsingNextEventEstimator() ) {
        throw std::runtime_error( "RayListController::setPointDetExclusionRadius - Next-Event Estimator not enabled." );
    }
    pNextEventEstimator->setExclusionRadius( r );
}

template<typename GRID_T, unsigned N>
void
RayListController<GRID_T,N>::copyPointDetTallyToCPU(void) {
    if( PA.getWorkGroupRank() != 0 ) { return; }

    if( ! isUsingNextEventEstimator() ) {
        throw std::runtime_error( "RayListController::copyPointDetTallyToCPU - Next-Event Estimator not enabled." );
    }
    pNextEventEstimator->copyToCPU();
}

template<typename GRID_T, unsigned N>
gpuTallyType_t
RayListController<GRID_T,N>::getPointDetTally(unsigned spatialIndex, unsigned timeIndex ) const {
    if( PA.getWorldRank() != 0 ) { return 0.0; }

    if( ! isUsingNextEventEstimator() ) {
        throw std::runtime_error( "RayListController::getPointDetTally - Next-Event Estimator not enabled." );
    }
    return pNextEventEstimator->getTally(spatialIndex, timeIndex);
}

template<typename GRID_T, unsigned N>
void
RayListController<GRID_T,N>::copyPointDetToGPU(void) {

    if( ! isUsingNextEventEstimator() ) {
        throw std::runtime_error( "RayListController::getPointDetTally - Next-Event Estimator not enabled." );
    }

    if( ! tallyInitialized ) {
        tallyInitialized = true;
    } else {
        return;
    }

    pNextEventEstimator->setGeometry( pGrid, pMatProps );
    pNextEventEstimator->setMaterialList( pMatList );
    pNextEventEstimator->setTimeBinEdges( TallyTimeBinEdges );
    pNextEventEstimator->initialize();
    pNextEventEstimator->copyToGPU();
}

template<typename GRID_T, unsigned N>
void
RayListController<GRID_T,N>::dumpPointDetForDebug(const std::string& baseFileName ) {
    // ensures setup - the copy to the GPU is not used
    copyPointDetToGPU();
    pNextEventEstimator->dumpState(currentBank->getPtrPoints(), baseFileName);
}

template<typename GRID_T, unsigned N>
void
RayListController<GRID_T,N>::gather() {
    if( pNextEventEstimator ) {
        pNextEventEstimator->gather();
    }
}


} /* namespace MonteRay */


#endif /* RAYLISTCONTROLLER_T_HH_ */
