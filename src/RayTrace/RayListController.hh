#ifndef RAYLISTCONTROLLER_HH_
#define RAYLISTCONTROLLER_HH_

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <iostream>
#include <algorithm>

#include "MonteRayMaterialList.hh"
#include "MonteRay_MaterialProperties.hh"
#include "gpuTally.hh"
#include "RayListInterface.hh"
#include "ExpectedPathLength.hh"
#include "GPUErrorCheck.hh"
#include "GPUSync.hh"
#include "MonteRayNextEventEstimator.hh"
#include "MonteRay_timer.hh"
#include "RayWorkInfo.hh"
#include "MonteRayTypes.hh"
#include "MonteRayParallelAssistant.hh"

namespace MonteRay {

class MonteRayMaterialListHost;
class MonteRay_MaterialProperties;
class gpuTallyHost;
class cpuTimer;

template<typename GRID_T>
class MonteRayNextEventEstimator;

template< unsigned N >
class RayListInterface;

template< unsigned N >
class Ray_t;

class RayWorkInfo;

template<typename GRID_T, unsigned N = 1>
class RayListController {
public:

    /// Ctor for the volumetric ray casting solver
    RayListController(int nBlocks,
            int nThreads,
            GRID_T*,
            MonteRayMaterialListHost*,
            MonteRay_MaterialProperties*,
            gpuTallyHost* );

    /// Ctor for the next event estimator solver
    RayListController(int nBlocks,
            int nThreads,
            GRID_T*,
            MonteRayMaterialListHost*,
            MonteRay_MaterialProperties*,
            unsigned numPointDets );

    /// Ctor for the writing next-event estimator collision and source points to file
    /// Can not launch a kernel
    RayListController( unsigned numPointDets, const std::string& filename );

    ~RayListController();

    void initialize();
    unsigned capacity(void) const;
    unsigned size(void) const;
    void setCapacity(unsigned n );

    void add( const Ray_t<N>& ray);
    void add( const Ray_t<N>* rayArray, unsigned num=1 );
    void add( const void* ray, unsigned num=1 ) { add(  (const Ray_t<N>*) ray, num  ); }

    unsigned addPointDet( gpuFloatType_t x, gpuFloatType_t y, gpuFloatType_t z );
    void setPointDetExclusionRadius(gpuFloatType_t r);
    void copyPointDetTallyToCPU(void);
    gpuTallyType_t getPointDetTally(unsigned spatialIndex, unsigned timeIndex=0 ) const;
    void copyPointDetToGPU(void);
    void dumpPointDetForDebug(const std::string& baseFileName = std::string() );
    void printPointDets( const std::string& outputFile, unsigned nSamples, unsigned constantDimension=2);
    void outputTimeBinnedTotal(std::ostream& out,unsigned nSamples=1, unsigned constantDimension=2);
    CUDAHOST_CALLABLE_MEMBER void updateMaterialProperties( MonteRay_MaterialProperties* pMPs);

    void flush(bool final=false);
    void finalFlush(void);
    void stopTimers(void);
    void startTimers(void);
    void swapBanks(void);

    void printCycleTime(float_t cpu, float_t gpu, float_t wall) const;
    void printTotalTime(void) const;

    double getCPUTime(void) const { return cpuTime; }
    double getGPUTime(void) const { return gpuTime; }
    unsigned getNFlushes(void) const { return nFlushs; }

    void sync(void);

    void clearTally(void);

    bool isSendingToFile(void) { return toFile; }

    void setOutputFileName(std::string name) {
        outputFileName = name;
        sendToFile();
    }

    size_t readCollisionsFromFile(std::string name);

    // reads a single block of rays to a buffer but doesn't flush them
    // usually for testing or debugging
    size_t readCollisionsFromFileToBuffer(std::string name);

    void flushToFile(bool final=false);

    void debugPrint() {
        currentBank->debugPrint();
    }

    bool isUsingNextEventEstimator(void) const {
        return usingNextEventEstimator;
    }

    template<typename T>
    void setTimeBinEdges( const std::vector<T>& edges) {
        TallyTimeBinEdges.assign( edges.begin(), edges.end() );
    }

    unsigned getWorldRank();

    template<typename T>
    void setEnergyBinEdges( const std::vector<T>& edges) {
        if( getWorldRank() == 0 ) {
            std::cout << "WARNING:  MonteRay does not currently support energy binned tallies.  Ignoring.\n";
        }
    }

    void gather();

private:
    unsigned nBlocks = 0;
    unsigned nThreads = 0;
    GRID_T* pGrid = nullptr;
    MonteRayMaterialListHost* pMatList = nullptr;
    MonteRay_MaterialProperties* pMatProps = nullptr;
    gpuTallyHost* pTally = nullptr;
    const MonteRayParallelAssistant& PA;

    std::shared_ptr<MonteRayNextEventEstimator<GRID_T>> pNextEventEstimator;
    std::vector<gpuFloatType_t> TallyTimeBinEdges;

    RayListInterface<N>* currentBank = nullptr;
    std::unique_ptr<RayListInterface<N>> bank1;
    std::unique_ptr<RayListInterface<N>> bank2;

    std::unique_ptr<RayWorkInfo> rayInfo;

    unsigned nFlushs = 0;

    std::unique_ptr<cpuTimer> pTimer;
    double cpuTime = 0.0;
    double gpuTime = 0.0;
    double wallTime= 0.0;
    bool toFile = false;
    bool fileIsOpen = false;
    bool tallyInitialized = false;

    std::string outputFileName;

    void sendToFile(void) { toFile = true; }

    bool usingNextEventEstimator = false;

    typedef std::function<void ( void )> kernel_t;
    kernel_t kernel;

    std::unique_ptr<cudaStream_t> stream1;
    std::unique_ptr<cudaEvent_t> startGPU;
    std::unique_ptr<cudaEvent_t> stopGPU;
    std::unique_ptr<cudaEvent_t> start;
    std::unique_ptr<cudaEvent_t> stop;
    std::unique_ptr<cudaEvent_t> copySync1;
    std::unique_ptr<cudaEvent_t> copySync2;
    cudaEvent_t* currentCopySync = nullptr;
};

template<class GRID_T>
using CollisionPointController = typename MonteRay::RayListController<GRID_T,1>;

template<class GRID_T>
using NextEventEstimatorController = typename MonteRay::RayListController<GRID_T,3>;

} /* namespace MonteRay */

// Begin RayListController.t.hh
//
namespace MonteRay {

template<typename GRID_T, unsigned N>
RayListController<GRID_T,N>::RayListController(
        int blocks,
        int threads,
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
    kernel = [&, blocks, threads ] ( void ) {
        if( PA.getWorkGroupRank() != 0 ) { return; }

        auto launchBounds = setLaunchBounds( threads, blocks,  currentBank->getPtrPoints()->size() );

#ifdef DEBUG
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

#ifdef __CUDACC__

        rayTraceTally<<<launchBounds.first,launchBounds.second,0, *stream1>>>(
                pGrid->getDevicePtr(),
                currentBank->getPtrPoints()->devicePtr,
                pMatList->ptr_device,
                pMatProps->ptrData_device,
                pMatList->getHashPtr()->getPtrDevice(),
                rayInfo->devicePtr,
                pTally->temp->tally );
#else
        rayTraceTally( pGrid->getPtr(),
                       currentBank->getPtrPoints(),
                       pMatList->getPtr(),
                       pMatProps->getPtr(),
                       pMatList->getHashPtr()->getPtr(),
                       rayInfo.get(),
                       pTally->getPtr()->tally );
#endif
    };

}

template<typename GRID_T, unsigned N>
RayListController<GRID_T,N>::RayListController(
        int blocks,
        int threads,
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
    kernel = [&, blocks, threads] ( void ) {
      if( currentBank->size() > 0 ) {
        pNextEventEstimator->launch_ScoreRayList(blocks,threads, currentBank->getPtrPoints(), rayInfo.get(), stream1.get() );
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
#ifdef __CUDACC__
    rayInfo.reset( new RayWorkInfo( totalNumThreads ) );
#else
    // allocate on the CPU
    rayInfo.reset( new RayWorkInfo( 1, true ) );
#endif
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
    gpuErrchk( cudaPeekAtLastError() );
    currentBank->copyToGPU();
    rayInfo->copyToGPU();
    gpuErrchk( cudaEventRecord(*currentCopySync, 0) );
    gpuErrchk( cudaEventSynchronize(*currentCopySync) );
#endif

    // launch kernel
    kernel();

    // only uncomment for testing, forces the cpu and gpu to sync
#ifdef DEBUG
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
void
RayListController<GRID_T,N>::updateMaterialProperties( MonteRay_MaterialProperties* pMPs) {
    if( PA.getWorkGroupRank() != 0 ) { return; }

    if( usingNextEventEstimator ) {
        pNextEventEstimator->updateMaterialProperties( pMPs );
    }
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
void RayListController<GRID_T,N>::dumpPointDetForDebug(const std::string& baseFileName ) {
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

} // end namespace MonteRay

#endif /* RAYLISTCONTROLLER_HH_ */
