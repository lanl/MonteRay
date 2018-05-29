#include "RayListController.hh"

#include <algorithm>

#include "GridBins.hh"
#include "MonteRayMaterialList.hh"
#include "MonteRay_MaterialProperties.hh"
#include "gpuTally.hh"
#include "RayListInterface.hh"
#include "ExpectedPathLength.hh"
#include "GPUErrorCheck.hh"
#include "GPUSync.hh"
#include "MonteRayNextEventEstimator.hh"

namespace MonteRay {

template<unsigned N >
RayListController<N>::RayListController(
		unsigned blocks,
        unsigned threads,
        GridBinsHost* pGB,
        MonteRayMaterialListHost* pML,
        MonteRay_MaterialProperties* pMP,
        gpuTallyHost* pT
	) :
        nBlocks(blocks),
        nThreads(threads),
        pGrid( pGB ),
        pMatList( pML ),
        pMatProps( pMP ),
        pTally(pT)
{
	pNextEventEstimator.reset();
	initialize();
	kernel = [&] ( void ) {
#ifdef __CUDACC__
		rayTraceTally<<<nBlocks,nThreads,0,stream1>>>(pGrid->ptr_device,
				currentBank->getPtrPoints()->devicePtr, pMatList->ptr_device,
				pMatProps->ptrData_device, pMatList->getHashPtr()->getPtrDevice(),
				pTally->temp->tally);
#else
		rayTraceTally(pGrid->getPtr(),
						currentBank->getPtrPoints(), pMatList->getPtr(),
						pMatProps->getPtr(), pMatList->getHashPtr()->getPtr(),
						pTally->getPtr()->tally );
#endif
	};

}

template<unsigned N >
RayListController<N>::RayListController(
		unsigned blocks,
        unsigned threads,
        GridBinsHost* pGB,
        MonteRayMaterialListHost* pML,
        MonteRay_MaterialProperties* pMP,
        unsigned numPointDets
	) :
        nBlocks(blocks),
        nThreads(threads),
        pGrid( pGB ),
        pMatList( pML ),
        pMatProps( pMP ),
        pTally(NULL)
{
	pNextEventEstimator = std::make_shared<MonteRayNextEventEstimator>( numPointDets );
	usingNextEventEstimator = true;
	initialize();
	kernel = [&] ( void ) {
		const bool debug = false;
		if( currentBank->size() > 0 ) {
			if( debug ) std::cout << "Debug: RayListController::kernel() -- Next Event Estimator kernel. Calling pNextEventEstimator->launch_ScoreRayList.\n";
#ifdef __CUDACC__
			pNextEventEstimator->launch_ScoreRayList(nBlocks,nThreads,stream1, currentBank->getPtrPoints());
#else
			pNextEventEstimator->launch_ScoreRayList(nBlocks,nThreads,currentBank->getPtrPoints());
#endif
		}
	};
}

template<unsigned N >
RayListController<N>::RayListController( unsigned numPointDets, std::string filename ) :
        nBlocks(0),
        nThreads(0),
        pGrid( NULL ),
        pMatList( NULL ),
        pMatProps( NULL ),
        pTally(NULL)
{
	initialize();
	pNextEventEstimator = std::make_shared<MonteRayNextEventEstimator>( numPointDets );
	setOutputFileName( filename );
	usingNextEventEstimator = true;
	kernel = [&] ( void ) {
		// do nothing
		return;
	};
}

template<unsigned N >
void
RayListController<N>::initialize(){
	nFlushs = 0;
	cpuTime = 0.0;
	gpuTime = 0.0;
	wallTime = 0.0;
	toFile = false;
	fileIsOpen = false;
	bank1 = new RayListInterface<N>(1000000); // default 1 millions
	bank2 = new RayListInterface<N>(1000000); // default 1 millions

#ifdef __CUDACC__
	cudaStreamCreate( &stream1 );
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	cudaEventCreate(&copySync1);
	cudaEventCreate(&copySync2);
#endif

	currentBank = bank1;

#ifdef __CUDACC__
	currentCopySync = &copySync1;
#endif
}

template<unsigned N >
RayListController<N>::~RayListController(){
	delete bank1;
	delete bank2;

#ifdef __CUDACC
	cudaStreamDestroy(stream1);
#endif
}

template<unsigned N >
unsigned
RayListController<N>::capacity(void) const {
	return currentBank->capacity();
}

template<unsigned N >
unsigned
RayListController<N>::size(void) const {
	return currentBank->size();
}

template<unsigned N >
void
RayListController<N>::setCapacity(unsigned n) {
	delete bank1;
	delete bank2;
	bank1 = new RayListInterface<N>(n);
	bank2 = new RayListInterface<N>(n);
	currentBank = bank1;
}

template<unsigned N >
void
RayListController<N>::add( const Ray_t<N>& ray){
	currentBank->add( ray );
	if( size() == capacity() ) {
		std::cout << "Debug: bank full, flushing.\n";
		flush();
	}
}

template<unsigned N >
void
RayListController<N>::add( const Ray_t<N>* rayArray, unsigned num){
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

template<unsigned N >
void
RayListController<N>::flush(bool final){
	const bool debug = false;
	if( debug ) std::cout << "Debug: RayListController<N>::flush\n";

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
	gpuErrchk( cudaEventRecord(stopGPU,stream1) );
	gpuErrchk( cudaStreamWaitEvent(stream1, stopGPU, 0) );
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

template<unsigned N >
void
RayListController<N>::flushToFile(bool final){
	const bool debug = false;

	if( debug ) {
		if( final ) {
			std::cout << "Debug: RayListController::flushToFile - starting -- final = true \n";
		} else {
			std::cout << "Debug: RayListController::flushToFile - starting -- final = false \n";
		}
	}

	if( ! fileIsOpen ) {
		try {
			if( debug ) std::cout << "Debug: RayListController::flushToFile - opening file, filename=" << outputFileName << "\n";
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
		if( debug )  std::cout << "Debug: RayListController::flushToFile - writing bank -- bank size = "<< currentBank->size() << "\n";
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
			if( debug ) std::cout << "Debug: RayListController::flushToFile - file flush, closing collision file\n";
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

template<unsigned N >
size_t
RayListController<N>::readCollisionsFromFile(std::string name) {

	bool end = false;
	unsigned numParticles = 0;
	do  {
		end = currentBank->readToBank(name, numParticles);
		numParticles += currentBank->size();
		flush(end);
	} while ( ! end );
	return numParticles;
}

template<unsigned N >
void
RayListController<N>::startTimers(){
	// start timers
	timer.start();
#ifdef __CUDACC__
	gpuErrchk( cudaEventRecord(start,0) );
	gpuErrchk( cudaEventRecord(startGPU,stream1) );
#endif
}

template<unsigned N >
void
RayListController<N>::stopTimers(){
	// stop timers and sync

	timer.stop();
	float_t cpuCycleTime = timer.getTime();
	cpuTime += cpuCycleTime;

#ifdef __CUDACC__
	gpuErrchk( cudaStreamSynchronize( stream1 ) );
	gpuErrchk( cudaEventRecord(stop, 0) );
	gpuErrchk( cudaEventSynchronize(stop) );

	float_t gpuCycleTime;
	gpuErrchk( cudaEventElapsedTime(&gpuCycleTime, startGPU, stopGPU ) );
	gpuCycleTime /= 1000.0;
	if( gpuCycleTime < 0.0 ) {
		gpuCycleTime = 0.0;
	}
	gpuTime += gpuCycleTime;

	float totalCycleTime;
	gpuErrchk( cudaEventElapsedTime(&totalCycleTime, start, stop ) );
	totalCycleTime /= 1000.0;
	wallTime += totalCycleTime;
	printCycleTime(cpuCycleTime, gpuCycleTime , totalCycleTime);
#else
	printCycleTime(cpuCycleTime, cpuCycleTime , cpuCycleTime);
#endif

}

template<unsigned N >
void
RayListController<N>::swapBanks(){
	// Swap banks
	if( currentBank == bank1 ) {
		currentBank = bank2;
#ifdef __CUDACC__
		currentCopySync = &copySync2;
#endif
	} else {
		currentBank = bank1;
#ifdef __CUDACC__
		currentCopySync = &copySync1;
#endif
	}

#ifdef __CUDACC__
	cudaEventSynchronize(*currentCopySync);
#endif
	currentBank->clear();
}

template<unsigned N >
void
RayListController<N>::sync(void){
	GPUSync sync;
	sync.sync();
}

template<unsigned N >
void
RayListController<N>::clearTally(void) {

	std::cout << "Debug: clearTally called \n";

	if( nFlushs > 0 ) {
		stopTimers();
	}
//	std::cout << "Debug: clearTally nFlushs =" <<nFlushs << " -- starting timers\n";
//	startTimers();
//
//	++nFlushs;
//
//	cudaEventRecord(stopGPU,stream1);
//	cudaStreamWaitEvent(stream1, stopGPU, 0);

	GPUSync sync;
	pTally->clear();
	bank1->clear();
	bank2->clear();
	sync.sync();
}

template<unsigned N >
void
RayListController<N>::printTotalTime() const{
	std::cout << "Debug: \n";
	std::cout << "Debug: total gpuTime = " << gpuTime << "\n";
	std::cout << "Debug: total cpuTime = " << cpuTime << "\n";
	std::cout << "Debug: total wallTime = " << wallTime << "\n";
}

template<unsigned N >
void
RayListController<N>::printCycleTime(float_t cpu, float_t gpu, float_t wall) const{
	std::cout << "Debug: \n";
	std::cout << "Debug: cycle gpuTime = " << gpu << "\n";
	std::cout << "Debug: cycle cpuTime = " << cpu << "\n";
	std::cout << "Debug: cycle wallTime = " << wall << "\n";
}

template<unsigned N >
unsigned
RayListController<N>::addPointDet( gpuFloatType_t x, gpuFloatType_t y, gpuFloatType_t z ){
	if( ! isUsingNextEventEstimator() ) {
		throw std::runtime_error( "RayListController::addPointDet - Next Event Estimator not enabled." );
	}
	return pNextEventEstimator->add( x, y, z );
}

template<unsigned N >
void
RayListController<N>::setPointDetExclusionRadius(gpuFloatType_t r){
	if( ! isUsingNextEventEstimator() ) {
		throw std::runtime_error( "RayListController::setPointDetExclusionRadius - Next Event Estimator not enabled." );
	}
	pNextEventEstimator->setExclusionRadius( r );
}

template<unsigned N >
void
RayListController<N>::copyPointDetTallyToCPU(void) {
	if( ! isUsingNextEventEstimator() ) {
		throw std::runtime_error( "RayListController::copyPointDetTallyToCPU - Next Event Estimator not enabled." );
	}
	pNextEventEstimator->copyToCPU();
}

template<unsigned N >
gpuTallyType_t
RayListController<N>::getPointDetTally(unsigned i ) const {
	if( ! isUsingNextEventEstimator() ) {
		throw std::runtime_error( "RayListController::getPointDetTally - Next Event Estimator not enabled." );
	}
	return pNextEventEstimator->getTally(i);
}

template<unsigned N >
void
RayListController<N>::copyPointDetToGPU(void) {
	if( ! isUsingNextEventEstimator() ) {
		throw std::runtime_error( "RayListController::getPointDetTally - Next Event Estimator not enabled." );
	}

	pNextEventEstimator->setGeometry( pGrid, pMatProps );
	pNextEventEstimator->setMaterialList( pMatList );
	pNextEventEstimator->copyToGPU();
}

}
template class MonteRay::RayListController<1>;
template class MonteRay::RayListController<3>;
