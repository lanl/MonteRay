#include "CollisionPointController.h"

#include <algorithm>

#include "GridBins.h"
#include "SimpleMaterialList.h"
#include "MonteRay_MaterialProperties.hh"
#include "gpuTally.h"
#include "CollisionPoints.h"
#include "ExpectedPathLength.h"
#include "GPUErrorCheck.hh"
#include "GPUSync.hh"

namespace MonteRay {

CollisionPointController::CollisionPointController(
		unsigned blocks,
        unsigned threads,
        GridBinsHost* pGB,
        SimpleMaterialListHost* pML,
        MonteRay_MaterialProperties* pMP,
        gpuTallyHost* pT ) :
        nBlocks(blocks),
        nThreads(threads),
        pGrid( pGB ),
        pMatList( pML ),
        pMatProps( pMP ),
        pTally(pT),
        nFlushs(0),
        cpuTime(0.0),
        gpuTime(0.0),
        wallTime(0.0),
        toFile( false ),
        fileIsOpen( false)
{
	bank1 = new CollisionPointsHost(1000000); // default 1 millions
	bank2 = new CollisionPointsHost(1000000); // default 1 millions

	cudaStreamCreate( &stream1 );
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	cudaEventCreate(&copySync1);
	cudaEventCreate(&copySync2);

	currentBank = bank1;
	currentCopySync = &copySync1;
}

CollisionPointController::~CollisionPointController(){
	delete bank1;
	delete bank2;

	cudaStreamDestroy(stream1);
}

unsigned
CollisionPointController::capacity(void) const {
	return currentBank->capacity();
}

unsigned
CollisionPointController::size(void) const {
	return currentBank->size();
}

void
CollisionPointController::setCapacity(unsigned n) {
	delete bank1;
	delete bank2;
	bank1 = new CollisionPointsHost(n);
	bank2 = new CollisionPointsHost(n);
	currentBank = bank1;
}

void
CollisionPointController::add(
		gpuFloatType_t pos[3],
		gpuFloatType_t dir[3],
		gpuFloatType_t energy, gpuFloatType_t weight, unsigned index) {

	add( pos[0], pos[1], pos[2],
		 dir[0], dir[1], dir[2],
		 energy, weight, index );
}

void
CollisionPointController::add(
		gpuFloatType_t x, gpuFloatType_t y, gpuFloatType_t z,
        gpuFloatType_t u, gpuFloatType_t v, gpuFloatType_t w,
        gpuFloatType_t energy, gpuFloatType_t weight, unsigned index) {

	currentBank->add(x,y,z,u,v,w,energy,weight,index);
	if( size() == capacity() ) {
		std::cout << "Debug: bank full, flushing.\n";
		flush();
	}
}

void
CollisionPointController::add( const gpuParticle_t& particle){
	currentBank->add( particle );
	if( size() == capacity() ) {
		std::cout << "Debug: bank full, flushing.\n";
		flush();
	}
}

void
CollisionPointController::add( const gpuParticle_t* particle, unsigned N){
	int NSpaces = capacity() - size();

	int NAdding = std::min(NSpaces, int(N));
	int NRemaining = N - NAdding;
	currentBank->add( particle, NAdding );
	if( size() == capacity() ) {
		std::cout << "Debug: bank full, flushing.\n";
		flush();
	}
	if( NRemaining > 0 ) {
		add( particle + NAdding, NRemaining );
	}
}

void
CollisionPointController::add( const void* particle, unsigned N){
	add(  (const gpuParticle_t*) particle, N  );
}

void
CollisionPointController::flush(bool final){
	if( isSendingToFile() ) { flushToFile(final); }

	if( currentBank->size() == 0 ) {
		return;
	}

	if( nFlushs > 0 ) {
		std::cout << "Debug: flush nFlushs =" <<nFlushs-1 << " -- stopping timers\n";
		stopTimers();
	}
	std::cout << "Debug: flush nFlushs =" <<nFlushs << " -- starting timers\n";

	startTimers();

	++nFlushs;
	currentBank->copyToGPU();
	gpuErrchk( cudaEventRecord(*currentCopySync, 0) );
	gpuErrchk( cudaEventSynchronize(*currentCopySync) );

	// launch kernel
	rayTraceTally<<<nBlocks,nThreads,0,stream1>>>(pGrid->ptr_device, currentBank->ptrPoints_device, pMatList->ptr_device, pMatProps->ptrData_device, pMatList->getHashPtr()->getPtrDevice(), pTally->ptr_device);

	// only uncomment for testing, forces the cpu and gpu to sync
//	gpuErrchk( cudaPeekAtLastError() );

	gpuErrchk( cudaEventRecord(stopGPU,stream1) );
	gpuErrchk( cudaStreamWaitEvent(stream1, stopGPU, 0) );

	if( final ) {
		std::cout << "Debug: final flush nFlushs =" <<nFlushs-1 << " -- stopping timers\n";
		stopTimers();
		printTotalTime();
		return;
	}

	swapBanks();
}

void
CollisionPointController::flushToFile(bool final){
	if( final ) {
		std::cout << "Debug: CollisionPointController::flushToFile - starting -- final = true \n";
	} else {
		std::cout << "Debug: CollisionPointController::flushToFile - starting -- final = false \n";
	}

	if( ! fileIsOpen ) {
		try {
			std::cout << "Debug: CollisionPointController::flushToFile - opening file, filename=" << outputFileName << "\n";
			currentBank->openOutput( outputFileName );
		} catch ( ... ) {
	        std::stringstream msg;
	        msg << "Failure opening file for collision writing!\n";
	        msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "CollisionPointController::flushToFile" << "\n\n";
	        std::cout << "MonteRay Error: " << msg.str();
	        throw std::runtime_error( msg.str() );
		}

		fileIsOpen = true;
	}

	try {
		std::cout << "Debug: CollisionPointController::flushToFile - writing bank -- bank size = "<< currentBank->size() << "\n";
		currentBank->writeBank();
	} catch ( ... ) {
        std::stringstream msg;
        msg << "Failure writing collisions to file!\n";
        msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "CollisionPointController::flushToFile" << "\n\n";
        std::cout << "MonteRay Error: " << msg.str();
        throw std::runtime_error( msg.str() );
	}

	currentBank->clear();

	if( final ) {
		try {
			std::cout << "Debug: CollisionPointController::flushToFile - file flush, closing collision file\n";
			currentBank->closeOutput();
		} catch ( ... ) {
	        std::stringstream msg;
	        msg << "Failure closing collision file!\n";
	        msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " <<"CollisionPointController::flushToFile" << "\n\n";
	        std::cout << "MonteRay Error: " << msg.str();
	        throw std::runtime_error( msg.str() );
		}

		fileIsOpen = false;
	}
}

void
CollisionPointController::readCollisionsFromFile(std::string name) {

	bool end = false;
	unsigned numParticles = 0;
	do  {
		end = currentBank->readToBank(name, numParticles);
		numParticles += currentBank->size();
		flush(end);
	} while ( ! end );
}

void
CollisionPointController::startTimers(){
	// start timers
	timer.start();
	gpuErrchk( cudaEventRecord(start,0) );
	gpuErrchk( cudaEventRecord(startGPU,stream1) );
}

void
CollisionPointController::stopTimers(){
	// stop timers and sync

	timer.stop();
	float_t cpuCycleTime = timer.getTime();
	cpuTime += cpuCycleTime;

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
}

void
CollisionPointController::swapBanks(){
	// Swap banks
	if( currentBank == bank1 ) {
		currentBank = bank2;
		currentCopySync = &copySync2;
	} else {
		currentBank = bank1;
		currentCopySync = &copySync1;
	}

	cudaEventSynchronize(*currentCopySync);
	currentBank->clear();
}

void CollisionPointController::sync(void){
	GPUSync sync;
	sync.sync();
}

void
CollisionPointController::clearTally(void) {

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

void
CollisionPointController::printTotalTime() const{
	std::cout << "Debug: \n";
	std::cout << "Debug: total gpuTime = " << gpuTime << "\n";
	std::cout << "Debug: total cpuTime = " << cpuTime << "\n";
	std::cout << "Debug: total wallTime = " << wallTime << "\n";
}

void
CollisionPointController::printCycleTime(float_t cpu, float_t gpu, float_t wall) const{
	std::cout << "Debug: \n";
	std::cout << "Debug: cycle gpuTime = " << gpu << "\n";
	std::cout << "Debug: cycle cpuTime = " << cpu << "\n";
	std::cout << "Debug: cycle wallTime = " << wall << "\n";

}

}
