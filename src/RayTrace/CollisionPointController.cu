#include "CollisionPointController.h"

#include "GridBins.h"
#include "SimpleMaterialList.h"
#include "SimpleMaterialProperties.h"
#include "gpuTally.h"
#include "CollisionPoints.h"
#include "ExpectedPathLength.h"

namespace MonteRay {

CollisionPointController::CollisionPointController(
		unsigned blocks,
        unsigned threads,
        GridBinsHost* pGB,
        SimpleMaterialListHost* pML,
        SimpleMaterialPropertiesHost* pMP,
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
        wallTime(0.0)
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
CollisionPointController::flush(bool final){
	if( nFlushs > 0 ) {
		std::cout << "Debug: flush nFlushs =" <<nFlushs-1 << " -- stopping timers\n";
		stopTimers();
	}
	std::cout << "Debug: flush nFlushs =" <<nFlushs << " -- starting timers\n";
	gpuErrchk( cudaPeekAtLastError() );
	startTimers();

	++nFlushs;
    gpuErrchk( cudaPeekAtLastError() );
	currentBank->copyToGPU();
	cudaEventRecord(*currentCopySync, 0);
	cudaEventSynchronize(*currentCopySync);

	// launch kernel
	rayTraceTally<<<nBlocks,nThreads,0,stream1>>>(pGrid->ptr_device, currentBank->ptrPoints_device, pMatList->ptr_device, pMatProps->ptr_device, pTally->ptr_device);
	cudaEventRecord(stopGPU,stream1);
	cudaStreamWaitEvent(stream1, stopGPU, 0);

	if( final ) {
		std::cout << "Debug: flush nFlushs =" <<nFlushs-1 << " -- stopping timers\n";
		stopTimers();
		printTotalTime();
		return;
	}

	swapBanks();
}

void
CollisionPointController::startTimers(){
	// start timers
	timer.start();
	cudaEventRecord(start,0);
	gpuErrchk( cudaPeekAtLastError() );
	cudaEventRecord(startGPU,stream1);
	gpuErrchk( cudaPeekAtLastError() );
}

void
CollisionPointController::stopTimers(){
	// stop timers and sync

	timer.stop();
	float_t cpuCycleTime = timer.getTime();
	cpuTime += cpuCycleTime;

	cudaStreamSynchronize( stream1 );
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float_t gpuCycleTime;
	cudaEventElapsedTime(&gpuCycleTime, startGPU, stopGPU );
	gpuCycleTime /= 1000.0;
	if( gpuCycleTime < 0.0 ) {
		gpuCycleTime = 0.0;
	}
	gpuTime += gpuCycleTime;

	float totalCycleTime;
	cudaEventElapsedTime(&totalCycleTime, start, stop );
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
	gpuSync sync;
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

	gpuSync sync;
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
