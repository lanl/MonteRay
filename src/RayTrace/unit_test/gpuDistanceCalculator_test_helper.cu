#include <cstring>

#include "MonteRayDefinitions.hh"
#include "GPUErrorCheck.hh"
#include "GPUUtilityFunctions.hh"

#include "gpuDistanceCalculator_test_helper.hh"
#include "GridBins.hh"
#include "RayWorkInfo.hh"

namespace MonteRay{

void
gpuDistanceCalculatorTestHelper::launchRayTrace( const Position_t& pos, const Direction_t& dir, gpuRayFloat_t distance, bool outsideDistances) {

#ifdef __CUDACC__
    pRayInfo->copyToGPU();
    cudaEvent_t sync;
    cudaEventCreate(&sync);
    kernelRayTrace<<<1,1>>>(
            pRayInfo->devicePtr,
            grid_device,
            pos[0], pos[1], pos[2],
            dir[0], dir[1], dir[2],
            distance,
            outsideDistances );
    gpuErrchk( cudaPeekAtLastError() );

    cudaEventRecord(sync, 0);
    cudaEventSynchronize(sync);
    pRayInfo->copyToCPU();
#else
    kernelRayTrace(
            pRayInfo,
            grid_device,
            pos[0], pos[1], pos[2],
            dir[0], dir[1], dir[2],
            distance,
            outsideDistances );
#endif

return;
}

gpuDistanceCalculatorTestHelper::gpuDistanceCalculatorTestHelper(){
    grid_device = NULL;
    pRayInfo.reset( new RayWorkInfo( 1, true ) );
    nCells = 0;
}

void gpuDistanceCalculatorTestHelper::gpuCheck() {
    //MonteRay::gpuCheck();
}

gpuDistanceCalculatorTestHelper::~gpuDistanceCalculatorTestHelper(){

    //	std::cout << "Debug: starting ~gpuDistanceCalculatorTestHelper()" << std::endl;

}

void gpuDistanceCalculatorTestHelper::copyGridtoGPU( GridBins* grid){

    nCells = grid->getNumCells();

#ifdef __CUDACC__
    // copy the grid
    grid->copyToGPU();
    grid_device = grid->devicePtr;

#else
    grid_device = grid;
#endif
}

void  gpuDistanceCalculatorTestHelper::copyDistancesFromGPU( gpuRayFloat_t* distance ) {
    for( unsigned i = 0; i < pRayInfo->getRayCastSize(0); ++i ){
        distance[i] = pRayInfo->getRayCastDist(0,i);
    }
}

void  gpuDistanceCalculatorTestHelper::copyCellsFromCPU( int* cells ) {
    for( unsigned i = 0; i < pRayInfo->getRayCastSize(0); ++i ){
        cells[i] = pRayInfo->getRayCastCell(0,i);
    }
}

unsigned gpuDistanceCalculatorTestHelper:: getNumCrossingsFromGPU(void) {
    return pRayInfo->getRayCastSize(0);
}


void gpuDistanceCalculatorTestHelper::setupTimers(){
#ifdef __CUDACC__
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
#else
    timer.start();
#endif
}

void gpuDistanceCalculatorTestHelper::stopTimers(){
#ifdef __CUDACC__
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;

    cudaEventElapsedTime(&elapsedTime, start, stop );
    std::cout << "Elapsed time in CUDA kernel=" << elapsedTime << " msec" << std::endl;
#else
    timer.stop();
    std::cout << "Elapsed time in non-CUDA kernel=" << timer.getTime()*1000.0  << " msec" << std::endl;
#endif


}

}
