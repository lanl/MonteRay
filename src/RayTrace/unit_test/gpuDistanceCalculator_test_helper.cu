#include <cstring>

#include "MonteRayDefinitions.hh"
#include "GPUErrorCheck.hh"
#include "GPUUtilityFunctions.hh"

#include "gpuDistanceCalculator_test_helper.hh"
#include "GridBins.hh"

namespace MonteRay{

void
gpuDistanceCalculatorTestHelper::launchRayTrace( const Position_t& pos, const Direction_t& dir, gpuRayFloat_t distance, bool outsideDistances) {

#ifdef __CUDACC__
	cudaEvent_t sync;
	cudaEventCreate(&sync);
	kernelRayTrace<<<1,1>>>(numCrossings_device,
			                                 grid_device,
			                                 cells_device,
			                                 distances_device,
			                                 pos[0], pos[1], pos[2],
			                                 dir[0], dir[1], dir[2],
			                                 distance,
			                                 outsideDistances );
	gpuErrchk( cudaPeekAtLastError() );

	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);
#else
	kernelRayTrace(numCrossings_device,
			                                 grid_device,
			                                 cells_device,
			                                 distances_device,
			                                 pos[0], pos[1], pos[2],
			                                 dir[0], dir[1], dir[2],
			                                 distance,
			                                 outsideDistances );
#endif

	return;
}

gpuDistanceCalculatorTestHelper::gpuDistanceCalculatorTestHelper(){
	grid_device = NULL;
	distances_device = NULL;
	cells_device = NULL;
	numCrossings_device = NULL;

	nCells = 0;
}

void gpuDistanceCalculatorTestHelper::gpuCheck() {
	MonteRay::gpuCheck();
}

gpuDistanceCalculatorTestHelper::~gpuDistanceCalculatorTestHelper(){

//	std::cout << "Debug: starting ~gpuDistanceCalculatorTestHelper()" << std::endl;

#ifdef __CUDACC__
//	if( grid_device != NULL ) {
//		CUDA_CHECK_RETURN( cudaFree( grid_device ));
//	}
	if( distances_device != NULL ) {
		CUDA_CHECK_RETURN( cudaFree( distances_device ));
	}
	if( cells_device != NULL ) {
		CUDA_CHECK_RETURN( cudaFree( cells_device ) );
	}
	if( numCrossings_device != NULL ) {
		CUDA_CHECK_RETURN( cudaFree( numCrossings_device ) );
	}
#else
	if( grid_device != NULL ) {
		free( grid_device );
	}
	if( distances_device != NULL ) {
		free( distances_device );
	}
	if( cells_device != NULL ) {
		free( cells_device );
	}
	if( numCrossings_device != NULL ) {
		free( numCrossings_device );
	}
#endif
//	std::cout << "Debug: exitting ~gpuDistanceCalculatorTestHelper()" << std::endl;
}

void gpuDistanceCalculatorTestHelper::copyGridtoGPU( GridBins* grid){

	nCells = grid->getNumCells();

#ifdef __CUDACC__
	// allocate and copy the grid
//	CUDA_CHECK_RETURN( cudaMalloc((void**) &grid_device, sizeof(GridBins) ));
//	CUDA_CHECK_RETURN( cudaMemcpy(grid_device, grid, sizeof(GridBins), cudaMemcpyHostToDevice ));
	grid->copyToGPU();
	grid_device = grid->devicePtr;

	// allocate the distances
	CUDA_CHECK_RETURN(cudaMalloc((void**) &distances_device, sizeof(gpuRayFloat_t) * nCells ));

	// allocate the cells
	CUDA_CHECK_RETURN(cudaMalloc((void**) &cells_device, sizeof(int) * nCells ));

	// allocate the num crossings
	CUDA_CHECK_RETURN(cudaMalloc((void**) &numCrossings_device, sizeof(unsigned) ));
#else
	grid_device = (MonteRay::GridBins*) malloc( sizeof(GridBins) );
	memcpy( grid_device, grid, sizeof(GridBins) );

	distances_device = (gpuRayFloat_t*) malloc( sizeof(float_t) * nCells );
	cells_device = (int*) malloc( sizeof(int) * nCells );

	numCrossings_device = (int*) malloc( sizeof(unsigned) );
#endif
}

void gpuDistanceCalculatorTestHelper::copyDistancesFromGPU( gpuRayFloat_t* distances){
	// copy distances back to the host
#ifdef __CUDACC__
	CUDA_CHECK_RETURN(cudaMemcpy(distances, distances_device, sizeof(gpuRayFloat_t) * nCells, cudaMemcpyDeviceToHost));
#else
	memcpy(distances, distances_device, sizeof(gpuRayFloat_t) * nCells);
#endif
}

void gpuDistanceCalculatorTestHelper::copyCellsFromCPU( int* cells){
	// copy cells back to the host
#ifdef __CUDACC__
	CUDA_CHECK_RETURN(cudaMemcpy(cells, cells_device, sizeof(int) * nCells, cudaMemcpyDeviceToHost));
#else
	memcpy(cells, cells_device, sizeof(int) * nCells );
#endif
}

unsigned gpuDistanceCalculatorTestHelper::getNumCrossingsFromGPU( void ){
	// copy num crossings
	unsigned num;
#ifdef __CUDACC__
	CUDA_CHECK_RETURN(cudaMemcpy(&num, numCrossings_device, sizeof(unsigned) * 1, cudaMemcpyDeviceToHost));
#else
	memcpy( &num, numCrossings_device, sizeof(unsigned) * 1 );
#endif
	return num;
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
