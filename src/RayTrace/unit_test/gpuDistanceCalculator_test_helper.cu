#include <cuda.h>
#include "global.h"
#include "gpuGlobal.h"

#include "gpuDistanceCalculator_test_helper.hh"
#include "cudaGridBins.h"

void
gpuDistanceCalculatorTestHelper::launchGetDistancesToAllCenters( unsigned nBlocks, unsigned nThreads, const Position_t& pos) {
	float_t x = pos[0];
	float_t y = pos[1];
	float_t z = pos[2];

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	kernelGetDistancesToAllCenters<<<nBlocks,nThreads>>>(grid_device, distances_device, x, y, z);
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);
	gpuErrchk( cudaPeekAtLastError() );
	return;
}

void
gpuDistanceCalculatorTestHelper::launchRayTrace( const Position_t& pos, const Direction_t& dir, float_t distance, bool outsideDistances) {

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	kernelCudaRayTrace<<<1,1>>>(numCrossings_device,
			                                 grid_device,
			                                 cells_device,
			                                 distances_device,
			                                 pos[0], pos[1], pos[2],
			                                 dir[0], dir[1], dir[2],
			                                 distance,
			                                 outsideDistances );

	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);
	gpuErrchk( cudaPeekAtLastError() );
	return;
}

gpuDistanceCalculatorTestHelper::gpuDistanceCalculatorTestHelper(){
	grid_device = NULL;
	distances_device = NULL;
	cells_device = NULL;
	numCrossings_device = NULL;

	nCells = 0;

	int deviceCount;

	cuInit(0);
	cuDeviceGetCount(&deviceCount);
	if (deviceCount == 0) {
		printf("No CUDA-compatible devices found\n");
		exit(1);
	}
	printf("Number of CUDA devices=%d\n",deviceCount);
	gpuErrchk( cudaPeekAtLastError() );
}

gpuDistanceCalculatorTestHelper::~gpuDistanceCalculatorTestHelper(){

//	std::cout << "Debug: starting ~gpuDistanceCalculatorTestHelper()" << std::endl;

	if( grid_device != NULL ) {
		cudaFree( grid_device );
		gpuErrchk( cudaPeekAtLastError() );
	}
	if( distances_device != NULL ) {
		cudaFree( distances_device );
		gpuErrchk( cudaPeekAtLastError() );
	}
	if( cells_device != NULL ) {
		cudaFree( cells_device );
		gpuErrchk( cudaPeekAtLastError() );
	}
	if( numCrossings_device != NULL ) {
		cudaFree( numCrossings_device );
		gpuErrchk( cudaPeekAtLastError() );
	}
//	std::cout << "Debug: exitting ~gpuDistanceCalculatorTestHelper()" << std::endl;
}

void gpuDistanceCalculatorTestHelper::copyGridtoGPU( GridBins* grid){
	// allocate and copy the grid
	CUDA_CHECK_RETURN( cudaMalloc((void**) &grid_device, sizeof(GridBins) ));
	CUDA_CHECK_RETURN( cudaMemcpy(grid_device, grid, sizeof(GridBins), cudaMemcpyHostToDevice ));

	nCells = getNumCells(grid);

	// allocate the distances
	CUDA_CHECK_RETURN(cudaMalloc((void**) &distances_device, sizeof(float_t) * nCells ));

	// allocate the cells
	CUDA_CHECK_RETURN(cudaMalloc((void**) &cells_device, sizeof(int) * nCells ));

	// allocate the num crossings
	CUDA_CHECK_RETURN(cudaMalloc((void**) &numCrossings_device, sizeof(unsigned) ));
}

void gpuDistanceCalculatorTestHelper::copyDistancesFromGPU( float_t* distances){
	// copy distances back to the host
	CUDA_CHECK_RETURN(cudaMemcpy(distances, distances_device, sizeof(float_t) * nCells, cudaMemcpyDeviceToHost));
}

void gpuDistanceCalculatorTestHelper::copyCellsFromCPU( int* cells){
	// copy cells back to the host
	CUDA_CHECK_RETURN(cudaMemcpy(cells, cells_device, sizeof(int) * nCells, cudaMemcpyDeviceToHost));
}

unsigned gpuDistanceCalculatorTestHelper::getNumCrossingsFromGPU( void ){
	// copy num crossings
	unsigned num;
	CUDA_CHECK_RETURN(cudaMemcpy(&num, numCrossings_device, sizeof(unsigned) * 1, cudaMemcpyDeviceToHost));
	return num;
}

void gpuDistanceCalculatorTestHelper::setupTimers(){
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
}

void gpuDistanceCalculatorTestHelper::stopTimers(){
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime;
	gpuErrchk( cudaPeekAtLastError() );

	cudaEventElapsedTime(&elapsedTime, start, stop );

	std::cout << "Elapsed time in CUDA kernel=" << elapsedTime << " msec" << std::endl;

	gpuErrchk( cudaPeekAtLastError() );

}


