#include <cuda.h>
#include "global.h"
#include "gpuGlobal.h"

#include "fi_genericGPU_test_helper.hh"
#include "ExpectedPathLength.h"

FIGenericGPUTestHelper::FIGenericGPUTestHelper(unsigned num){
	int deviceCount;
	nCells = num;
	grid_device = NULL;

	cuInit(0);
	cuDeviceGetCount(&deviceCount);
	if (deviceCount == 0) {
		printf("No CUDA-compatible devices found\n");
		exit(1);
	}
	printf("Number of CUDA devices=%d\n",deviceCount);
	gpuErrchk( cudaPeekAtLastError() );
}

FIGenericGPUTestHelper::~FIGenericGPUTestHelper(){
	free( tally );
	if( grid_device != NULL ) {
		cudaFree( grid_device );
	}
}

void FIGenericGPUTestHelper::setupTimers(){
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
}

void FIGenericGPUTestHelper::stopTimers(){
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime;
	gpuErrchk( cudaPeekAtLastError() );

	cudaEventElapsedTime(&elapsedTime, start, stop );

	std::cout << "Elapsed time in CUDA kernel=" << elapsedTime << " msec" << std::endl;

	gpuErrchk( cudaPeekAtLastError() );
}

#ifdef CUDA
__global__ void testTallyCrossSection(CollisionPoints* pCP, SimpleCrossSection* pXS, gpuFloatType_t* results){

	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int N = pCP->size;
	while( tid < N ) {
		gpuFloatType_t E = getEnergy(pCP, tid);
		results[tid] = getTotalXS(pXS, E);
		tid += blockDim.x*gridDim.x;
	}
	return;
}
#endif

void FIGenericGPUTestHelper::launchTallyCrossSection(unsigned nBlocks, unsigned nThreads, CollisionPointsHost* pCP, SimpleCrossSectionHost* pXS ){
	gpuFloatType_t* tally_device;

	unsigned long long allocSize = sizeof(gpuFloatType_t)*nCells;
	tally = (gpuFloatType_t*) malloc ( allocSize );
	CUDA_CHECK_RETURN( cudaMalloc( &tally_device, allocSize ));
	CUDA_CHECK_RETURN( cudaMemset(tally_device, 0, allocSize));
	gpuErrchk( cudaPeekAtLastError() );

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	testTallyCrossSection<<<nBlocks,nThreads>>>(pCP->ptrPoints_device, pXS->xs_device, tally_device);
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);
	gpuErrchk( cudaPeekAtLastError() );

	CUDA_CHECK_RETURN(cudaMemcpy(tally, tally_device, allocSize, cudaMemcpyDeviceToHost));
	gpuErrchk( cudaPeekAtLastError() );
	cudaFree( tally_device );
	return;
}

#ifdef CUDA
__global__ void testTallyCrossSection(CollisionPoints* pCP, SimpleMaterialList* pMatList, unsigned matIndex, gpuFloatType_t density, gpuFloatType_t* results){

	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int N = pCP->size;
	while( tid < N ) {
		gpuFloatType_t E = getEnergy(pCP, tid);
		results[tid] = getTotalXS(pMatList, matIndex, E, density);
		tid += blockDim.x*gridDim.x;
	}
	return;
}
#endif

void FIGenericGPUTestHelper::launchTallyCrossSection(unsigned nBlocks, unsigned nThreads, CollisionPointsHost* pCP, SimpleMaterialListHost* pMatList, unsigned matIndex, gpuFloatType_t density ){
	gpuFloatType_t* tally_device;

	unsigned long long allocSize = sizeof(gpuFloatType_t)*nCells;
	tally = (gpuFloatType_t*) malloc ( allocSize );
	CUDA_CHECK_RETURN( cudaMalloc( &tally_device, allocSize ));
	CUDA_CHECK_RETURN( cudaMemset(tally_device, 0, allocSize));
	gpuErrchk( cudaPeekAtLastError() );

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	testTallyCrossSection<<<nBlocks,nThreads>>>(pCP->ptrPoints_device, pMatList->ptr_device, matIndex, density, tally_device);
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);
	gpuErrchk( cudaPeekAtLastError() );

	CUDA_CHECK_RETURN(cudaMemcpy(tally, tally_device, allocSize, cudaMemcpyDeviceToHost));
	gpuErrchk( cudaPeekAtLastError() );
	cudaFree( tally_device );
	return;
}

#ifdef CUDA
__device__ __host__
gpuFloatType_t getTotalXSByMatProp(SimpleMaterialProperties* matProps, SimpleMaterialList* pMatList, unsigned cell, gpuFloatType_t E) {
	gpuFloatType_t total = 0.0f;
	for( unsigned i=0; i< matProps->props[cell].numMats; ++i) {
		gpuFloatType_t density = matProps->props[cell].density[i];
		unsigned matID = matProps->props[cell].matID[i];
		total += getTotalXS(pMatList, matID, E, density);
	}
	return total;
}
#endif

gpuFloatType_t nonCudaGetTotalXSByMatProp(SimpleMaterialProperties* matProps, SimpleMaterialList* pMatList, unsigned cell, gpuFloatType_t E) {
	gpuFloatType_t total = 0.0f;
	for( unsigned i=0; i< matProps->props[cell].numMats; ++i) {
		gpuFloatType_t density = matProps->props[cell].density[i];
		unsigned matID = matProps->props[cell].matID[i];
		total += getTotalXS(pMatList, matID, E, density);
	}
	return total;
}

gpuFloatType_t FIGenericGPUTestHelper::getTotalXSByMatProp(SimpleMaterialProperties* matProps, SimpleMaterialList* pMatList, unsigned cell, gpuFloatType_t E) {
	return nonCudaGetTotalXSByMatProp( matProps, pMatList, cell, E);
}

#ifdef CUDA
__global__ void testTallyCrossSectionAtCollision(CollisionPoints* pCP, SimpleMaterialList* pMatList, SimpleMaterialProperties* pMatProps, gpuFloatType_t* results){

	unsigned tid = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned N = pCP->size;

	while( tid < N ) {
		gpuFloatType_t E = getEnergy(pCP, tid);
		unsigned cell = getIndex( pCP, tid);

		results[tid] = getTotalXSByMatProp(pMatProps, pMatList, cell, E);
		tid += blockDim.x*gridDim.x;
	}
	return;
}
#endif

#ifdef CUDA
__global__ void testSumCrossSectionAtCollisionLocation(CollisionPoints* pCP, SimpleMaterialList* pMatList, SimpleMaterialProperties* pMatProps, gpuFloatType_t* results){

	unsigned tid = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned N = pCP->size;

	while( tid < N ) {
		gpuFloatType_t E = getEnergy(pCP, tid);
		unsigned cell = getIndex( pCP, tid);

		gpuFloatType_t value = getTotalXSByMatProp(pMatProps, pMatList, cell, E);
		atomicAdd( &results[cell], value);
		tid += blockDim.x*gridDim.x;
	}
	return;
}
#endif

void FIGenericGPUTestHelper::launchTallyCrossSectionAtCollision(unsigned nBlocks, unsigned nThreads, CollisionPointsHost* pCP, SimpleMaterialListHost* pMatList, SimpleMaterialPropertiesHost* pMatProps ){
	gpuFloatType_t* tally_device;

	unsigned long long allocSize = sizeof(gpuFloatType_t)*nCells;
	tally = (gpuFloatType_t*) malloc ( allocSize );
	CUDA_CHECK_RETURN( cudaMalloc( &tally_device, allocSize ));
	CUDA_CHECK_RETURN( cudaMemset(tally_device, 0, allocSize));
	gpuErrchk( cudaPeekAtLastError() );

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	testTallyCrossSectionAtCollision<<<nBlocks,nThreads>>>(pCP->ptrPoints_device, pMatList->ptr_device, pMatProps->ptr_device, tally_device);
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);
	gpuErrchk( cudaPeekAtLastError() );

	CUDA_CHECK_RETURN(cudaMemcpy(tally, tally_device, allocSize, cudaMemcpyDeviceToHost));
	gpuErrchk( cudaPeekAtLastError() );
	cudaFree( tally_device );
	return;
}

void FIGenericGPUTestHelper::launchSumCrossSectionAtCollisionLocation(unsigned nBlocks, unsigned nThreads, CollisionPointsHost* pCP, SimpleMaterialListHost* pMatList, SimpleMaterialPropertiesHost* pMatProps ){
	gpuFloatType_t* tally_device;

	unsigned long long allocSize = sizeof(gpuFloatType_t)*nCells;
	tally = (gpuFloatType_t*) malloc ( allocSize );
	CUDA_CHECK_RETURN( cudaMalloc( &tally_device, allocSize ));
	CUDA_CHECK_RETURN( cudaMemset(tally_device, 0, allocSize));
	gpuErrchk( cudaPeekAtLastError() );

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	testSumCrossSectionAtCollisionLocation<<<nBlocks,nThreads>>>(pCP->ptrPoints_device, pMatList->ptr_device, pMatProps->ptr_device, tally_device);
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);
	gpuErrchk( cudaPeekAtLastError() );

	CUDA_CHECK_RETURN(cudaMemcpy(tally, tally_device, allocSize, cudaMemcpyDeviceToHost));
	gpuErrchk( cudaPeekAtLastError() );
	cudaFree( tally_device );
	return;
}

void FIGenericGPUTestHelper::launchRayTraceTally(unsigned nBlocks, unsigned nThreads, CollisionPointsHost* pCP, SimpleMaterialListHost* pMatList, SimpleMaterialPropertiesHost* pMatProps ){
	gpuFloatType_t* tally_device;
	unsigned long long allocSize = sizeof(gpuFloatType_t)*nCells;
	tally = (gpuFloatType_t*) malloc ( allocSize );
	CUDA_CHECK_RETURN( cudaMalloc( &tally_device, allocSize ));
	CUDA_CHECK_RETURN( cudaMemset(tally_device, 0, allocSize));
	gpuErrchk( cudaPeekAtLastError() );

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	rayTraceTally<<<nBlocks,nThreads>>>(grid_device, pCP->ptrPoints_device, pMatList->ptr_device, pMatProps->ptr_device, tally_device);
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);
	gpuErrchk( cudaPeekAtLastError() );

	CUDA_CHECK_RETURN(cudaMemcpy(tally, tally_device, allocSize, cudaMemcpyDeviceToHost));
	gpuErrchk( cudaPeekAtLastError() );
	cudaFree( tally_device );
	return;
}

void FIGenericGPUTestHelper::copyGridtoGPU( GridBins* grid){
	// allocate and copy the grid
	CUDA_CHECK_RETURN( cudaMalloc( &grid_device, sizeof(GridBins) ));
	CUDA_CHECK_RETURN( cudaMemcpy(grid_device, grid, sizeof(GridBins), cudaMemcpyHostToDevice ));

	nCells = getNumCells(grid);
}

