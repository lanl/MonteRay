#include <cuda.h>

#include "MonteRayDefinitions.hh"
#include "GPUErrorCheck.hh"
#include "GPUAtomicAdd.hh"
#include "ExpectedPathLength.h"

#include "RayTraceNightly_genericGPU_test_helper.hh"


FIGenericGPUTestHelper::FIGenericGPUTestHelper(unsigned num){
	nCells = num;
	tally = NULL;
	grid_device = NULL;
}

FIGenericGPUTestHelper::~FIGenericGPUTestHelper(){
	if( tally != NULL ) {
		free( tally );
	}
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

#ifdef __CUDACC__
__global__ void testTallyCrossSection(CollisionPoints* pCP, MonteRayCrossSection* pXS, gpuTallyType_t* results){

	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int N = pCP->size();
	while( tid < N ) {
		gpuFloatType_t E = pCP->getEnergy(tid);
		results[tid] = getTotalXS(pXS, E);
		tid += blockDim.x*gridDim.x;
	}
	return;
}
#endif

void FIGenericGPUTestHelper::launchTallyCrossSection(unsigned nBlocks, unsigned nThreads, CollisionPointsHost* pCP, MonteRayCrossSectionHost* pXS ){
	gpuTallyType_t* tally_device;

	unsigned long long allocSize = sizeof(gpuTallyType_t)*nCells;
	tally = (gpuTallyType_t*) malloc ( allocSize );
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
__global__ void testTallyCrossSection(CollisionPoints* pCP, SimpleMaterialList* pMatList, unsigned matIndex, HashLookup* pHash, gpuFloatType_t density, gpuTallyType_t* results){

	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int N = pCP->size();
	while( tid < N ) {
		gpuFloatType_t E = pCP->getEnergy(tid);
		unsigned HashBin = getHashBin( pHash, E);
		results[tid] = getTotalXS(pMatList, matIndex, pHash, HashBin, E, density);
		tid += blockDim.x*gridDim.x;
	}
	return;
}
#endif

void FIGenericGPUTestHelper::launchTallyCrossSection(unsigned nBlocks, unsigned nThreads, CollisionPointsHost* pCP, SimpleMaterialListHost* pMatList, unsigned matIndex, gpuFloatType_t density ){
	gpuTallyType_t* tally_device;

	unsigned long long allocSize = sizeof(gpuTallyType_t)*nCells;
	tally = (gpuTallyType_t*) malloc ( allocSize );
	CUDA_CHECK_RETURN( cudaMalloc( &tally_device, allocSize ));
	CUDA_CHECK_RETURN( cudaMemset(tally_device, 0, allocSize));
	gpuErrchk( cudaPeekAtLastError() );

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	testTallyCrossSection<<<nBlocks,nThreads>>>(pCP->ptrPoints_device, pMatList->ptr_device, matIndex, pMatList->getHashPtr()->getPtrDevice(), density, tally_device);
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
gpuFloatType_t getTotalXSByMatProp(MonteRay_MaterialProperties_Data* matProps, SimpleMaterialList* pMatList, HashLookup* pHash, unsigned HashBin, unsigned cell, gpuFloatType_t E) {
	gpuFloatType_t total = 0.0f;
	for( unsigned i=0; i< getNumMats(matProps,cell); ++i) {
		gpuFloatType_t density = getDensity(matProps,cell,i);
		unsigned matID = getMatID(matProps,cell,i);
//		unsigned materialIndex = materialIDtoIndex(pMatList, matID);
		total += getTotalXS(pMatList, matID, pHash, HashBin, E, density);
	}
	return total;
}
#endif

gpuFloatType_t nonCudaGetTotalXSByMatProp(MonteRay_MaterialProperties* matProps, SimpleMaterialList* pMatList, HashLookup* pHash, unsigned HashBin, unsigned cell, gpuFloatType_t E) {
	gpuFloatType_t total = 0.0f;
	for( unsigned i=0; i< matProps->getNumMaterials(cell); ++i) {
		gpuFloatType_t density = matProps->getMaterialDensity(cell,i);
		MonteRay_MaterialProperties::MatID_t matID = matProps->getMaterialID(cell,i);
		//unsigned materialIndex = materialIDtoIndex(pMatList, matID);
		total += getTotalXS(pMatList, matID, pHash, HashBin, E, density);
	}
	return total;
}

gpuFloatType_t nonCudaGetTotalXSByMatProp(MonteRay_MaterialProperties* matProps, SimpleMaterialList* pMatList, unsigned cell, gpuFloatType_t E) {
	gpuFloatType_t total = 0.0f;
	for( unsigned i=0; i< matProps->getNumMaterials(cell); ++i) {
		gpuFloatType_t density = matProps->getMaterialDensity(cell,i);
		unsigned matID = matProps->getMaterialID(cell,i);
		//unsigned materialIndex = materialIDtoIndex(pMatList, matID);
		total += getTotalXS(pMatList, matID, E, density);
	}
	return total;
}

gpuFloatType_t FIGenericGPUTestHelper::getTotalXSByMatProp(MonteRay_MaterialProperties* matProps, SimpleMaterialList* pMatList, HashLookup* pHash, unsigned HashBin, unsigned cell, gpuFloatType_t E) {
	return nonCudaGetTotalXSByMatProp( matProps, pMatList, pHash, HashBin, cell, E);
}

gpuFloatType_t FIGenericGPUTestHelper::getTotalXSByMatProp(MonteRay_MaterialProperties* matProps, SimpleMaterialList* pMatList, unsigned cell, gpuFloatType_t E) {
	return nonCudaGetTotalXSByMatProp( matProps, pMatList, cell, E);
}

#ifdef CUDA
__global__ void testTallyCrossSectionAtCollision(CollisionPoints* pCP, SimpleMaterialList* pMatList, MonteRay_MaterialProperties_Data* pMatProps, HashLookup* pHash, gpuTallyType_t* results){

	unsigned tid = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned N = pCP->size();

	while( tid < N ) {
		gpuFloatType_t E = pCP->getEnergy(tid);
		unsigned HashBin = getHashBin( pHash, E);
		unsigned cell = pCP->getIndex(tid);

		results[tid] = getTotalXSByMatProp(pMatProps, pMatList, pHash, HashBin, cell, E);
		tid += blockDim.x*gridDim.x;
	}
	return;
}
#endif

#ifdef CUDA
__global__ void testSumCrossSectionAtCollisionLocation(CollisionPoints* pCP, SimpleMaterialList* pMatList, MonteRay_MaterialProperties_Data* pMatProps, HashLookup* pHash, gpuTallyType_t* results){

	unsigned tid = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned N = pCP->size();

	while( tid < N ) {
		gpuFloatType_t E = pCP->getEnergy(tid);
		unsigned HashBin = getHashBin( pHash, E);
		unsigned cell = pCP->getIndex(tid);

		gpuTallyType_t value = getTotalXSByMatProp(pMatProps, pMatList, pHash, HashBin, cell, E);

		gpu_atomicAdd( &results[cell], value);
		tid += blockDim.x*gridDim.x;
	}
	return;
}
#endif

void FIGenericGPUTestHelper::launchTallyCrossSectionAtCollision(unsigned nBlocks, unsigned nThreads, CollisionPointsHost* pCP, SimpleMaterialListHost* pMatList, MonteRay_MaterialProperties* pMatProps ){
	gpuTallyType_t* tally_device;

	unsigned long long allocSize = sizeof(gpuTallyType_t)*nCells;
	tally = (gpuTallyType_t*) malloc ( allocSize );
	CUDA_CHECK_RETURN( cudaMalloc( &tally_device, allocSize ));
	CUDA_CHECK_RETURN( cudaMemset(tally_device, 0, allocSize));
	gpuErrchk( cudaPeekAtLastError() );

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	testTallyCrossSectionAtCollision<<<nBlocks,nThreads>>>(pCP->ptrPoints_device, pMatList->ptr_device, pMatProps->ptrData_device, pMatList->getHashPtr()->getPtrDevice(), tally_device);
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);
	gpuErrchk( cudaPeekAtLastError() );

	CUDA_CHECK_RETURN(cudaMemcpy(tally, tally_device, allocSize, cudaMemcpyDeviceToHost));
	gpuErrchk( cudaPeekAtLastError() );
	cudaFree( tally_device );
	return;
}

void FIGenericGPUTestHelper::launchSumCrossSectionAtCollisionLocation(unsigned nBlocks, unsigned nThreads, CollisionPointsHost* pCP, SimpleMaterialListHost* pMatList, MonteRay_MaterialProperties* pMatProps ){
	gpuTallyType_t* tally_device;

	unsigned long long allocSize = sizeof(gpuTallyType_t)*nCells;
	tally = (gpuTallyType_t*) malloc ( allocSize );
	CUDA_CHECK_RETURN( cudaMalloc( &tally_device, allocSize ));
	CUDA_CHECK_RETURN( cudaMemset(tally_device, 0, allocSize));
	gpuErrchk( cudaPeekAtLastError() );

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	testSumCrossSectionAtCollisionLocation<<<nBlocks,nThreads>>>(pCP->ptrPoints_device, pMatList->ptr_device, pMatProps->ptrData_device, pMatList->getHashPtr()->getPtrDevice(), tally_device);
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);
	gpuErrchk( cudaPeekAtLastError() );

	CUDA_CHECK_RETURN(cudaMemcpy(tally, tally_device, allocSize, cudaMemcpyDeviceToHost));
	gpuErrchk( cudaPeekAtLastError() );
	cudaFree( tally_device );
	return;
}

void FIGenericGPUTestHelper::launchRayTraceTally(unsigned nBlocks, unsigned nThreads, CollisionPointsHost* pCP, SimpleMaterialListHost* pMatList, MonteRay_MaterialProperties* pMatProps ){
	gpuTallyType_t* tally_device;
	unsigned long long allocSize = sizeof(gpuTallyType_t)*nCells;
	tally = (gpuTallyType_t*) malloc ( allocSize );
	CUDA_CHECK_RETURN( cudaMalloc( &tally_device, allocSize ));
	CUDA_CHECK_RETURN( cudaMemset(tally_device, 0, allocSize));
	gpuErrchk( cudaPeekAtLastError() );

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	rayTraceTally<<<nBlocks,nThreads>>>(grid_device, pCP->ptrPoints_device, pMatList->ptr_device, pMatProps->ptrData_device, pMatList->getHashPtr()->getPtrDevice(), tally_device);
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

