#include "MonteRayDefinitions.hh"
#include "GPUErrorCheck.hh"
#include "GPUAtomicAdd.hh"
#include "ExpectedPathLength.hh"

#include "fi_genericGPU_test_helper.hh"

template<unsigned N>
FIGenericGPUTestHelper<N>::FIGenericGPUTestHelper(unsigned num){
	nCells = num;
	tally = NULL;
	grid_device = NULL;
}

template<unsigned N>
FIGenericGPUTestHelper<N>::~FIGenericGPUTestHelper(){
	if( tally != NULL ) {
		free( tally );
	}
	if( grid_device != NULL ) {
#ifdef __CUDACC__
		cudaFree( grid_device );
#endif
	}
}

template<unsigned N>
void FIGenericGPUTestHelper<N>::setupTimers(){
#ifdef __CUDACC__
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#else
	timer.start();
#endif
}

template<unsigned N>
void FIGenericGPUTestHelper<N>::stopTimers(){
#ifdef __CUDACC__
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime;
	gpuErrchk( cudaPeekAtLastError() );

	cudaEventElapsedTime(&elapsedTime, start, stop );

	std::cout << "Elapsed time in CUDA kernel=" << elapsedTime << " msec" << std::endl;

	gpuErrchk( cudaPeekAtLastError() );
#else
	timer.stop();
	std::cout << "Elapsed time in non-CUDA kernel=" << timer.getTime()*1000.0 << " msec" << std::endl;
#endif
}

template<unsigned N>
CUDA_CALLABLE_KERNEL void testTallyCrossSection(const RayList_t<N>* pCP, const MonteRayCrossSection* pXS, gpuTallyType_t* results){

#ifdef __CUDACC__
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
#else
	int tid = 0;
#endif

	int num = pCP->size();
	while( tid < num ) {
		gpuFloatType_t E = pCP->getEnergy(tid);
		results[tid] = getTotalXS(pXS, E);
#ifdef __CUDACC__
		tid += blockDim.x*gridDim.x;
#else
		tid++;
#endif
	}
	return;
}

template CUDA_CALLABLE_KERNEL void
testTallyCrossSection<1>(const RayList_t<1>* pCP, const MonteRayCrossSection* pXS, gpuTallyType_t* results);

template CUDA_CALLABLE_KERNEL void
testTallyCrossSection<3>(const RayList_t<3>* pCP, const MonteRayCrossSection* pXS, gpuTallyType_t* results);

template<unsigned N>
void FIGenericGPUTestHelper<N>::launchTallyCrossSection(
		unsigned nBlocks, unsigned nThreads,
		const RayListInterface<N>* pCP,
		const MonteRayCrossSectionHost* pXS )
{
	unsigned long long allocSize = sizeof(gpuTallyType_t)*nCells;
	tally = (gpuTallyType_t*) malloc ( allocSize );

#ifdef __CUDACC__
	gpuTallyType_t* tally_device;
	CUDA_CHECK_RETURN( cudaMalloc( &tally_device, allocSize ));
	CUDA_CHECK_RETURN( cudaMemset(tally_device, 0, allocSize));
	gpuErrchk( cudaPeekAtLastError() );

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	testTallyCrossSection<N><<<nBlocks,nThreads>>>(pCP->getPtrPoints()->devicePtr, pXS->xs_device, tally_device);
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);
	gpuErrchk( cudaPeekAtLastError() );

	CUDA_CHECK_RETURN(cudaMemcpy(tally, tally_device, allocSize, cudaMemcpyDeviceToHost));
	gpuErrchk( cudaPeekAtLastError() );
	cudaFree( tally_device );
#else
	testTallyCrossSection<N>(pCP->getPtrPoints(), pXS->getPtr(), tally);
#endif
	return;
}

template< unsigned N>
CUDA_CALLABLE_KERNEL void testTallyCrossSection(
		const RayList_t<N>* pCP,
		const MonteRayMaterialList* pMatList,
		unsigned matIndex,
		const HashLookup* pHash,
		gpuFloatType_t density,
		gpuTallyType_t* results)
{
#ifdef __CUDACC__
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
#else
	int tid = 0;
#endif
	int num = pCP->size();
	while( tid < num ) {
		gpuFloatType_t E = pCP->getEnergy(tid);
		unsigned HashBin = getHashBin( pHash, E);
		results[tid] = getTotalXS(pMatList, matIndex, pHash, HashBin, E, density);
#ifdef __CUDACC__
		tid += blockDim.x*gridDim.x;
#else
		tid++;
#endif
	}
	return;
}

template<unsigned N>
void FIGenericGPUTestHelper<N>::launchTallyCrossSection(
		unsigned nBlocks, unsigned nThreads,
		const RayListInterface<N>* pCP,
		const MonteRayMaterialListHost* pMatList,
		unsigned matIndex,
		gpuFloatType_t density )
{
	unsigned long long allocSize = sizeof(gpuTallyType_t)*nCells;
	tally = (gpuTallyType_t*) malloc ( allocSize );

#ifdef __CUDACC__
	gpuTallyType_t* tally_device;
	CUDA_CHECK_RETURN( cudaMalloc( &tally_device, allocSize ));
	CUDA_CHECK_RETURN( cudaMemset(tally_device, 0, allocSize));
	gpuErrchk( cudaPeekAtLastError() );

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	testTallyCrossSection<N><<<nBlocks,nThreads>>>(pCP->getPtrPoints()->devicePtr, pMatList->ptr_device,
			matIndex, pMatList->getHashPtr()->getPtrDevice(), density, tally_device);
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);
	gpuErrchk( cudaPeekAtLastError() );

	CUDA_CHECK_RETURN(cudaMemcpy(tally, tally_device, allocSize, cudaMemcpyDeviceToHost));
	gpuErrchk( cudaPeekAtLastError() );
	cudaFree( tally_device );
#else
	testTallyCrossSection<N>(pCP->getPtrPoints(),
			                 pMatList->getPtr(),
			                 matIndex,
			                 pMatList->getHashPtr()->getPtr(),
			                 density,
			                 tally);
#endif
	return;
}


CUDA_CALLABLE_MEMBER
gpuFloatType_t getTotalXSByMatProp(
		const MonteRay_MaterialProperties_Data* matProps,
		const MonteRayMaterialList* pMatList,
		const HashLookup* pHash,
		unsigned HashBin,
		unsigned cell,
		gpuFloatType_t E)
{
	gpuFloatType_t total = 0.0f;
	for( unsigned i=0; i< getNumMats(matProps,cell); ++i) {
		gpuFloatType_t density = getDensity(matProps,cell,i);
		unsigned matID = getMatID(matProps,cell,i);
//		unsigned materialIndex = materialIDtoIndex(pMatList, matID);
		total += getTotalXS(pMatList, matID, pHash, HashBin, E, density);
	}
	return total;
}

gpuFloatType_t nonCudaGetTotalXSByMatProp(
		const MonteRay_MaterialProperties* matProps,
		const MonteRayMaterialList* pMatList,
		const HashLookup* pHash,
		unsigned HashBin,
		unsigned cell,
		gpuFloatType_t E)
{
	gpuFloatType_t total = 0.0f;
	for( unsigned i=0; i< matProps->getNumMaterials(cell); ++i) {
		gpuFloatType_t density = matProps->getMaterialDensity(cell,i);
		MonteRay_MaterialProperties::MatID_t matID = matProps->getMaterialID(cell,i);
		//unsigned materialIndex = materialIDtoIndex(pMatList, matID);
		total += getTotalXS(pMatList, matID, pHash, HashBin, E, density);
	}
	return total;
}

gpuFloatType_t nonCudaGetTotalXSByMatProp(
		const MonteRay_MaterialProperties* matProps,
		const MonteRayMaterialList* pMatList,
		unsigned cell,
		gpuFloatType_t E)
{
	gpuFloatType_t total = 0.0f;
	for( unsigned i=0; i< matProps->getNumMaterials(cell); ++i) {
		gpuFloatType_t density = matProps->getMaterialDensity(cell,i);
		unsigned matID = matProps->getMaterialID(cell,i);
		//unsigned materialIndex = materialIDtoIndex(pMatList, matID);
		total += getTotalXS(pMatList, matID, E, density);
	}
	return total;
}

template<unsigned N>
gpuFloatType_t
FIGenericGPUTestHelper<N>::getTotalXSByMatProp(
		const MonteRay_MaterialProperties* matProps,
		const MonteRayMaterialList* pMatList,
		const HashLookup* pHash,
		unsigned HashBin,
		unsigned cell,
		gpuFloatType_t E)
{
	return nonCudaGetTotalXSByMatProp( matProps, pMatList, pHash, HashBin, cell, E);
}

template<unsigned N>
gpuFloatType_t
FIGenericGPUTestHelper<N>::getTotalXSByMatProp(
		const MonteRay_MaterialProperties* matProps,
		const MonteRayMaterialList* pMatList,
		unsigned cell,
		gpuFloatType_t E)
{
	return nonCudaGetTotalXSByMatProp( matProps, pMatList, cell, E);
}

template<unsigned N>
CUDA_CALLABLE_KERNEL void
testTallyCrossSectionAtCollision(
		const RayList_t<N>* pCP,
		const MonteRayMaterialList* pMatList,
		const MonteRay_MaterialProperties_Data* pMatProps,
		const HashLookup* pHash,
		gpuTallyType_t* results)
{
#ifdef __CUDACC__
	unsigned tid = threadIdx.x + blockIdx.x*blockDim.x;
#else
	unsigned tid = 0;
#endif

	unsigned num = pCP->size();

	while( tid < num ) {
		gpuFloatType_t E = pCP->getEnergy(tid);
		unsigned HashBin = getHashBin( pHash, E);
		unsigned cell = pCP->getIndex(tid);

		results[tid] = getTotalXSByMatProp(pMatProps, pMatList, pHash, HashBin, cell, E);
#ifdef __CUDACC__
		tid += blockDim.x*gridDim.x;
#else
		tid++;
#endif
	}
	return;
}


template<unsigned N>
CUDA_CALLABLE_KERNEL void testSumCrossSectionAtCollisionLocation(
		const RayList_t<N>* pCP,
		const MonteRayMaterialList* pMatList,
		const MonteRay_MaterialProperties_Data* pMatProps,
		const HashLookup* pHash,
		gpuTallyType_t* results)
{
#ifdef __CUDACC__
	unsigned tid = threadIdx.x + blockIdx.x*blockDim.x;
#else
	unsigned tid = 0;
#endif

	unsigned num = pCP->size();

	while( tid < num ) {
		gpuFloatType_t E = pCP->getEnergy(tid);
		unsigned HashBin = getHashBin( pHash, E);
		unsigned cell = pCP->getIndex(tid);

		gpuTallyType_t value = getTotalXSByMatProp(pMatProps, pMatList, pHash, HashBin, cell, E);

		gpu_atomicAdd( &results[cell], value);
#ifdef __CUDACC__
		tid += blockDim.x*gridDim.x;
#else
		tid++;
#endif
	}
	return;
}

template<unsigned N>
void FIGenericGPUTestHelper<N>::launchTallyCrossSectionAtCollision(
		unsigned nBlocks, unsigned nThreads,
		const RayListInterface<N>* pCP,
		const MonteRayMaterialListHost* pMatList,
		const MonteRay_MaterialProperties* pMatProps )
{

	unsigned long long allocSize = sizeof(gpuTallyType_t)*nCells;
	tally = (gpuTallyType_t*) malloc ( allocSize );

#ifdef __CUDACC__
	gpuTallyType_t* tally_device;
	CUDA_CHECK_RETURN( cudaMalloc( &tally_device, allocSize ));
	CUDA_CHECK_RETURN( cudaMemset(tally_device, 0, allocSize));
	gpuErrchk( cudaPeekAtLastError() );

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	testTallyCrossSectionAtCollision<<<nBlocks,nThreads>>>(pCP->getPtrPoints()->devicePtr,
			pMatList->ptr_device, pMatProps->ptrData_device, pMatList->getHashPtr()->getPtrDevice(),
			tally_device);
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);
	gpuErrchk( cudaPeekAtLastError() );

	CUDA_CHECK_RETURN(cudaMemcpy(tally, tally_device, allocSize, cudaMemcpyDeviceToHost));
	gpuErrchk( cudaPeekAtLastError() );
	cudaFree( tally_device );
#else
	testTallyCrossSectionAtCollision(
			pCP->getPtrPoints(),
			pMatList->getPtr(),
			pMatProps->getPtr(),
			pMatList->getHashPtr()->getPtr(),
			tally);
#endif
	return;
}

template<unsigned N>
void FIGenericGPUTestHelper<N>::launchSumCrossSectionAtCollisionLocation(
		unsigned nBlocks,
		unsigned nThreads,
		const RayListInterface<N>* pCP,
		const MonteRayMaterialListHost* pMatList,
		const MonteRay_MaterialProperties* pMatProps )
{

	unsigned long long allocSize = sizeof(gpuTallyType_t)*nCells;
	tally = (gpuTallyType_t*) malloc ( allocSize );

#ifdef __CUDACC__
	gpuTallyType_t* tally_device;
	CUDA_CHECK_RETURN( cudaMalloc( &tally_device, allocSize ));
	CUDA_CHECK_RETURN( cudaMemset(tally_device, 0, allocSize));
	gpuErrchk( cudaPeekAtLastError() );

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	testSumCrossSectionAtCollisionLocation<<<nBlocks,nThreads>>>(pCP->getPtrPoints()->devicePtr,
			pMatList->ptr_device, pMatProps->ptrData_device, pMatList->getHashPtr()->getPtrDevice(),
			tally_device);
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);
	gpuErrchk( cudaPeekAtLastError() );

	CUDA_CHECK_RETURN(cudaMemcpy(tally, tally_device, allocSize, cudaMemcpyDeviceToHost));
	gpuErrchk( cudaPeekAtLastError() );
	cudaFree( tally_device );
#else
	testSumCrossSectionAtCollisionLocation(
			pCP->getPtrPoints(),
			pMatList->getPtr(),
			pMatProps->getPtr(),
			pMatList->getHashPtr()->getPtr(),
			tally);
#endif

	return;
}

template<unsigned N>
void FIGenericGPUTestHelper<N>::launchRayTraceTally(
		unsigned nBlocks,
		unsigned nThreads,
		const RayListInterface<N>* pCP,
		const MonteRayMaterialListHost* pMatList,
		const MonteRay_MaterialProperties* pMatProps )
{

	unsigned long long allocSize = sizeof(gpuTallyType_t)*nCells;
	tally = (gpuTallyType_t*) malloc ( allocSize );
	for( unsigned i = 0; i < nCells; ++i ) {
		tally[i] = 0.0;
	}

#ifdef __CUDACC__
	gpuTallyType_t* tally_device;
	CUDA_CHECK_RETURN( cudaMalloc( &tally_device, allocSize ));
	CUDA_CHECK_RETURN(cudaMemcpy(tally_device, tally, allocSize, cudaMemcpyHostToDevice));

	gpuErrchk( cudaPeekAtLastError() );

	cudaEvent_t sync;
	cudaEventCreate(&sync);
	rayTraceTally<<<nBlocks,nThreads>>>(grid_device, pCP->getPtrPoints()->devicePtr,
			pMatList->ptr_device, pMatProps->ptrData_device, pMatList->getHashPtr()->getPtrDevice(),
			tally_device);
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);
	gpuErrchk( cudaPeekAtLastError() );

	CUDA_CHECK_RETURN(cudaMemcpy(tally, tally_device, allocSize, cudaMemcpyDeviceToHost));
	gpuErrchk( cudaPeekAtLastError() );
	cudaFree( tally_device );
#else
	rayTraceTally(
			grid_device,
			pCP->getPtrPoints(),
			pMatList->getPtr(),
			pMatProps->getPtr(),
			pMatList->getHashPtr()->getPtr(),
			tally);
#endif
	return;
}

template<unsigned N>
void FIGenericGPUTestHelper<N>::copyGridtoGPU( const GridBins* grid){
	// allocate and copy the grid
#ifdef __CUDACC__
	CUDA_CHECK_RETURN( cudaMalloc( &grid_device, sizeof(GridBins) ));
	CUDA_CHECK_RETURN( cudaMemcpy(grid_device, grid, sizeof(GridBins), cudaMemcpyHostToDevice ));
#else
	grid_device = (GridBins*) malloc( sizeof(GridBins) );
	memcpy( grid_device, grid, sizeof(GridBins) );
#endif

	nCells = grid->getNumCells();
}

template class FIGenericGPUTestHelper<1>;
template class FIGenericGPUTestHelper<3>;


