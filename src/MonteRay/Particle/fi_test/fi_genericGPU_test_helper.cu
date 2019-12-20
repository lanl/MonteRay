#include "MonteRayDefinitions.hh"
#include "GPUErrorCheck.hh"
#include "GPUAtomicAdd.hh"
#include "ExpectedPathLength.t.hh"
#include "RayList.hh"

#include "fi_genericGPU_test_helper.hh"

template<unsigned N>
FIGenericGPUTestHelper<N>::FIGenericGPUTestHelper(unsigned num){
    nCells = num;
}

template<unsigned N>
FIGenericGPUTestHelper<N>::~FIGenericGPUTestHelper(){
    if( tally ) {
        free( tally );
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
CUDA_CALLABLE_KERNEL  testTallyCrossSection(const RayList_t<N>* pCP, const CrossSection* pXS, gpuTallyType_t* results){

#ifdef __CUDACC__
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
#else
    int tid = 0;
#endif

    int num = pCP->size();
    while( tid < num ) {
        gpuFloatType_t E = pCP->getEnergy(tid);
        results[tid] = pXS->getTotalXS(E);
#ifdef __CUDACC__
        tid += blockDim.x*gridDim.x;
#else
        tid++;
#endif
    }
    return;
}

template CUDA_CALLABLE_KERNEL 
testTallyCrossSection<1>(const RayList_t<1>* pCP, const CrossSection* pXS, gpuTallyType_t* results);

template CUDA_CALLABLE_KERNEL 
testTallyCrossSection<3>(const RayList_t<3>* pCP, const CrossSection* pXS, gpuTallyType_t* results);

template<unsigned N>
void FIGenericGPUTestHelper<N>::launchTallyCrossSection(
        unsigned nBlocks, unsigned nThreads,
        const RayListInterface<N>* pCP,
        const CrossSection* pXS )
        {
    unsigned long long allocSize = sizeof(gpuTallyType_t)*nCells;

    if( tally ) {
        free( tally );
    }
    tally = (gpuTallyType_t*) malloc ( allocSize );
    for( unsigned i=0; i<nCells; ++i) {
        tally[i] = 0.0;
    }

#ifdef __CUDACC__
    auto launchBounds = setLaunchBounds( nThreads, nBlocks, pCP->getPtrPoints()->size() );
    nBlocks = launchBounds.first;
    nThreads = launchBounds.second;

    gpuTallyType_t* tally_device;
    CUDA_CHECK_RETURN( cudaMalloc( &tally_device, allocSize ));
    CUDA_CHECK_RETURN( cudaMemset(tally_device, 0, allocSize));
    gpuErrchk( cudaPeekAtLastError() );

    cudaEvent_t sync;
    cudaEventCreate(&sync);
    testTallyCrossSection<N><<<nBlocks,nThreads>>>(pCP->getPtrPoints(), pXS, tally_device);
    cudaEventRecord(sync, 0);
    cudaEventSynchronize(sync);
    gpuErrchk( cudaPeekAtLastError() );

    CUDA_CHECK_RETURN(cudaMemcpy(tally, tally_device, allocSize, cudaMemcpyDeviceToHost));
    gpuErrchk( cudaPeekAtLastError() );
    cudaFree( tally_device );
#else
    testTallyCrossSection<N>(pCP->getPtrPoints(), pXS, tally);
#endif
    return;
        }

template< unsigned N>
CUDA_CALLABLE_KERNEL  testTallyCrossSection(
        const RayList_t<N>* pCP,
        const MaterialList* pMatList,
        unsigned matIndex,
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
        results[tid] = pMatList->material(matIndex).getTotalXS(E, density);
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
        const MaterialList* pMatList,
        unsigned matIndex,
        gpuFloatType_t density )
        {
    unsigned long long allocSize = sizeof(gpuTallyType_t)*nCells;
    tally = (gpuTallyType_t*) malloc ( allocSize );

#ifdef __CUDACC__
    auto launchBounds = setLaunchBounds( nThreads, nBlocks, pCP->getPtrPoints()->size() );
    nBlocks = launchBounds.first;
    nThreads = launchBounds.second;

    gpuTallyType_t* tally_device;
    CUDA_CHECK_RETURN( cudaMalloc( &tally_device, allocSize ));
    CUDA_CHECK_RETURN( cudaMemset(tally_device, 0, allocSize));
    gpuErrchk( cudaPeekAtLastError() );

    cudaEvent_t sync;
    cudaEventCreate(&sync);
    testTallyCrossSection<N><<<nBlocks,nThreads>>>(pCP->getPtrPoints(), pMatList,
            matIndex, density, tally_device);
    cudaEventRecord(sync, 0);
    cudaEventSynchronize(sync);
    gpuErrchk( cudaPeekAtLastError() );

    CUDA_CHECK_RETURN(cudaMemcpy(tally, tally_device, allocSize, cudaMemcpyDeviceToHost));
    gpuErrchk( cudaPeekAtLastError() );
    cudaFree( tally_device );
#else
    testTallyCrossSection<N>(pCP->getPtrPoints(),
            pMatList,
            matIndex,
            density,
            tally);
#endif
return;
        }


CUDA_CALLABLE_MEMBER
gpuFloatType_t getTotalXSByMatProp(
        const MaterialProperties* pMatProps,
        const MaterialList* pMatList,
        unsigned cell,
        gpuFloatType_t E)
{
    gpuFloatType_t total = 0.0f;
    for( unsigned i=0; i< pMatProps->numMaterials(cell); ++i) {
        gpuFloatType_t density = pMatProps->getMaterialDensity(cell,i);
        unsigned matID = pMatProps->getMaterialID(cell,i);
        //		unsigned materialIndex = materialIDtoIndex(pMatList, matID);
        total += pMatList->material(matID).getTotalXS(E, density);
    }
    return total;
}

gpuFloatType_t nonCudaGetTotalXSByMatProp(
        const MaterialProperties* pMatProps,
        const MaterialList* pMatList,
        unsigned cell,
        gpuFloatType_t E)
{
    gpuFloatType_t total = 0.0f;
    for( unsigned i=0; i< pMatProps->numMaterials(cell); ++i) {
        gpuFloatType_t density = pMatProps->getMaterialDensity(cell,i);
        MaterialProperties::MatID_t matID = pMatProps->getMaterialID(cell,i);
        //unsigned materialIndex = materialIDtoIndex(pMatList, matID);
        total += pMatList->material(matID).getTotalXS(E, density);
    }
    return total;
}

template<unsigned N>
gpuFloatType_t
FIGenericGPUTestHelper<N>::getTotalXSByMatProp(
        const MaterialProperties* pMatProps,
        const MaterialList* pMatList,
        unsigned cell,
        gpuFloatType_t E)
        {
    return nonCudaGetTotalXSByMatProp( pMatProps, pMatList, cell, E);
        }

template<unsigned N>
CUDA_CALLABLE_KERNEL 
testTallyCrossSectionAtCollision(
        const RayList_t<N>* pCP,
        const MaterialList* pMatList,
        const MaterialProperties* pMatProps,
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
        unsigned cell = pCP->getIndex(tid);

        results[tid] = getTotalXSByMatProp(pMatProps, pMatList, cell, E);
#ifdef __CUDACC__
        tid += blockDim.x*gridDim.x;
#else
        tid++;
#endif
    }
    return;
}


template<unsigned N>
CUDA_CALLABLE_KERNEL  testSumCrossSectionAtCollisionLocation(
        const RayList_t<N>* pCP,
        const MaterialList* pMatList,
        const MaterialProperties* pMatProps,
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
        unsigned cell = pCP->getIndex(tid);

        gpuTallyType_t value = getTotalXSByMatProp(pMatProps, pMatList, cell, E);

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
        const MaterialList* pMatList,
        const MaterialProperties* pMatProps )
        {

    unsigned long long allocSize = sizeof(gpuTallyType_t)*nCells;
    tally = (gpuTallyType_t*) malloc ( allocSize );

#ifdef __CUDACC__
    auto launchBounds = setLaunchBounds( nThreads, nBlocks, pCP->getPtrPoints()->size() );
    nBlocks = launchBounds.first;
    nThreads = launchBounds.second;

    gpuTallyType_t* tally_device;
    CUDA_CHECK_RETURN( cudaMalloc( &tally_device, allocSize ));
    CUDA_CHECK_RETURN( cudaMemset(tally_device, 0, allocSize));
    gpuErrchk( cudaPeekAtLastError() );

    cudaEvent_t sync;
    cudaEventCreate(&sync);
    testTallyCrossSectionAtCollision<<<nBlocks,nThreads>>>(pCP->getPtrPoints(),
            pMatList, pMatProps, tally_device);
    cudaEventRecord(sync, 0);
    cudaEventSynchronize(sync);
    gpuErrchk( cudaPeekAtLastError() );

    CUDA_CHECK_RETURN(cudaMemcpy(tally, tally_device, allocSize, cudaMemcpyDeviceToHost));
    gpuErrchk( cudaPeekAtLastError() );
    cudaFree( tally_device );
#else
    testTallyCrossSectionAtCollision(
            pCP->getPtrPoints(),
            pMatList,
            pMatProps,
            tally);
#endif
    return;
        }

template<unsigned N>
void FIGenericGPUTestHelper<N>::launchSumCrossSectionAtCollisionLocation(
        unsigned nBlocks,
        unsigned nThreads,
        const RayListInterface<N>* pCP,
        const MaterialList* pMatList,
        const MaterialProperties* pMatProps )
  {

    unsigned long long allocSize = sizeof(gpuTallyType_t)*nCells;
    if( tally ) {
        free( tally );
    }
    tally = (gpuTallyType_t*) malloc ( allocSize );
    for( unsigned i=0; i < nCells; ++i ) {
        tally[i] = 0.0;
    }

#ifdef __CUDACC__
    auto launchBounds = setLaunchBounds( nThreads, nBlocks, pCP->getPtrPoints()->size() );
    nBlocks = launchBounds.first;
    nThreads = launchBounds.second;

    gpuTallyType_t* tally_device;
    CUDA_CHECK_RETURN( cudaMalloc( &tally_device, allocSize ));
    CUDA_CHECK_RETURN( cudaMemset(tally_device, 0, allocSize));
    gpuErrchk( cudaPeekAtLastError() );

    cudaEvent_t sync;
    cudaEventCreate(&sync);
    testSumCrossSectionAtCollisionLocation<<<nBlocks,nThreads>>>(pCP->getPtrPoints(),
            pMatList, pMatProps, tally_device);
    cudaEventRecord(sync, 0);
    cudaEventSynchronize(sync);
    gpuErrchk( cudaPeekAtLastError() );

    CUDA_CHECK_RETURN(cudaMemcpy(tally, tally_device, allocSize, cudaMemcpyDeviceToHost));
    gpuErrchk( cudaPeekAtLastError() );
    cudaFree( tally_device );
#else
    testSumCrossSectionAtCollisionLocation(
            pCP->getPtrPoints(),
            pMatList,
            pMatProps,
            tally);
#endif

    return;
  }


template class FIGenericGPUTestHelper<1>;
template class FIGenericGPUTestHelper<3>;


