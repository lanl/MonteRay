#ifndef FI_TEST_GENERICGPU_TEST_HELPER_HH_
#define FI_TEST_GENERICGPU_TEST_HELPER_HH_

#include "MonteRayConstants.hh"
#include "MonteRayDefinitions.hh"

#include "RayListInterface.hh"
#include "CrossSection.hh"
#include "MaterialList.hh"
#include "MaterialProperties.hh"
#include "MonteRay_SpatialGrid.hh"
#include "GPUUtilityFunctions.hh"

#ifndef __CUDACC__
#include "MonteRay_timer.hh"
#endif

using namespace MonteRay;

template<unsigned N = 1>
class FIGenericGPUTestHelper
{
public:

    FIGenericGPUTestHelper(unsigned nCells);

    ~FIGenericGPUTestHelper();

    void setupTimers();

    void stopTimers();

    void launchTallyCrossSection(unsigned nBlocks, unsigned nThreads, const RayListInterface<N>* pCP, const CrossSection* pXS );
    void launchTallyCrossSection(unsigned nBlocks, unsigned nThreads, const RayListInterface<N>* pCP, const MaterialList* pMatList, unsigned matIndex, gpuFloatType_t density );
    void launchTallyCrossSectionAtCollision(unsigned nBlocks, unsigned nThreads, const RayListInterface<N>* pCP, const MaterialList* pMatList, const MaterialProperties* pMatProps );

    gpuFloatType_t getTotalXSByMatProp(const MaterialProperties* matProps, const MaterialList* pMatList, unsigned HashBin, unsigned cell, gpuFloatType_t E);
    gpuFloatType_t getTotalXSByMatProp(const MaterialProperties* matProps, const MaterialList* pMatList, unsigned cell, gpuFloatType_t E);

    void launchSumCrossSectionAtCollisionLocation(unsigned nBlocks, unsigned nThreads, const RayListInterface<N>* pCP, const MaterialList* pMatList, const MaterialProperties* pMatProps );

    template<typename Geometry>
    void launchRayTraceTally(
            unsigned nBlocks,
            unsigned nThreads,
            const RayListInterface<N>* pCP,
            const MaterialList* pMatList,
            const MaterialProperties* pMatProps,
            const Geometry* pGrid)
    {
#ifdef __CUDACC__
      gpuErrchk( cudaPeekAtLastError() );
#endif

      unsigned long long allocSize = sizeof(gpuTallyType_t)*nCells;
      tally = (gpuTallyType_t*) malloc ( allocSize );
      for( unsigned i = 0; i < nCells; ++i ) {
          tally[i] = 0.0;
      }

#ifdef __CUDACC__
      gpuTallyType_t* tally_device;
      CUDA_CHECK_RETURN( cudaMalloc( &tally_device, allocSize ));
      CUDA_CHECK_RETURN(cudaMemcpy(tally_device, tally, allocSize, cudaMemcpyHostToDevice));

      std::cout << "Debug: FIGenericGPUTestHelper::launchRayTraceTally, requesting kernel with " <<
                      nBlocks << " blocks, " << nThreads << " threads, nBlocks*nThreads= " <<
                      nBlocks*nThreads << ", to process " << pCP->getPtrPoints()->size() << "rays. \n";

      auto launchBounds = setLaunchBounds( nThreads, nBlocks,  pCP->getPtrPoints()->size() );
      nThreads = launchBounds.second;
      nBlocks = launchBounds.first;

      std::cout << "Debug: FIGenericGPUTestHelper::launchRayTraceTally, launching kernel with " <<
                   nBlocks << " blocks, " << nThreads << " threads, nBlocks*nThreads= " <<
                   nBlocks*nThreads << "\n";

      RayWorkInfo rayInfo( nThreads*nBlocks );
      auto pRayInfo = std::make_unique<RayWorkInfo>(nThreads*nBlocks);

      gpuErrchk( cudaPeekAtLastError() );

      cudaEvent_t sync;
      cudaEventCreate(&sync);
      rayTraceTally<<<nBlocks,nThreads>>>(
              pGrid,
              pCP->getPtrPoints(),
              pMatList,
              pMatProps,
              pRayInfo.get(),
              tally_device);
      cudaEventRecord(sync, 0);
      cudaEventSynchronize(sync);
      gpuErrchk( cudaPeekAtLastError() );

      CUDA_CHECK_RETURN(cudaMemcpy(tally, tally_device, allocSize, cudaMemcpyDeviceToHost));
      gpuErrchk( cudaPeekAtLastError() );
      cudaFree( tally_device );
#else
      RayWorkInfo rayInfo( 1 );

      rayTraceTally(
              pGrid,
              pCP->getPtrPoints(),
              pMatList,
              pMatProps,
              &rayInfo,
              tally);
#endif
      return;
    }




    gpuFloatType_t getTally(unsigned i) const { return tally[i]; }

private:

#ifdef __CUDACC__
    cudaEvent_t start, stop;
#else
    cpuTimer timer;
#endif

	unsigned nCells;

	gpuTallyType_t* tally = nullptr;


};

#endif /* FI_TEST_GENERICGPU_TEST_HELPER_HH_ */

