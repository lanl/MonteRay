#ifndef GPUDISTANCECALCULATOR_TEST_HELPER_HH_
#define GPUDISTANCECALCULATOR_TEST_HELPER_HH_

#include <memory>

#include "MonteRayDefinitions.hh"
#include "MonteRayConstants.hh"
#include "MonteRay_timer.hh"
#include "RayWorkInfo.hh"
#include "GPUErrorCheck.hh"
#include "GPUUtilityFunctions.hh"
#include "RayWorkInfo.hh"

namespace MonteRay{

template <typename Geometry, typename Position, typename Direction>
CUDA_CALLABLE_KERNEL kernelRayTrace(
			RayWorkInfo* pRayInfo,
			Geometry* ptrGrid,
      Position pos,
      Direction dir,
			gpuFloatType_t distance,
			bool outsideDistances) {
    ptrGrid->rayTrace(0U, *pRayInfo, pos, dir, distance, outsideDistances);
}

class gpuDistanceCalculatorTestHelper
{
public:

    gpuDistanceCalculatorTestHelper();

    ~gpuDistanceCalculatorTestHelper();

    void gpuCheck();

    void setupTimers();

    void stopTimers();

    //	void launchGetDistancesToAllCenters( unsigned nBlocks, unsigned nThreads, const Position_t& pos);
    template <typename Geometry, typename Position_t, typename Direction_t>
    void launchRayTrace( const Position_t& pos, const Direction_t& dir, gpuRayFloat_t distance, bool outsideDistances, const Geometry* const pGeometry) {

#ifdef __CUDACC__
      cudaEvent_t sync;
      cudaEventCreate(&sync);
      kernelRayTrace<<<1,1>>>(
              pRayInfo.get(),
              pGeometry,
              pos,
              dir,
              distance,
              outsideDistances );
      gpuErrchk( cudaPeekAtLastError() );

      cudaEventRecord(sync, 0);
      cudaEventSynchronize(sync);
#else
      kernelRayTrace(
              pRayInfo.get(),
              pGeometry,
              pos,
              dir,
              distance,
              outsideDistances );
#endif

}





    auto getDistancesFromGPU( ) {
      std::vector<gpuRayFloat_t> distances(pRayInfo->getRayCastSize(0));
      for( unsigned i = 0; i < pRayInfo->getRayCastSize(0); ++i ){
          distances[i] = pRayInfo->getRayCastDist(0,i);
      }
      return distances;
    }

    auto getCellsFromCPU( ) {
        std::vector<int> cells(pRayInfo->getRayCastSize(0));
        for( unsigned i = 0; i < pRayInfo->getRayCastSize(0); ++i ){
            cells[i] = pRayInfo->getRayCastCell(0,i);
        }
        return cells;
    }

    unsigned getNumCrossingsFromGPU(void);

    std::unique_ptr< RayWorkInfo > pRayInfo;
private:

    unsigned nCells;

#ifdef __CUDACC__
    cudaEvent_t start, stop;
#else
    cpuTimer timer;
#endif

};

}
#endif /* GPUDISTANCECALCULATOR_TEST_HELPER_HH_ */

