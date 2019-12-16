#ifndef MONTERAY_CROSSINGDISTANCE_HELPER_HH_
#define MONTERAY_CROSSINGDISTANCE_HELPER_HH_

#include "RayWorkInfo.hh"
#include "GPUUtilityFunctions.hh"
#include "MonteRay_GridSystemInterface.hh"

namespace MonteRay{

template <typename Position, typename Direction, typename Grid>
CUDA_CALLABLE_KERNEL  kernelCrossingDistance(const Grid* pGrid, RayWorkInfo* pRayInfo,
        unsigned d, Position pos, Direction dir, gpuRayFloat_t distance ) {
    pGrid->crossingDistance( d, 0, *pRayInfo, pos, dir, distance);
}

template <typename Position, typename Direction, typename Grid>
singleDimRayTraceMap_t crossingDistanceOnCPU( const Grid* pGrid, unsigned d, const Position& pos, const Direction& dir, gpuRayFloat_t distance) {
  auto pRayInfo = std::make_unique<RayWorkInfo>(1);
  pGrid->crossingDistance(d, 0, *pRayInfo, pos, dir, distance);
  return singleDimRayTraceMap_t( *pRayInfo, 0, d );
}

template <typename Position, typename Direction, typename Grid>
singleDimRayTraceMap_t crossingDistanceOnGPU( const Grid* pGrid, unsigned d, const Position& pos, const Direction& dir, gpuRayFloat_t distance) {
  auto pRayInfo = std::make_unique<RayWorkInfo>(1);
#ifdef __CUDACC__
  cudaDeviceSynchronize();
  kernelCrossingDistance<<<1,1>>>(
          pGrid,
          pRayInfo.get(),
          d, pos, dir, distance );
  cudaDeviceSynchronize();

  gpuErrchk( cudaPeekAtLastError() );
#endif
  return singleDimRayTraceMap_t( *pRayInfo, 0, d );
}

template <typename Position, typename Direction, typename Grid>
singleDimRayTraceMap_t crossingDistance( const Grid* pGrid, unsigned d, const Position& pos, const Direction& dir, gpuRayFloat_t distance) {
#ifdef __CUDACC__
  return crossingDistanceOnGPU(pGrid, d, pos, dir, distance);
#else
  return crossingDistanceOnCPU(pGrid, d, pos, dir, distance);
#endif
}

} // end namespace MonteRay

#endif
