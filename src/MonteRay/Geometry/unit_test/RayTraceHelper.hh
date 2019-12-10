#ifndef MONTERAY_RAYTRACE_HELPER_HH_
#define MONTERAY_RAYTRACE_HELPER_HH_

#include "RayWorkInfo.hh"
#include "GPUUtilityFunctions.hh"
#include "MonteRay_GridSystemInterface.hh"

namespace MonteRay{

template <typename Grid, typename Position, typename Direction>
CUDA_CALLABLE_KERNEL  kernelRayTrace(const Grid* pGrid, RayWorkInfo* pRayInfo,
    const Position pos, const Direction dir, gpuRayFloat_t distance, bool outside) {
    pGrid->rayTrace( 0U, *pRayInfo, pos, dir, distance, outside);
}

template <typename Grid, typename Position, typename Direction>
rayTraceList_t rayTrace(const Grid* pGrid, Position pos, Direction dir, gpuRayFloat_t distance, bool outside=false ) {
  auto pRayInfo = std::make_unique<RayWorkInfo>(1);

#ifdef __CUDACC__

  cudaDeviceSynchronize();
  kernelRayTrace<<<1,1>>>( pGrid, pRayInfo.get(), pos, dir, distance, outside );
  cudaDeviceSynchronize();

  gpuErrchk( cudaPeekAtLastError() );

#else
  kernelRayTrace( pGrid, pRayInfo.get(), pos, dir, distance, outside );
#endif

  rayTraceList_t rayTraceList;
  for( unsigned i = 0; i < pRayInfo->getRayCastSize(0); ++i ) {
      rayTraceList.add( pRayInfo->getRayCastCell(0,i), pRayInfo->getRayCastDist(0,i) );
  }
  return rayTraceList;
}

} // end namespace MonteRay

#endif
