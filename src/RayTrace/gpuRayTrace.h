#include <cuda.h>

#include "cudaGridBins.h"
#include "MonteRayConstants.hh"

namespace MonteRay{

#ifdef __CUDACC__
CUDA_CALLABLE_MEMBER unsigned
cudaCalcCrossings(const float_t* const vertices, unsigned nVertices,
				int* cells, float_t* distances, float_t pos, float_t dir,
				float_t distance, int index );

CUDA_CALLABLE_MEMBER unsigned
cudaOrderCrossings(const GridBins* const grid, int* global_indices,
		float_t* distances, unsigned num, const int* const cells,
		const float_t* const crossingDistances, const uint3& numCrossings,
		const int3& cudaindices, float_t distance,
		bool outsideDistances );

CUDA_CALLABLE_MEMBER unsigned
cudaRayTrace(const GridBins* const grid, int* global_indices, float_t* distances,
		const float3_t& pos, const float3_t& dir, float_t distance,
		bool outsideDistances);

CUDA_CALLABLE_KERNEL void kernelCudaRayTrace(void* ptrNumCrossings,
		                           void* ptrGrid,
		                           void* ptrCells,
		                           void* ptrDistances,
		                           float_t x, float_t y, float_t z,
		                           float_t u, float_t v, float_t w,
		                           float_t distance,
		                           bool outsideDistances);

CUDA_CALLABLE_KERNEL void kernelCudaRayTraceToAllCenters(
		                           void* ptrGrid,
		                           void* ptrDistances,
		                           float_t x, float_t y, float_t z);
#endif

}
