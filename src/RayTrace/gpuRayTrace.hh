#include "MonteRayDefinitions.hh"
#include "GridBins.hh"
#include "MonteRayConstants.hh"
#include "MonteRayVector3D.hh"

namespace MonteRay{

//CUDA_CALLABLE_MEMBER unsigned
//cudaCalcCrossings(
//		const float_t* const vertices,
//		unsigned nVertices,
//		int* cells,
//		gpuRayFloat_t* distances,
//		gpuRayFloat_t pos,
//		gpuRayFloat_t dir,
//		gpuRayFloat_t distance,
//		int index
//);
//
//CUDA_CALLABLE_MEMBER unsigned
//cudaOrderCrossings(
//		const GridBins* const grid,
//		int* global_indices,
//		gpuRayFloat_t* distances,
//		unsigned num,
//		const int* const cells,
//		const gpuRayFloat_t* const crossingDistances,
//		const uint3& numCrossings,
//		const int3& cudaindices,
//		gpuRayFloat_t distance,
//		bool outsideDistances
//);
//
//CUDA_CALLABLE_MEMBER unsigned
//cudaRayTrace(
//		const GridBins* const grid,
//		int* global_indices,
//		gpuRayFloat_t* distances,
//		const float3_t& pos,
//		const float3_t& dir,
//		gpuRayFloat_t distance,
//		bool outsideDistances
//);
//
//CUDA_CALLABLE_MEMBER unsigned
//cudaRayTrace(
//		const GridBins* const grid,
//		int* global_indices,
//		gpuRayFloat_t* distances,
//		const MonteRay::Vector3D<gpuRayFloat_t>& pos,
//		const MonteRay::Vector3D<gpuRayFloat_t>& dir,
//		gpuRayFloat_t distance,
//		bool outsideDistances
//);

CUDA_CALLABLE_KERNEL void kernelCudaRayTrace(void* ptrNumCrossings,
		                           GridBins* ptrGrid,
		                           int* ptrCells,
		                           gpuRayFloat_t* ptrDistances,
		                           float_t x, float_t y, float_t z,
		                           float_t u, float_t v, float_t w,
		                           float_t distance,
		                           bool outsideDistances);

//CUDA_CALLABLE_KERNEL void kernelCudaRayTraceToAllCenters(
//		                           void* ptrGrid,
//		                           void* ptrDistances,
//		                           float_t x, float_t y, float_t z);
//
//}

}
