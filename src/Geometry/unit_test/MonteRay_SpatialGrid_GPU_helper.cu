#include "MonteRay_SpatialGrid_GPU_helper.hh"

namespace MonteRay_SpatialGrid_helper {

using namespace MonteRay;

	typedef MonteRay_SpatialGrid Grid_t;
	using Position_t = MonteRay_SpatialGrid::Position_t;

	template<typename T>
	using resultClass = MonteRay_SingleValueCopyMemory<T>;

   	CUDA_CALLABLE_KERNEL void kernelGetNumCells(Grid_t* pSpatialGrid, resultClass<unsigned>* pResult) {
   		pResult->v = pSpatialGrid->getNumCells();
	}

   	CUDA_CALLABLE_KERNEL void kernelGetCoordinateSystem(Grid_t* pSpatialGrid, resultClass<TransportMeshTypeEnum::TransportMeshTypeEnum_t>* pResult) {
   		pResult->v = pSpatialGrid->getCoordinateSystem();
   	}

   	CUDA_CALLABLE_KERNEL void kernelGetDimension(Grid_t* pSpatialGrid, resultClass<unsigned>* pResult) {
   		pResult->v = pSpatialGrid->getDimension();
   	}

   	CUDA_CALLABLE_KERNEL void kernelIsInitialized(Grid_t* pSpatialGrid, resultClass<bool>* pResult) {
   		pResult->v = pSpatialGrid->isInitialized();
   	}

   	CUDA_CALLABLE_KERNEL void kernelGetNumGridBins(Grid_t* pSpatialGrid, resultClass<unsigned>* pResult, unsigned index) {
   		pResult->v = pSpatialGrid->getNumGridBins(index);
   	}

   	CUDA_CALLABLE_KERNEL void kernelGetMinVertex(Grid_t* pSpatialGrid, resultClass<gpuFloatType_t>* pResult, unsigned index) {
   		pResult->v = pSpatialGrid->getMinVertex(index);
   	}

   	CUDA_CALLABLE_KERNEL void kernelGetMaxVertex(Grid_t* pSpatialGrid, resultClass<gpuFloatType_t>* pResult, unsigned index) {
   		pResult->v = pSpatialGrid->getMaxVertex(index);
   	}

   	CUDA_CALLABLE_KERNEL void kernelGetDelta(Grid_t* pSpatialGrid, resultClass<gpuFloatType_t>* pResult, unsigned index) {
   		pResult->v = pSpatialGrid->getDelta(index);
   	}

   	CUDA_CALLABLE_KERNEL void kernelGetVertex(Grid_t* pSpatialGrid, resultClass<gpuFloatType_t>* pResult, unsigned d, unsigned index) {
   		pResult->v = pSpatialGrid->getVertex(d,index);
   	}

   	CUDA_CALLABLE_KERNEL void kernelGetVolume(Grid_t* pSpatialGrid, resultClass<gpuFloatType_t>* pResult, unsigned index) {
   		pResult->v = pSpatialGrid->getVolume(index);
   	}

   	CUDA_CALLABLE_KERNEL void kernelGetIndex(Grid_t* pSpatialGrid, resultClass<unsigned>* pResult, Position_t pos) {
   		pResult->v = pSpatialGrid->getIndex(pos);
   	}

   	CUDA_CALLABLE_KERNEL void kernelRayTrace(Grid_t* pSpatialGrid, resultClass<rayTraceList_t>* pResult, Position_t pos, Position_t dir, gpuFloatType_t distance) {
   		pSpatialGrid->rayTrace(pResult->v, pos, dir, distance);
   	}

}

