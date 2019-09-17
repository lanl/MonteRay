#include "MonteRay_SpatialGrid_GPU_helper.hh"

namespace MonteRay_SpatialGrid_helper {

using namespace MonteRay;

typedef MonteRay_SpatialGrid Grid_t;
using Position_t = MonteRay_SpatialGrid::Position_t;

template<typename T>
using resultClass = MonteRay_SingleValueCopyMemory<T>;

CUDA_CALLABLE_KERNEL  kernelGetNumCells(Grid_t* pSpatialGrid, resultClass<unsigned>* pResult) {
    pResult->v = pSpatialGrid->getNumCells();
}

CUDA_CALLABLE_KERNEL  kernelGetCoordinateSystem(Grid_t* pSpatialGrid, resultClass<TransportMeshTypeEnum::TransportMeshTypeEnum_t>* pResult) {
    pResult->v = pSpatialGrid->getCoordinateSystem();
}

CUDA_CALLABLE_KERNEL  kernelGetDimension(Grid_t* pSpatialGrid, resultClass<unsigned>* pResult) {
    pResult->v = pSpatialGrid->getDimension();
}

CUDA_CALLABLE_KERNEL  kernelIsInitialized(Grid_t* pSpatialGrid, resultClass<bool>* pResult) {
    pResult->v = pSpatialGrid->isInitialized();
}

CUDA_CALLABLE_KERNEL  kernelGetNumGridBins(Grid_t* pSpatialGrid, resultClass<unsigned>* pResult, unsigned index) {
    pResult->v = pSpatialGrid->getNumGridBins(index);
}

CUDA_CALLABLE_KERNEL  kernelGetMinVertex(Grid_t* pSpatialGrid, resultClass<gpuRayFloat_t>* pResult, unsigned index) {
    pResult->v = pSpatialGrid->getMinVertex(index);
}

CUDA_CALLABLE_KERNEL  kernelGetMaxVertex(Grid_t* pSpatialGrid, resultClass<gpuRayFloat_t>* pResult, unsigned index) {
    pResult->v = pSpatialGrid->getMaxVertex(index);
}

CUDA_CALLABLE_KERNEL  kernelGetDelta(Grid_t* pSpatialGrid, resultClass<gpuRayFloat_t>* pResult, unsigned index) {
    pResult->v = pSpatialGrid->getDelta(index);
}

CUDA_CALLABLE_KERNEL  kernelGetVertex(Grid_t* pSpatialGrid, resultClass<gpuRayFloat_t>* pResult, unsigned d, unsigned index) {
    pResult->v = pSpatialGrid->getVertex(d,index);
}

CUDA_CALLABLE_KERNEL  kernelGetVolume(Grid_t* pSpatialGrid, resultClass<gpuRayFloat_t>* pResult, unsigned index) {
    pResult->v = pSpatialGrid->getVolume(index);
}

CUDA_CALLABLE_KERNEL  kernelGetIndex(Grid_t* pSpatialGrid, resultClass<unsigned>* pResult, Position_t pos ) {
    pResult->v = pSpatialGrid->getIndex(pos);
}

CUDA_CALLABLE_KERNEL  kernelRayTrace(Grid_t* pSpatialGrid, RayWorkInfo* pRayInfo,
        gpuRayFloat_t x, gpuRayFloat_t y, gpuRayFloat_t z, gpuRayFloat_t u, gpuRayFloat_t v, gpuRayFloat_t w,
        gpuRayFloat_t distance, bool outside) {
    Position_t pos = Position_t( x,y,z);
    Position_t dir = Position_t( u,v,w);
    pSpatialGrid->rayTrace( 0U, *pRayInfo, pos, dir, distance, outside);
}

CUDA_CALLABLE_KERNEL  kernelCrossingDistance(Grid_t* pSpatialGrid, RayWorkInfo* pRayInfo,
        unsigned d, gpuRayFloat_t pos, gpuRayFloat_t dir, gpuRayFloat_t distance ) {
    pSpatialGrid->crossingDistance( d, 0, *pRayInfo, pos, dir, distance);
}

CUDA_CALLABLE_KERNEL  kernelCrossingDistance(Grid_t* pSpatialGrid, RayWorkInfo* pRayInfo,
        unsigned d, Position_t pos, Position_t dir, gpuRayFloat_t distance ) {
    pSpatialGrid->crossingDistance( d, 0, *pRayInfo, pos, dir, distance);
}

}

