#include "MonteRayDefinitions.hh"
#include "GridBins.h"

namespace MonteRay{

CUDA_CALLABLE_MEMBER unsigned cudaCalcIndex(const GridBins* const grid, uint3& indices);
CUDA_CALLABLE_MEMBER unsigned cudaCalcIndex(const GridBins* const grid, unsigned* indices);
CUDA_CALLABLE_MEMBER unsigned cudaCalcIndex(const GridBins* const grid, int* indices);

CUDA_CALLABLE_MEMBER void cudaCalcIJK(const GridBins* const grid, unsigned index, uint3& indices );

CUDA_CALLABLE_MEMBER float_t cudaGetVertex(const GridBins* const grid, unsigned dim, unsigned index );

CUDA_CALLABLE_MEMBER  void cudaGetCenterPointByIndices(const GridBins* const grid, uint3& indices,  float3_t& pos );

CUDA_CALLABLE_MEMBER float_t cudaGetDistance( float3_t& pos1, float3_t& pos2);

CUDADEVICE_CALLABLE_MEMBER void cudaGetDistancesToAllCenters2(const GridBins* const grid, float_t* distances, float3_t pos);

CUDA_CALLABLE_KERNEL void kernelGetDistancesToAllCenters(void* ptrGrid,  void* ptrDistances, float_t x, float_t y, float_t z);

CUDA_CALLABLE_MEMBER float_t cudaMin(const GridBins* const grid, unsigned dim);

CUDA_CALLABLE_MEMBER float_t cudaMax(const GridBins* const grid, unsigned dim);

CUDA_CALLABLE_MEMBER unsigned cudaGetNumVertices(const GridBins* const grid, unsigned dim);

CUDA_CALLABLE_MEMBER unsigned cudaGetNumBins(const GridBins* const grid, unsigned dim);

CUDA_CALLABLE_MEMBER unsigned cudaGetNumBins(const GridBins* const grid, unsigned dim, unsigned index);

CUDA_CALLABLE_MEMBER unsigned cudaGetIndexBinaryFloat(const float_t* const values, unsigned count, float_t value );

CUDA_CALLABLE_MEMBER int cudaGetDimIndex(const GridBins* const grid, unsigned dim, float_t pos );

CUDA_CALLABLE_MEMBER bool cudaIsIndexOutside(const GridBins* const grid, unsigned dim, int i);


CUDA_CALLABLE_MEMBER bool cudaIsOutside(const GridBins* const grid, const int3& indices );
CUDA_CALLABLE_MEMBER bool cudaIsOutside(const GridBins* const grid, uint1* indices );
CUDA_CALLABLE_MEMBER bool cudaIsOutside(const GridBins* const grid, int* indices );


CUDA_CALLABLE_MEMBER unsigned cudaGetIndex(const GridBins* const grid, const float3_t& pos);

}
