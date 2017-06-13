#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "GridBins.h"

namespace MonteRay{

#ifdef CUDA
typedef float1 float1_t;
typedef float3 float3_t;

__device__ unsigned cudaCalcIndex(const GridBins* const grid, uint3& indices);
__device__ unsigned cudaCalcIndex(const GridBins* const grid, unsigned* indices);
__device__ unsigned cudaCalcIndex(const GridBins* const grid, int* indices);

__device__ void cudaCalcIJK(const GridBins* const grid, unsigned index, uint3& indices );

__device__ float_t cudaGetVertex(const GridBins* const grid, unsigned dim, unsigned index );

__device__  void cudaGetCenterPointByIndices(const GridBins* const grid, uint3& indices,  float3_t& pos );

__device__ float_t cudaGetDistance( float3_t& pos1, float3_t& pos2);

__device__ void cudaGetDistancesToAllCenters(const GridBins* const grid, float_t* distances, float3_t pos);

__device__ void cudaGetDistancesToAllCenters2(const GridBins* const grid, float_t* distances, float3_t pos);

__global__ void kernelGetDistancesToAllCenters(void* ptrGrid,  void* ptrDistances, float_t x, float_t y, float_t z);

__device__ float_t cudaMin(const GridBins* const grid, unsigned dim);

__device__ float_t cudaMax(const GridBins* const grid, unsigned dim);

__device__ unsigned cudaGetNumVertices(const GridBins* const grid, unsigned dim);

__device__ unsigned cudaGetNumBins(const GridBins* const grid, unsigned dim);

__device__ unsigned cudaGetNumBins(const GridBins* const grid, unsigned dim, unsigned index);

__device__ unsigned cudaGetIndexBinaryFloat(const float_t* const values, unsigned count, float_t value );

__device__ int cudaGetDimIndex(const GridBins* const grid, unsigned dim, float_t pos );

__device__ bool cudaIsIndexOutside(const GridBins* const grid, unsigned dim, int i);


__device__ bool cudaIsOutside(const GridBins* const grid, const int3& indices );
__device__ bool cudaIsOutside(const GridBins* const grid, uint1* indices );
__device__ bool cudaIsOutside(const GridBins* const grid, int* indices );


__device__ unsigned cudaGetIndex(const GridBins* const grid, const float3_t& pos);
#endif

}
