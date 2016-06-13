#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "cudaGridBins.h"

__device__ unsigned cudaCalcIndex(const GridBins* const grid, uint3& indices) {
    return indices.x + indices.y*grid->num[0] + indices.z*(grid->numXY);
}

__device__ unsigned cudaCalcIndex(const GridBins* const grid, unsigned* indices) {
    return indices[0] + indices[1]*grid->num[0] + indices[2]*(grid->numXY);
}

__device__ unsigned cudaCalcIndex(const GridBins* const grid, int* indices) {
    return indices[0] + indices[1]*grid->num[0] + indices[2]*(grid->numXY);
}


__device__ void cudaCalcIJK(const GridBins* const grid, unsigned index, uint3& indices ) {
    indices.z = index / grid->numXY;
    indices.y = (index - indices.z*grid->numXY ) / grid->num[0];
    indices.x = (index - indices.z*grid->numXY - indices.y * grid->num[0]);

}

__device__ float_t cudaGetVertex(const GridBins* const grid, unsigned dim, unsigned index ) {
	return grid->vertices[ grid->offset[dim] + index ];
}

__device__  void cudaGetCenterPointByIndices(const GridBins* const grid, uint3& indices,  float3_t& pos ){
	pos.x = ( cudaGetVertex(grid, 0, indices.x) + cudaGetVertex(grid, 0, indices.x+1)) / 2.0f ;
	pos.y = ( cudaGetVertex(grid, 1, indices.y) + cudaGetVertex(grid, 1, indices.y+1)) / 2.0f ;
	pos.z = ( cudaGetVertex(grid, 2, indices.z) + cudaGetVertex(grid, 2, indices.z+1)) / 2.0f ;
}


__device__ float_t cudaGetDistance( float3_t& pos1, float3_t& pos2) {
	float3_t deltaSq;
	deltaSq.x = (pos1.x - pos2.x)*(pos1.x - pos2.x);
	deltaSq.y = (pos1.y - pos2.y)*(pos1.y - pos2.y);
	deltaSq.z = (pos1.z - pos2.z)*(pos1.z - pos2.z);
	return sqrtf( deltaSq.x + deltaSq.y + deltaSq.z );
}

__device__ void cudaGetDistancesToAllCenters(const GridBins* const grid, float_t* distances, float3_t pos) {

	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	unsigned N = grid->num[0]*grid->num[1]*grid->num[2];
	unsigned index = 0;
	uint3 indices;
	for( unsigned i = 0; i < grid->num[0]; ++i ) {
		indices.x = i;
		for( unsigned j = 0; j < grid->num[1]; ++j ) {
			indices.y = j;
			for( unsigned k = 0; k < grid->num[2]; ++k ) {
				if( tid == index ) {
					indices.z = k;
					float3 pixelPoint;
					cudaGetCenterPointByIndices(grid, indices, pixelPoint);
					distances[index] = cudaGetDistance( pixelPoint, pos );
					tid += blockDim.x*gridDim.x;
				}
				if( tid >= N ) {
					return;
				}
				++index;
			}
		}
	}
}

__device__ void cudaGetDistancesToAllCenters2(const GridBins* const grid, float_t* distances, float3_t pos) {

	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	unsigned N = grid->numXY*grid->num[2];
	float3 pixelPoint;
	uint3 indices;

	while( tid < N ) {
		cudaCalcIJK(grid, tid, indices );
		cudaGetCenterPointByIndices(grid, indices, pixelPoint);
		distances[tid] = cudaGetDistance( pixelPoint, pos );
		tid += blockDim.x*gridDim.x;
	}
}


__global__ void kernelGetDistancesToAllCenters(void* ptrGrid,  void* ptrDistances, float_t x, float_t y, float_t z) {
	float3_t pos = make_float3(x,y,z);
	GridBins* grid = (GridBins*) ptrGrid;
	float_t* distances = (float_t*) ptrDistances;
	cudaGetDistancesToAllCenters2(grid, distances, pos);
}


__device__ float_t cudaMin(const GridBins* const grid, unsigned dim) {
	return grid->minMax[dim*2];
}

__device__ float_t cudaMax(const GridBins* const grid, unsigned dim) {
	return grid->minMax[dim*2+1];
}

__device__ unsigned cudaGetNumVertices(const GridBins* const grid, unsigned dim) {
	return grid->num[dim] + 1;
}

__device__ unsigned cudaGetNumBins(const GridBins* const grid, unsigned dim) {
	return grid->num[dim];
}

__device__ unsigned cudaGetNumBins(const GridBins* const grid, unsigned dim, unsigned index) {
	return grid->num[dim];
}


__device__ int cudaGetDimIndex(const GridBins* const grid, unsigned dim, float_t pos ) {
     // returns -1 for one neg side of mesh
     // and number of bins on the pos side of the mesh
     // need to call isIndexOutside(dim, grid, index) to check if the
     // index is in the mesh

	int dim_index;
	float_t min = cudaMin(grid, dim);

	if( pos <= min ) {
		dim_index = -1;
	} else if( pos >= cudaMax(grid, dim)  ) {
		dim_index = grid->num[dim];
	} else {
		dim_index = ( pos -  min ) / grid->delta[dim];
	}
	return dim_index;
}

__device__ bool cudaIsIndexOutside(const GridBins* const grid, unsigned dim, int i) {
	if( i < 0 ||  i >= cudaGetNumBins(grid, dim) ) return true;
	return false;
}

__device__ bool cudaIsOutside(const GridBins* const grid, const int3& indices ) {
	 if( cudaIsIndexOutside(grid, 0, indices.x) ) return true;
	 if( cudaIsIndexOutside(grid, 0, indices.y) ) return true;
	 if( cudaIsIndexOutside(grid, 0, indices.z) ) return true;
	 return false;
}

__device__ bool cudaIsOutside(const GridBins* const grid, uint1* indices ) {
	 if( cudaIsIndexOutside(grid, 0, indices[0].x) ) return true;
	 if( cudaIsIndexOutside(grid, 0, indices[1].x) ) return true;
	 if( cudaIsIndexOutside(grid, 0, indices[2].x) ) return true;
	 return false;
}

__device__ bool cudaIsOutside(const GridBins* const grid, int* indices ) {
	 if( cudaIsIndexOutside(grid, 0, indices[0]) ) return true;
	 if( cudaIsIndexOutside(grid, 0, indices[1]) ) return true;
	 if( cudaIsIndexOutside(grid, 0, indices[2]) ) return true;
	 return false;
}


__device__ unsigned cudaGetIndex(const GridBins* const grid, const float3_t& pos) {

	uint3 indices;

	indices.x = cudaGetDimIndex(grid, 0, pos.x);
	if( cudaIsIndexOutside(grid, 0, indices.x) ) { return UINT_MAX; }

	indices.y = cudaGetDimIndex(grid, 1, pos.y);
	if( cudaIsIndexOutside(grid, 1, indices.y) ) { return UINT_MAX; }

	indices.z = cudaGetDimIndex(grid, 2, pos.z);
	if( cudaIsIndexOutside(grid, 2, indices.z) ) { return UINT_MAX; }

    return cudaCalcIndex(grid, indices );
}

