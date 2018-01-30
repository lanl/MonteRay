#include "cudaGridBins.h"

#include <stdio.h>
#include <stdlib.h>

namespace MonteRay{

CUDA_CALLABLE_MEMBER unsigned cudaCalcIndex(const GridBins* const grid, uint3& indices) {
    return indices.x + indices.y*grid->num[0] + indices.z*(grid->numXY);
}

CUDA_CALLABLE_MEMBER unsigned cudaCalcIndex(const GridBins* const grid, unsigned* indices) {
    return indices[0] + indices[1]*grid->num[0] + indices[2]*(grid->numXY);
}

CUDA_CALLABLE_MEMBER unsigned cudaCalcIndex(const GridBins* const grid, int* indices) {
	const bool debug = false;

	if( debug ) {
		printf("cudaCalcIndex:: indices[0] = %d, indices[1] = %d, indices[2] = %d\n", indices[0], indices[1], indices[2]);
		printf("cudaCalcIndex:: grid->num[0] = %d, grid->numXY = %d\n", grid->num[0], grid->numXY );
	}

    return indices[0] + indices[1]*grid->num[0] + indices[2]*(grid->numXY);
}


CUDA_CALLABLE_MEMBER void cudaCalcIJK(const GridBins* const grid, unsigned index, uint3& indices ) {
    indices.z = index / grid->numXY;
    indices.y = (index - indices.z*grid->numXY ) / grid->num[0];
    indices.x = (index - indices.z*grid->numXY - indices.y * grid->num[0]);

}

CUDA_CALLABLE_MEMBER float_t cudaGetVertex(const GridBins* const grid, unsigned dim, unsigned index ) {
	return grid->vertices[ grid->offset[dim] + index ];
}

CUDA_CALLABLE_MEMBER  void cudaGetCenterPointByIndices(const GridBins* const grid, uint3& indices,  float3_t& pos ){
	pos.x = ( cudaGetVertex(grid, 0, indices.x) + cudaGetVertex(grid, 0, indices.x+1)) / 2.0f ;
	pos.y = ( cudaGetVertex(grid, 1, indices.y) + cudaGetVertex(grid, 1, indices.y+1)) / 2.0f ;
	pos.z = ( cudaGetVertex(grid, 2, indices.z) + cudaGetVertex(grid, 2, indices.z+1)) / 2.0f ;
}


CUDA_CALLABLE_MEMBER float_t cudaGetDistance( float3_t& pos1, float3_t& pos2) {
	float3_t deltaSq;
	deltaSq.x = (pos1.x - pos2.x)*(pos1.x - pos2.x);
	deltaSq.y = (pos1.y - pos2.y)*(pos1.y - pos2.y);
	deltaSq.z = (pos1.z - pos2.z)*(pos1.z - pos2.z);
	return sqrtf( deltaSq.x + deltaSq.y + deltaSq.z );
}


CUDADEVICE_CALLABLE_MEMBER void cudaGetDistancesToAllCenters2(const GridBins* const grid, float_t* distances, float3_t pos) {
	unsigned N = grid->numXY*grid->num[2];
	float3 pixelPoint;
	uint3 indices;

#ifdef __CUDA__
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	while( tid < N ) {
		cudaCalcIJK(grid, tid, indices );
		cudaGetCenterPointByIndices(grid, indices, pixelPoint);
		distances[tid] = cudaGetDistance( pixelPoint, pos );
		tid += blockDim.x*gridDim.x;
	}
#else
	int tid = 0;
	while( tid < N ) {
		cudaCalcIJK(grid, tid, indices );
		cudaGetCenterPointByIndices(grid, indices, pixelPoint);
		distances[tid] = cudaGetDistance( pixelPoint, pos );
		tid += 1;
	}
#endif

}


CUDA_CALLABLE_KERNEL void kernelGetDistancesToAllCenters(void* ptrGrid,  void* ptrDistances, float_t x, float_t y, float_t z) {
	float3_t pos = make_float3(x,y,z);
	GridBins* grid = (GridBins*) ptrGrid;
	float_t* distances = (float_t*) ptrDistances;
	cudaGetDistancesToAllCenters2(grid, distances, pos);
}


CUDA_CALLABLE_MEMBER float_t cudaMin(const GridBins* const grid, unsigned dim) {
	return grid->minMax[dim*2];
}

CUDA_CALLABLE_MEMBER float_t cudaMax(const GridBins* const grid, unsigned dim) {
	return grid->minMax[dim*2+1];
}

CUDA_CALLABLE_MEMBER unsigned cudaGetNumVertices(const GridBins* const grid, unsigned dim) {
	return grid->num[dim] + 1;
}

CUDA_CALLABLE_MEMBER unsigned cudaGetNumBins(const GridBins* const grid, unsigned dim) {
	return grid->num[dim];
}

CUDA_CALLABLE_MEMBER unsigned cudaGetNumBins(const GridBins* const grid, unsigned dim, unsigned index) {
	return grid->num[dim];
}

CUDA_CALLABLE_MEMBER unsigned cudaGetIndexBinaryFloat(const float_t* const values, unsigned count, float_t value ) {
    // modified from http://en.cppreference.com/w/cpp/algorithm/upper_bound
    unsigned it, step;
    unsigned first = 0U;

    while (count > 0U) {
        it = first;
        step = count / 2U;
        it += step;
        if(!(value < values[it])) {
            first = ++it;
            count -= step + 1;
        } else {
            count = step;
        }
    }
    if( first > 0U ) { --first; }
    return first;
}

CUDA_CALLABLE_MEMBER int cudaGetDimIndex(const GridBins* const grid, unsigned dim, float_t pos ) {
     // returns -1 for one neg side of mesh
     // and number of bins on the pos side of the mesh
     // need to call isIndexOutside(dim, grid, index) to check if the
     // index is in the mesh
	const bool debug = false;

	int dim_index;
	float_t minimum = cudaMin(grid, dim);
	unsigned numBins = grid->num[dim];

	if( debug ) {
		printf("cudaGetDimIndex:: Starting cudaGetDimIndex, dim = %d\n", dim);
		printf("cudaGetDimIndex:: pos = %f\n", pos);
		printf("cudaGetDimIndex:: num bins = %d\n", numBins);
		printf("cudaGetDimIndex:: min = %f\n", minimum);
		printf("cudaGetDimIndex:: max = %f\n", cudaMax(grid, dim));
		if( grid->isRegular[dim] ) {
			printf("cudaGetDimIndex:: grid is regular\n");
		} else {
			printf("cudaGetDimIndex:: grid is not regular\n");
		}
	}

	if( pos <= minimum ) {
		dim_index = -1;
	} else if( pos >= cudaMax(grid, dim)  ) {
		dim_index = numBins;
	} else {
		if( grid->isRegular[dim] ) {
			dim_index = ( pos -  minimum ) / grid->delta[dim];
		} else {
			dim_index = cudaGetIndexBinaryFloat( grid->vertices + grid->offset[dim], numBins+1, pos  );
		}
	}

	if( debug ) {
		printf("cudaGetDimIndex:: dim_index = %d\n", dim_index);
	}
	return dim_index;
}

CUDA_CALLABLE_MEMBER bool cudaIsIndexOutside(const GridBins* const grid, unsigned dim, int i) {
	const bool debug = false;
	if( debug ) printf("Debug: cudaGridBins.cc::cudaIsIndexOutside -- dim=%u  i=%d  number of bins=%u\n", dim, i, cudaGetNumBins(grid, dim) );
	if( i < 0 ||  i >= cudaGetNumBins(grid, dim) ) return true;
	return false;
}

CUDA_CALLABLE_MEMBER bool cudaIsOutside(const GridBins* const grid, const int3& indices ) {
	 if( cudaIsIndexOutside(grid, 0, indices.x) ) return true;
	 if( cudaIsIndexOutside(grid, 1, indices.y) ) return true;
	 if( cudaIsIndexOutside(grid, 2, indices.z) ) return true;
	 return false;
}

CUDA_CALLABLE_MEMBER bool cudaIsOutside(const GridBins* const grid, uint1* indices ) {
	 if( cudaIsIndexOutside(grid, 0, indices[0].x) ) return true;
	 if( cudaIsIndexOutside(grid, 1, indices[1].x) ) return true;
	 if( cudaIsIndexOutside(grid, 2, indices[2].x) ) return true;
	 return false;
}

CUDA_CALLABLE_MEMBER bool cudaIsOutside(const GridBins* const grid, int* indices ) {
	 if( cudaIsIndexOutside(grid, 0, indices[0]) ) return true;
	 if( cudaIsIndexOutside(grid, 1, indices[1]) ) return true;
	 if( cudaIsIndexOutside(grid, 2, indices[2]) ) return true;
	 return false;
}


CUDA_CALLABLE_MEMBER unsigned cudaGetIndex(const GridBins* const grid, const float3_t& pos) {

	uint3 indices;

	indices.x = cudaGetDimIndex(grid, 0, pos.x);
	if( cudaIsIndexOutside(grid, 0, indices.x) ) { return UINT_MAX; }

	indices.y = cudaGetDimIndex(grid, 1, pos.y);
	if( cudaIsIndexOutside(grid, 1, indices.y) ) { return UINT_MAX; }

	indices.z = cudaGetDimIndex(grid, 2, pos.z);
	if( cudaIsIndexOutside(grid, 2, indices.z) ) { return UINT_MAX; }

    return cudaCalcIndex(grid, indices );
}

}
