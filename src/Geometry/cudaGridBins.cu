#include "cudaGridBins.hh"

#include <stdio.h>
#include <stdlib.h>

namespace MonteRay{

//CUDA_CALLABLE_MEMBER unsigned cudaCalcIndex(const GridBins* const grid, uint3& indices) {
//    return indices.x + indices.y*grid->num[0] + indices.z*(grid->numXY);
//}
//
//CUDA_CALLABLE_MEMBER unsigned cudaCalcIndex(const GridBins* const grid, unsigned* indices) {
//    return indices[0] + indices[1]*grid->num[0] + indices[2]*(grid->numXY);
//}
//
//CUDA_CALLABLE_MEMBER unsigned cudaCalcIndex(const GridBins* const grid, int* indices) {
//	const bool debug = false;
//
//	if( debug ) {
//		printf("cudaCalcIndex:: indices[0] = %d, indices[1] = %d, indices[2] = %d\n", indices[0], indices[1], indices[2]);
//		printf("cudaCalcIndex:: grid->num[0] = %d, grid->numXY = %d\n", grid->num[0], grid->numXY );
//	}
//
//    return indices[0] + indices[1]*grid->num[0] + indices[2]*(grid->numXY);
//}
//
//
//CUDA_CALLABLE_MEMBER void cudaCalcIJK(const GridBins* const grid, unsigned index, uint3& indices ) {
//    indices.z = index / grid->numXY;
//    indices.y = (index - indices.z*grid->numXY ) / grid->num[0];
//    indices.x = (index - indices.z*grid->numXY - indices.y * grid->num[0]);
//
//}
//
//CUDA_CALLABLE_MEMBER float_t cudaGetVertex(const GridBins* const grid, unsigned dim, unsigned index ) {
//	return grid->vertices[ grid->offset[dim] + index ];
//}
//
//CUDA_CALLABLE_MEMBER  void cudaGetCenterPointByIndices(const GridBins* const grid, uint3& indices, MonteRay::Vector3D<gpuRayFloat_t>& pos ){
//	pos[0] = ( cudaGetVertex(grid, 0, indices.x) + cudaGetVertex(grid, 0, indices.x+1)) / 2.0 ;
//	pos[1] = ( cudaGetVertex(grid, 1, indices.y) + cudaGetVertex(grid, 1, indices.y+1)) / 2.0 ;
//	pos[2] = ( cudaGetVertex(grid, 2, indices.z) + cudaGetVertex(grid, 2, indices.z+1)) / 2.0 ;
//}
//
//
//CUDA_CALLABLE_MEMBER float_t cudaGetDistance( MonteRay::Vector3D<gpuRayFloat_t>& pos1, MonteRay::Vector3D<gpuRayFloat_t>& pos2) {
//	MonteRay::Vector3D<gpuRayFloat_t> delta = pos1 - pos2;
//	return delta.magnitude();
//}


//CUDADEVICE_CALLABLE_MEMBER void cudaGetDistancesToAllCenters2(const GridBins* const grid, gpuRayFloat_t* distances, MonteRay::Vector3D<gpuRayFloat_t> pos) {
//	unsigned N = grid->numXY*grid->num[2];
//	MonteRay::Vector3D<gpuRayFloat_t> pixelPoint;
//	uint3 indices;
//
//#ifdef __CUDA__
//	int tid = threadIdx.x + blockIdx.x*blockDim.x;
//
//	while( tid < N ) {
//		cudaCalcIJK(grid, tid, indices );
//		cudaGetCenterPointByIndices(grid, indices, pixelPoint);
//		distances[tid] = cudaGetDistance( pixelPoint, pos );
//		tid += blockDim.x*gridDim.x;
//	}
//#else
//	int tid = 0;
//	while( tid < N ) {
//		cudaCalcIJK(grid, tid, indices );
//		cudaGetCenterPointByIndices(grid, indices, pixelPoint);
//		distances[tid] = cudaGetDistance( pixelPoint, pos );
//		tid += 1;
//	}
//#endif
//
//}


//CUDA_CALLABLE_KERNEL void kernelGetDistancesToAllCenters(void* ptrGrid,  void* ptrDistances, gpuRayFloat_t x, gpuRayFloat_t y, gpuRayFloat_t z) {
//	MonteRay::Vector3D<gpuRayFloat_t> pos(x,y,z);
//	GridBins* grid = (GridBins*) ptrGrid;
//	gpuRayFloat_t* distances = (gpuRayFloat_t*) ptrDistances;
//	cudaGetDistancesToAllCenters2(grid, distances, pos);
//}


//CUDA_CALLABLE_MEMBER float_t cudaMin(const GridBins* const grid, unsigned dim) {
//	return grid->minMax[dim*2];
//}
//
//CUDA_CALLABLE_MEMBER float_t cudaMax(const GridBins* const grid, unsigned dim) {
//	return grid->minMax[dim*2+1];
//}
//
//CUDA_CALLABLE_MEMBER unsigned cudaGetNumVertices(const GridBins* const grid, unsigned dim) {
//	return grid->num[dim] + 1;
//}
//
//CUDA_CALLABLE_MEMBER unsigned cudaGetNumBins(const GridBins* const grid, unsigned dim) {
//	return grid->num[dim];
//}
//
//CUDA_CALLABLE_MEMBER unsigned cudaGetNumBins(const GridBins* const grid, unsigned dim, unsigned index) {
//	return grid->num[dim];
//}
//
//template<typename T1,typename T2>
//CUDA_CALLABLE_MEMBER unsigned cudaGetIndexBinaryFloat(const T1* const values, unsigned count, T2 value ) {
//    // modified from http://en.cppreference.com/w/cpp/algorithm/upper_bound
//    unsigned it, step;
//    unsigned first = 0U;
//
//    while (count > 0U) {
//        it = first;
//        step = count / 2U;
//        it += step;
//        if(!(value < T2(values[it])) ) {
//            first = ++it;
//            count -= step + 1;
//        } else {
//            count = step;
//        }
//    }
//    if( first > 0U ) { --first; }
//    return first;
//}
//template CUDA_CALLABLE_MEMBER unsigned cudaGetIndexBinaryFloat(const float_t* const values, unsigned count, float_t value );
//template CUDA_CALLABLE_MEMBER unsigned cudaGetIndexBinaryFloat(const float_t* const values, unsigned count, double_t value );
//template CUDA_CALLABLE_MEMBER unsigned cudaGetIndexBinaryFloat(const double_t* const values, unsigned count, double_t value );
//
//
//CUDA_CALLABLE_MEMBER int cudaGetDimIndex(const GridBins* const grid, unsigned dim, gpuRayFloat_t pos ) {
//     // returns -1 for one neg side of mesh
//     // and number of bins on the pos side of the mesh
//     // need to call isIndexOutside(dim, grid, index) to check if the
//     // index is in the mesh
//	const bool debug = false;
//
//	int dim_index;
//	gpuRayFloat_t minimum = cudaMin(grid, dim);
//	unsigned numBins = grid->num[dim];
//
//	if( debug ) {
//		printf("cudaGetDimIndex:: Starting cudaGetDimIndex, dim = %d\n", dim);
//		printf("cudaGetDimIndex:: pos = %f\n", pos);
//		printf("cudaGetDimIndex:: num bins = %d\n", numBins);
//		printf("cudaGetDimIndex:: min = %f\n", minimum);
//		printf("cudaGetDimIndex:: max = %f\n", cudaMax(grid, dim));
//		if( grid->regular[dim] ) {
//			printf("cudaGetDimIndex:: grid is regular\n");
//		} else {
//			printf("cudaGetDimIndex:: grid is not regular\n");
//		}
//	}
//
//	if( pos <= minimum ) {
//		dim_index = -1;
//	} else if( pos >= cudaMax(grid, dim)  ) {
//		dim_index = numBins;
//	} else {
//		if( grid->regular[dim] ) {
//			dim_index = ( pos -  minimum ) / gpuRayFloat_t(grid->delta[dim]);
//		} else {
//			dim_index = cudaGetIndexBinaryFloat( grid->vertices + grid->offset[dim], numBins+1, pos  );
//		}
//	}
//
//	if( debug ) {
//		printf("cudaGetDimIndex:: dim_index = %d\n", dim_index);
//	}
//	return dim_index;
//}
//
//CUDA_CALLABLE_MEMBER bool cudaIsIndexOutside(const GridBins* const grid, unsigned dim, int i) {
//	const bool debug = false;
//	if( debug ) printf("Debug: cudaGridBins.cc::cudaIsIndexOutside -- dim=%u  i=%d  number of bins=%u\n", dim, i, cudaGetNumBins(grid, dim) );
//	if( i < 0 ||  i >= cudaGetNumBins(grid, dim) ) return true;
//	return false;
//}
//
//CUDA_CALLABLE_MEMBER bool cudaIsOutside(const GridBins* const grid, const int3& indices ) {
//	 if( cudaIsIndexOutside(grid, 0, indices.x) ) return true;
//	 if( cudaIsIndexOutside(grid, 1, indices.y) ) return true;
//	 if( cudaIsIndexOutside(grid, 2, indices.z) ) return true;
//	 return false;
//}
//
//CUDA_CALLABLE_MEMBER bool cudaIsOutside(const GridBins* const grid, uint1* indices ) {
//	 if( cudaIsIndexOutside(grid, 0, indices[0].x) ) return true;
//	 if( cudaIsIndexOutside(grid, 1, indices[1].x) ) return true;
//	 if( cudaIsIndexOutside(grid, 2, indices[2].x) ) return true;
//	 return false;
//}
//
//CUDA_CALLABLE_MEMBER bool cudaIsOutside(const GridBins* const grid, int* indices ) {
//	 if( cudaIsIndexOutside(grid, 0, indices[0]) ) return true;
//	 if( cudaIsIndexOutside(grid, 1, indices[1]) ) return true;
//	 if( cudaIsIndexOutside(grid, 2, indices[2]) ) return true;
//	 return false;
//}
//
//
//CUDA_CALLABLE_MEMBER unsigned cudaGetIndex(const GridBins* const grid, const MonteRay::Vector3D<gpuRayFloat_t>& pos) {
//
//	uint3 indices;
//
//	indices.x = cudaGetDimIndex(grid, 0, pos[0]);
//	if( cudaIsIndexOutside(grid, 0, indices.x) ) { return UINT_MAX; }
//
//	indices.y = cudaGetDimIndex(grid, 1, pos[1]);
//	if( cudaIsIndexOutside(grid, 1, indices.y) ) { return UINT_MAX; }
//
//	indices.z = cudaGetDimIndex(grid, 2, pos[2]);
//	if( cudaIsIndexOutside(grid, 2, indices.z) ) { return UINT_MAX; }
//
//    return cudaCalcIndex(grid, indices );
//}

}
