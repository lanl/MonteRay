#include "gpuRayTrace.hh"

#include <iostream>
#include <math.h>

#include "MonteRayDefinitions.hh"
#include "MonteRayConstants.hh"
#include "GridBins.hh"

#ifdef __CUDACC__
#include "math_constants.h"
#endif

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
//		int index )
//{
//	const bool debug = false;
//
//#ifdef __CUDACC__
//	constexpr gpuRayFloat_t minFloat = CUDART_TWO_TO_M126_F;
//#else
//	constexpr gpuRayFloat_t minFloat = 1.175494351e-38f;
//#endif
//
//	unsigned nDistances = 0;
//
//	if( debug ) {
//		printf("cudaCalcCrossings:: Starting cudaCalcCrossings ******************\n");
//		printf("cudaCalcCrossings:: pos=%f\n", pos);
//		printf("cudaCalcCrossings:: dir=%f\n", dir);
//		printf("cudaCalcCrossings:: index=%d\n", index);
//	}
//
//	//if( abs(1/dir) >= CUDART_NORM_HUGE_F )
//    if( abs(dir) <= minFloat )  // CUDART_TWO_TO_M126_F    1.175494351e-38f
//    {
//    	return nDistances;
//    }
//
//    int start_index = index;
//    int cell_index = start_index;
//
//    if( start_index < 0 ) {
//        if( dir < 0.0 ) {
//            return nDistances;
//        }
//    }
//
//    int nBins = nVertices - 1;
//
//	if( debug ) {
//		printf("cudaCalcCrossings:: nBins=%d\n", nBins);
//	}
//
//    if( start_index >= nBins ) {
//        if( dir > 0.0 ) {
//        	return nDistances;
//        }
//    }
//
//    unsigned offset = 0;
//    if( dir > 0.0 ) {
//    	offset = 1;
//    }
//
//    int end_index = offset*(nBins-1);;
//
//    int dirIncrement = copysign( 1.0f, dir );
//
//    unsigned num_indices = abs(end_index - start_index ) + 1;
//
//    int current_index = start_index;
//
//    // Calculate boundary crossing distances
//    gpuRayFloat_t invDir = 1/dir;
//    bool rayTerminated = false;
//    for( int i = 0; i < num_indices ; ++i ) {
//
//    	gpuRayFloat_t minDistance = ( gpuRayFloat_t(vertices[current_index + offset]) - pos) * invDir;
//
//        if( minDistance >= distance ) {
//        	cells[nDistances] = cell_index;
//        	distances[nDistances] = distance;
//
//        	if( debug ) {
//        		printf("cudaCalcCrossings:: crossing num=%d, index=%d, distance=%f\n",
//        				nDistances,
//        				cells[nDistances],
//        				distances[nDistances]);
//        	}
//
//        	++nDistances;
//            rayTerminated = true;
//            break;
//        }
//
//        cells[nDistances] = cell_index;
//        distances[nDistances] = minDistance;
//
//    	if( debug ) {
//    		printf("cudaCalcCrossings:: crossing num=%d, index=%d, distance=%f\n",
//    				nDistances,
//    				cells[nDistances],
//    				distances[nDistances]);
//    	}
//
//        ++nDistances;
//
//        current_index += dirIncrement;
//        cell_index = current_index;
//    }
//
//    if( !rayTerminated ) {
//        // finish with distance into area outside
//    	cells[nDistances] = cell_index;
//    	distances[nDistances] = distance;
//
//    	if( debug ) {
//    		printf("cudaCalcCrossings:: crossing num=%d, index=%d, distance=%f\n",
//    				nDistances,
//    				cells[nDistances],
//    				distances[nDistances]);
//    	}
//
//    	++nDistances;
//        rayTerminated = true;
//    }
//
//    return nDistances;
//}
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
//		bool outsideDistances )
//{
//    // Order the distance crossings to provide a rayTrace
//
//	const bool debug = false;
//
//#ifdef __CUDACC__
//	constexpr gpuRayFloat_t maxFloat = CUDART_NORM_HUGE_F;
//#else
//	constexpr gpuRayFloat_t maxFloat = 3.402823466e38f;
//#endif
//
//
//	if( debug ) {
//		printf("cudaRayTrace:: Starting cudaOrderCrossings %%%%%%%%%%%%%%%%%%%%%%\n");
//		printf("cudaRayTrace:: cudaindices.x = %d, cudaindices.y = %d, cudaindices.z = %d,\n", cudaindices.x, cudaindices.y, cudaindices.z);
//	}
//
//
//	unsigned end[3] = { numCrossings.x, numCrossings.y, numCrossings.z}; //    last location in the distance[i] vector
//    int indices[3];
//    indices[0] = cudaindices.x; indices[1] = cudaindices.y; indices[2] = cudaindices.z;
//
//    int maxNumCrossings = numCrossings.x + numCrossings.y + numCrossings.z;
//    if( debug ) printf("cudaRayTrace::cudaOrderCrossings  maxNumCrossings = %d\n", maxNumCrossings);
//    gpuRayFloat_t minDistances[3];
//
//    bool outside;
//    gpuRayFloat_t priorDistance = 0.0;
//    unsigned start[3] = {0, 0, 0}; // current location in the distance[i] vector
//
//    unsigned numRayCrossings = 0;
//    for( unsigned i=0; i<maxNumCrossings; ++i){
//
//    	unsigned minDim;
//    	gpuRayFloat_t minimumDistance = maxFloat;
//        for( unsigned j = 0; j<3; ++j) {
//            if( start[j] < end[j] ) {
//            	minDistances[j] = *((crossingDistances+j*num)+start[j]);
//            	if( minDistances[j] < minimumDistance ) {
//            		minimumDistance = minDistances[j];
//            		minDim = j;
//            	}
//            } else {
//                minDistances[j] = maxFloat;
//            }
//        }
//        if( debug ) printf("cudaRayTrace::cudaOrderCrossings  crossing # %d, min dimension = %d, distance = %f\n", i, minDim, minimumDistance);
//
//        indices[minDim] =  *((cells+minDim*num) + start[minDim]);
//        if( debug ) printf("cudaRayTrace::cudaOrderCrossings  current indices: i = %d, j = %d, k = %d\n", indices[0], indices[1], indices[2]);
//
//        // test for outside of the grid
//        outside = cudaIsOutside(grid, indices );
//
//        if( debug ) {
//        	if( outside ) {
//        		printf("cudaRayTrace::cudaOrderCrossings  -- ray is outside the mesh\n");
//        	}
//        }
//
//        gpuRayFloat_t currentDistance = minimumDistance;
//
//        if( !outside || outsideDistances ) {
//        	gpuRayFloat_t deltaDistance = currentDistance - priorDistance;
//
//            if( deltaDistance > 0.0  ) {
//                unsigned global_index;
//                if( !outside ) {
//                    global_index = cudaCalcIndex(grid, indices );
//                } else {
//                    global_index = UINT_MAX;
//                }
//                global_indices[numRayCrossings] = global_index;
//                distances[numRayCrossings] = deltaDistance;
//
//            	if( debug ) {
//            		printf("cudaRayTrace:: crossing num=%d, index=%d, distance=%f\n", numRayCrossings,
//            				                                                          global_indices[numRayCrossings],
//            				                                                          distances[numRayCrossings]);
//            	}
//
//                ++numRayCrossings;
//            }
//        }
//
//        if( currentDistance >= distance ) {
//            break;
//        }
//
//        indices[minDim] = *((cells+minDim*num) + start[minDim]+1);
//
//        if( ! outside ) {
//            if( cudaIsIndexOutside(grid, minDim, indices[minDim] ) ) {
//                // ray has moved outside of grid
//                break;
//            }
//        }
//
//        ++start[minDim];
//        priorDistance = currentDistance;
//    }
//
//    return numRayCrossings;
//}
//
//CUDA_CALLABLE_MEMBER unsigned cudaRayTrace(const GridBins* const grid,
//		                         int* global_indices,
//		                         gpuRayFloat_t* distances,
//		                         const float3_t& pos,
//		                         const float3_t& dir,
//		                         gpuRayFloat_t distance,
//		                         bool outsideDistances)
//{
//	const bool debug = false;
//
//	if( debug ) {
//		printf("cudaRayTrace:: Starting cudaRayTrace ******************\n");
//	}
//
//	int3 current_indices;
//
//    int cells[3][MAXNUMVERTICES];
//    gpuRayFloat_t crossingDistances[3][MAXNUMVERTICES];
//    uint3 numCrossings;
//
//    current_indices.x = cudaGetDimIndex(grid, 0, pos.x );
//	numCrossings.x = cudaCalcCrossings( grid->vertices + grid->offset[0], grid->num[0]+1, cells[0], crossingDistances[0], pos.x, dir.x, distance, current_indices.x);
//
//	if( debug ) {
//		printf("cudaRayTrace:: current_indices.x =%d\n", current_indices.x );
//		printf("cudaRayTrace:: numCrossings.x =%d\n", numCrossings.x );
//	}
//
//	if( cudaIsIndexOutside(grid, 0, current_indices.x ) && numCrossings.x == 0  ) {return 0U;}
//
//    current_indices.y = cudaGetDimIndex(grid, 1, pos.y );
//	numCrossings.y = cudaCalcCrossings( grid->vertices + grid->offset[1], grid->num[1]+1, cells[1], crossingDistances[1], pos.y, dir.y, distance, current_indices.y);
//
//	if( debug ) {
//		printf("cudaRayTrace:: current_indices.y =%d\n", current_indices.y );
//		printf("cudaRayTrace:: numCrossings.y =%d\n", numCrossings.y );
//	}
//
//	if( cudaIsIndexOutside(grid, 1, current_indices.y ) && numCrossings.y == 0  ) {return 0U;}
//
//	current_indices.z = cudaGetDimIndex(grid, 2, pos.z );
//	numCrossings.z = cudaCalcCrossings( grid->vertices + grid->offset[2], grid->num[2]+1, cells[2], crossingDistances[2], pos.z, dir.z, distance, current_indices.z);
//
//	if( debug ) {
//		printf("cudaRayTrace:: current_indices.z =%d\n", current_indices.z );
//		printf("cudaRayTrace:: numCrossings.z =%d\n", numCrossings.z );
//	}
//
//	if( cudaIsIndexOutside(grid, 2, current_indices.z ) && numCrossings.z == 0  ) {return 0U;}
//
//    unsigned numRayCrossings = cudaOrderCrossings(grid, global_indices, distances, MAXNUMVERTICES, cells[0], crossingDistances[0], numCrossings, current_indices, distance, outsideDistances);
//
//	if( debug ) {
//		printf("cudaRayTrace:: numRayCrossings=%d\n", numRayCrossings );
//	}
//
//    return numRayCrossings;
//}
//
//CUDA_CALLABLE_MEMBER unsigned cudaRayTrace(const GridBins* const grid,
//		                         int* global_indices,
//		                         gpuRayFloat_t* distances,
//		                         const MonteRay::Vector3D<gpuRayFloat_t>& pos,
//		                         const MonteRay::Vector3D<gpuRayFloat_t>& dir,
//		                         gpuRayFloat_t distance,
//		                         bool outsideDistances)
//{
//	const bool debug = false;
//
//	if( debug ) {
//		printf("cudaRayTrace:: Starting cudaRayTrace ******************\n");
//	}
//
//	int3 current_indices;
//
//    int cells[3][MAXNUMVERTICES];
//    gpuRayFloat_t crossingDistances[3][MAXNUMVERTICES];
//    uint3 numCrossings;
//
//    current_indices.x = cudaGetDimIndex(grid, 0, pos[0] );
//	numCrossings.x = cudaCalcCrossings( grid->vertices + grid->offset[0], grid->num[0]+1, cells[0], crossingDistances[0], pos[0], dir[0], distance, current_indices.x);
//
//	if( debug ) {
//		printf("cudaRayTrace:: current_indices.x =%d\n", current_indices.x );
//		printf("cudaRayTrace:: numCrossings.x =%d\n", numCrossings.x );
//	}
//
//	if( cudaIsIndexOutside(grid, 0, current_indices.x ) && numCrossings.x == 0  ) {return 0U;}
//
//    current_indices.y = cudaGetDimIndex(grid, 1, pos[1] );
//	numCrossings.y = cudaCalcCrossings( grid->vertices + grid->offset[1], grid->num[1]+1, cells[1], crossingDistances[1], pos[1], dir[1], distance, current_indices.y);
//
//	if( debug ) {
//		printf("cudaRayTrace:: current_indices.y =%d\n", current_indices.y );
//		printf("cudaRayTrace:: numCrossings.y =%d\n", numCrossings.y );
//	}
//
//	if( cudaIsIndexOutside(grid, 1, current_indices.y ) && numCrossings.y == 0  ) {return 0U;}
//
//	current_indices.z = cudaGetDimIndex(grid, 2, pos[2] );
//	numCrossings.z = cudaCalcCrossings( grid->vertices + grid->offset[2], grid->num[2]+1, cells[2], crossingDistances[2], pos[2], dir[2], distance, current_indices.z);
//
//	if( debug ) {
//		printf("cudaRayTrace:: current_indices.z =%d\n", current_indices.z );
//		printf("cudaRayTrace:: numCrossings.z =%d\n", numCrossings.z );
//	}
//
//	if( cudaIsIndexOutside(grid, 2, current_indices.z ) && numCrossings.z == 0  ) {return 0U;}
//
//    unsigned numRayCrossings = cudaOrderCrossings(grid, global_indices, distances, MAXNUMVERTICES, cells[0], crossingDistances[0], numCrossings, current_indices, distance, outsideDistances);
//
//	if( debug ) {
//		printf("cudaRayTrace:: numRayCrossings=%d\n", numRayCrossings );
//	}
//
//    return numRayCrossings;
//}

CUDA_CALLABLE_KERNEL
void
kernelCudaRayTrace(void* ptrNumCrossings,
		GridBins* ptrGrid,
		int* ptrCells,
		gpuRayFloat_t* ptrDistances,
		gpuFloatType_t x, gpuFloatType_t y, gpuFloatType_t z,
		gpuFloatType_t u, gpuFloatType_t v, gpuFloatType_t w,
		gpuFloatType_t distance,
		bool outsideDistances) {

	const bool debug = false;

	if( debug ) {
		printf("kernelCudaRayTrace:: Starting kernelCudaRayTrace ******************\n");
	}

	unsigned* numCrossings = (unsigned*) ptrNumCrossings;

//	float3_t pos = make_float3( x, y, z);
//	float3_t dir = make_float3( u, v, w);
	Position_t pos( x, y, z );
	Direction_t dir( u, v, w );

	numCrossings[0] = ptrGrid->rayTrace( ptrCells, ptrDistances, pos, dir, distance, outsideDistances);

	if( debug ) {
		printf("kernelCudaRayTrace:: numCrossings=%d\n",numCrossings[0]);
	}
}


//CUDA_CALLABLE_KERNEL
//void
//kernelCudaRayTraceToAllCenters(
//		void* ptrGrid,
//		void* ptrDistances,
//		float_t x, float_t y, float_t z)
//{
//	const bool debug = false;
//
//	if( debug ) {
//		printf("kernelCudaRayTraceToAllCenters:: Starting kernelCudaRayTraceToAllCenters ******************\n");
//	}
//
//	GridBins* grid = (GridBins*) ptrGrid;
//	gpuRayFloat_t* distances = (gpuRayFloat_t*) ptrDistances;
//
//#ifdef __CUDACC__
//	int tid = threadIdx.x + blockIdx.x*blockDim.x;
//#else
//	int tid = 0;
//#endif
//
//	if( debug ) {
//		printf("kernelCudaRayTraceToAllCenters:: tid=%d\n", tid );
//#ifdef __CUDACC__
//		printf("kernelCudaRayTraceToAllCenters:: threadIdx.x=%d\n", threadIdx.x );
//		printf("kernelCudaRayTraceToAllCenters::  blockIdx.x=%d\n", blockIdx.x );
//		printf("kernelCudaRayTraceToAllCenters::  blockDim.x=%d\n", blockDim.x );
//#endif
//	}
//
//	int N = grid->numXY*grid->num[2];
//
//	if( debug ) {
//		printf("kernelCudaRayTraceToAllCenters:: Num Cells =%d\n", N );
//	}
//
//	uint3 indices;
//
//	//float3_t pos1 = make_float3( x, y, z);
//	MonteRay::Vector3D<gpuRayFloat_t> pos1( x, y, z );
//
//	//float3_t pos2;
//	MonteRay::Vector3D<gpuRayFloat_t> pos2;
//
//	//float3 dir;
//	MonteRay::Vector3D<gpuRayFloat_t> dir;
//
//	int cells[2*MAXNUMVERTICES];
//	gpuRayFloat_t crossingDistances[2*MAXNUMVERTICES];
//
//	while( tid < N ) {
//
//		if( debug ) {
//			printf("kernelCudaRayTraceToAllCenters:: tid=%d\n", tid );
//			printf("------------------------------------------" );
//		}
//
//		cudaCalcIJK(grid, tid, indices );
//		cudaGetCenterPointByIndices(grid, indices, pos2);
//
//	    dir = pos2 - pos1;
//	    gpuRayFloat_t length = dir.magnitude();
//
//		if( debug ) {
//			printf("kernelCudaRayTraceToAllCenters:: length=%f\n", length );
//		}
//
//		dir[0] = length / dir[0];
//		dir[1] = length / dir[1];
//		dir[2] = length / dir[2];
//
//
//		if( length > 0.0 ) {
////			dir.x /= length;
////			dir.y /= length;
////			dir.z /= length;
//
//			if( debug ) {
//				printf("kernelCudaRayTraceToAllCenters:: u=%f v=%f w=%f\n",dir[0],dir[1],dir[2]);
//			}
//
//			unsigned numCrossings;
//			numCrossings = cudaRayTrace( grid, cells, crossingDistances, pos1, dir, length, false);
//
//			if( debug ) {
//				printf("kernelCudaRayTraceToAllCenters:: numCrossings=%d\n",numCrossings );
//			}
////
////			if( debug ) {
////				printf("kernelCudaRayTraceToAllCenters:: Exiting\n" );
////			}
////			return;
//
//			gpuRayFloat_t length2 = 0.0f;
//			for( unsigned i=0; i < numCrossings; ++i){
//				length2 += crossingDistances[i];
//				//length2 += cells[i]*crossingDistances[i];
//			}
//			distances[tid] = length2;
//
//		} else {
//			distances[tid] = 0.0f;
//		}
//
//		if( debug ) {
//			printf("kernelCudaRayTraceToAllCenters:: distance=%f\n", distances[tid] );
//			printf("------------------------------------------" );
//		}
//
//#ifdef __CUDACC__
//		tid += blockDim.x*gridDim.x;
//#else
//		++tid;
//#endif
//		//if( tid >= 10 ) return;
//
//	}
//}

}
