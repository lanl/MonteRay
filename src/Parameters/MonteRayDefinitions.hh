/*
 * MonteRayDefinitions.hh
 *
 *  Created on: Mar 20, 2017
 *      Author: jsweezy  - jsweezy@lanl.gov
 */

#ifndef MONTERAYDEFINITIONS_HH_
#define MONTERAYDEFINITIONS_HH_

#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif

#include "MonteRayPreprocessorDefinitions.hh"

#define TALLY_DOUBLEPRECISION 1 // turn on (1) and off (0) for double precision tally array and compute
#define RAY_DOUBLEPRECISION 1 // turn on (1) and off (0) for double precision tally array and compute

namespace MonteRay{

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDAHOST_CALLABLE_MEMBER __host__
#define CUDADEVICE_CALLABLE_MEMBER __device__
#define CUDA_CALLABLE_KERNEL __global__
#else
#define CUDA_CALLABLE_MEMBER
#define CUDAHOST_CALLABLE_MEMBER
#define CUDA_CALLABLE_KERNEL
#define CUDADEVICE_CALLABLE_MEMBER
#endif

// typedefs
typedef float float_t;
typedef float gpuFloatType_t;

#if RAY_DOUBLEPRECISION < 1
typedef float gpuRayFloat_t;
#else
typedef double gpuRayFloat_t;
#endif


#if TALLY_DOUBLEPRECISION < 1
typedef float gpuTallyType_t;
#else
typedef double gpuTallyType_t;
#endif

typedef long long clock64_t;


#ifndef __CUDACC__

struct uint1 {
	unsigned int x;
};

struct uint3 {
	unsigned int x, y, z;
};

struct float1{
	float x;
};

struct float3{
	float3() : x(0.0), y(0.0), z(0.0) {}
	float3(float arg_x, float arg_y, float arg_z ) : x(arg_x), y(arg_y), z(arg_z) {}
	float x, y, z;
};

inline float3 make_float3(float x, float y, float z) {
	return float3(x,y,z);
}

struct int3{
    int x, y, z;
};

#endif

typedef float1 float1_t;
typedef float3 float3_t;

}

#endif /* MONTERAYDEFINITIONS_HH_ */
