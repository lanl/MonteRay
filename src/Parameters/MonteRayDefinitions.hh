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

#ifdef __CUDACC__
typedef float1 float1_t;
typedef float3 float3_t;
#endif

#if TALLY_DOUBLEPRECISION < 1
typedef float gpuTallyType_t;
#else
typedef double gpuTallyType_t;
#endif

typedef long long clock64_t;

}

#endif /* MONTERAYDEFINITIONS_HH_ */
