#ifndef MONTERAYTYPES_HH_
#define MONTERAYTYPES_HH_

#include "Containers.hh"

#define TALLY_DOUBLEPRECISION 1 // turn on (1) and off (0) for double precision tally array and compute
#define RAY_DOUBLEPRECISION 0 // turn on (1) and off (0) for double precision tally array and compute

#define MAXNUMMATERIALS 70
#define MAXNUMVERTICES 1000
#define MAXNUMRAYCELLS 2000 // 2*MAXNUMVERTICES

namespace MonteRay{

// typedefs
typedef float float_t;
typedef float gpuFloatType_t;
typedef short int ParticleType_t;

template <typename T>
using Vector = SimpleVector<T>;
template <typename T>
using View = SimpleView<T>;

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

#define MONTERAY_MAX_THREADS 100000U // 3125 x 32 - should be a multiple of 32
#define MONTERAY_MAX_THREADS_PER_BLOCK 512U

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDAHOST_CALLABLE_MEMBER __host__
#define CUDADEVICE_CALLABLE_MEMBER __device__
#define CUDA_CALLABLE_KERNEL __global__ void __launch_bounds__ ( MONTERAY_MAX_THREADS_PER_BLOCK )
#else
#define CUDA_CALLABLE_MEMBER
#define CUDAHOST_CALLABLE_MEMBER
#define CUDA_CALLABLE_KERNEL void
#define CUDADEVICE_CALLABLE_MEMBER
#endif


}

#endif /* MONTERAYTYPES_HH_ */
