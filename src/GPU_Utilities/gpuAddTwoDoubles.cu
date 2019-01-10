#include "gpuAddTwoDoubles.hh"
#include "GPUAtomicAdd.hh"
#include "GPUSync.hh"
#include "MonteRayDefinitions.hh"
#include "MonteRayMemory.hh"

#include <iostream>
#include <stdio.h>

namespace MonteRay {

CUDA_CALLABLE_KERNEL  add_single(unsigned N, float *a, float *b, float *c ) {

#ifdef __CUDA_ARCH__
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if( bid < N ) {
        if( tid == 0 ) {
            gpu_atomicAdd_single( &c[bid], a[bid] );
        } else if ( tid == 1 ) {
            gpu_atomicAdd_single( &c[bid], b[bid] );
        }
    }
#else
    for( auto bid = 0; bid < N; ++bid ) {
        for( auto tid = 0; tid < 2; ++tid ) {
            if( tid == 0 ) {
                gpu_atomicAdd_single( &c[bid], a[bid] );
            } else if ( tid == 1 ) {
                gpu_atomicAdd_single( &c[bid], b[bid] );
            }
        }
    }
#endif
}

CUDA_CALLABLE_KERNEL  add_double(unsigned N, double *a, double *b, double *c ) {
    //	printf("Debug: GPU_Utilities/unit_test/gpuAddTwoDoubles.cc::add_double **************\n");
#ifdef __CUDA_ARCH__
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if( bid < N ) {
        if( tid == 0 ) {
            gpu_atomicAdd_double( &c[bid], a[bid] );
        } else if ( tid == 1 ) {
            gpu_atomicAdd_double( &c[bid], b[bid] );
        }
    }
#else
    for( auto bid = 0; bid < N; ++bid ) {
        for( auto tid = 0; tid < 2; ++tid ) {
            if( tid == 0 ) {
                gpu_atomicAdd_double( &c[bid], a[bid] );
            } else if ( tid == 1 ) {
                gpu_atomicAdd_double( &c[bid], b[bid] );
            }
        }
    }
#endif
}

/// Adds two doubles
double gpuAddTwoDoubles( double A, double B) {
    typedef double value_t;

    // Adds two doubles, but uses array notation
    value_t C;

    unsigned N=1;

    value_t a_host[N];
    value_t b_host[N];
    value_t c_host[N];

    unsigned allocSize = N*sizeof(value_t);

    a_host[0] = A;
    b_host[0] = B;
    c_host[0] = 0.0;

#ifdef __CUDACC__
    //	std::cout <<"Debug: GPU_Utilities/unit_test/gpuAddTwoDoubles.cc::gpuAddTwoDoubles - CUDA defined.\n";

    value_t* pA_device;
    value_t* pB_device;
    value_t* pC_device;

    GPUSync sync;

    pA_device = (value_t*) MONTERAYDEVICEALLOC( allocSize, std::string("gpuAddTwoDoubles::pA_device") );
    pB_device = (value_t*) MONTERAYDEVICEALLOC( allocSize, std::string("gpuAddTwoDoubles::pB_device") );
    pC_device = (value_t*) MONTERAYDEVICEALLOC( allocSize, std::string("gpuAddTwoDoubles::pC_device") );

    cudaMemcpy( pA_device, a_host, allocSize, cudaMemcpyHostToDevice);
    cudaMemcpy( pB_device, b_host, allocSize, cudaMemcpyHostToDevice);
    cudaMemcpy( pC_device, c_host, allocSize, cudaMemcpyHostToDevice);

    add_double<<<N,2>>>(N, pA_device, pB_device, pC_device );
    sync.sync();

    cudaMemcpy( c_host, pC_device, allocSize, cudaMemcpyDeviceToHost);

    MonteRayDeviceFree( pA_device );
    MonteRayDeviceFree( pB_device );
    MonteRayDeviceFree( pC_device );
#else
    add_double(N, a_host, b_host, c_host );
#endif
    C = c_host[0];
    return C;
}

/// Adds two floats
float gpuAddTwoFloats( float A, float B) {
    typedef float value_t;

    // Adds two doubles, but uses array notation
    value_t C;

    unsigned N=1;

    value_t a_host[N];
    value_t b_host[N];
    value_t c_host[N];

    unsigned allocSize = N*sizeof(value_t);

    a_host[0] = A;
    b_host[0] = B;
    c_host[0] = 0.0;

#ifdef __CUDACC__
    value_t* pA_device;
    value_t* pB_device;
    value_t* pC_device;

    pA_device = (value_t*) MONTERAYDEVICEALLOC( allocSize, std::string("gpuAddTwoFloats::pA_device") );
    pB_device = (value_t*) MONTERAYDEVICEALLOC( allocSize, std::string("gpuAddTwoFloats::pB_device") );
    pC_device = (value_t*) MONTERAYDEVICEALLOC( allocSize, std::string("gpuAddTwoFloats::pC_device") );

    GPUSync sync;

    cudaMemcpy( pA_device, a_host, allocSize, cudaMemcpyHostToDevice);
    cudaMemcpy( pB_device, b_host, allocSize, cudaMemcpyHostToDevice);
    cudaMemcpy( pC_device, c_host, allocSize, cudaMemcpyHostToDevice);

    add_single<<<N,2>>>(N, pA_device, pB_device, pC_device );
    sync.sync();

    cudaMemcpy( c_host, pC_device, allocSize, cudaMemcpyDeviceToHost);


    MonteRayDeviceFree( pA_device );
    MonteRayDeviceFree( pB_device );
    MonteRayDeviceFree( pC_device );
#else
    add_single(N, a_host, b_host, c_host );
#endif
    C = c_host[0];
    return C;
}

}
