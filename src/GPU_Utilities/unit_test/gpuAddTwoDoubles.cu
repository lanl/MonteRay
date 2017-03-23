#include "gpuAddTwoDoubles.hh"
#include "GPUAtomicAdd.hh"
#include "GPUSync.hh"

namespace MonteRay {

#ifdef CUDA
__global__ void add_single(unsigned N, float *a, float *b, float *c ) {
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	if( bid < N ) {
		if( tid == 0 ) {
			gpu_atomicAdd_single( &c[bid], a[bid] );
		} else if ( tid == 1 ) {
			gpu_atomicAdd_single( &c[bid], b[bid] );
		}
	}
}

__global__ void add_double(unsigned N, double *a, double *b, double *c ) {
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	if( bid < N ) {
		if( tid == 0 ) {
			gpu_atomicAdd_double( &c[bid], a[bid] );
		} else if ( tid == 1 ) {
			gpu_atomicAdd_double( &c[bid], b[bid] );
		}
	}
}
#endif

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

#ifdef CUDA
	value_t* pA_device;
	value_t* pB_device;
	value_t* pC_device;

	GPUSync sync;

	cudaMalloc( &pA_device, allocSize );
	cudaMalloc( &pB_device, allocSize );
	cudaMalloc( &pC_device, allocSize );

	cudaMemcpy( pA_device, a_host, allocSize, cudaMemcpyHostToDevice);
	cudaMemcpy( pB_device, b_host, allocSize, cudaMemcpyHostToDevice);
	cudaMemcpy( pC_device, c_host, allocSize, cudaMemcpyHostToDevice);

	add_double<<<N,2>>>(N, pA_device, pB_device, pC_device );
	sync.sync();

	cudaMemcpy( c_host, pC_device, allocSize, cudaMemcpyDeviceToHost);
	C = c_host[0];

	cudaFree( pA_device );
	cudaFree( pB_device );
	cudaFree( pC_device );
#else
	C = 0.0;
#endif

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

#ifdef CUDA
	value_t* pA_device;
	value_t* pB_device;
	value_t* pC_device;

	cudaMalloc( &pA_device, allocSize );
	cudaMalloc( &pB_device, allocSize );
	cudaMalloc( &pC_device, allocSize );

	GPUSync sync;

	cudaMemcpy( pA_device, a_host, allocSize, cudaMemcpyHostToDevice);
	cudaMemcpy( pB_device, b_host, allocSize, cudaMemcpyHostToDevice);
	cudaMemcpy( pC_device, c_host, allocSize, cudaMemcpyHostToDevice);

	add_single<<<N,2>>>(N, pA_device, pB_device, pC_device );
	sync.sync();

	cudaMemcpy( c_host, pC_device, allocSize, cudaMemcpyDeviceToHost);
	C = c_host[0];

	cudaFree( pA_device );
	cudaFree( pB_device );
	cudaFree( pC_device );
#else
	C = 0.0;
#endif

	return C;
}

}
