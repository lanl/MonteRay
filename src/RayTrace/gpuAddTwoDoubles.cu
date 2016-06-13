#include "gpuAddTwoDoubles.hh"

namespace MonteRay {

#ifdef CUDA
__global__ void add(unsigned N, int *a, int *b, int *c ) {
	int tid = blockIdx.x;
	if( tid < N ) {
		c[tid] = a[tid] + b[tid];
	}
}
#endif

/// Adds two doubles
value_t gpuAddTwoDoubles( value_t A, value_t B) {
	// Adds two doubles, but uses array notation
	value_t C;

	unsigned N=1;

	int a_host[N];
	int b_host[N];
	int c_host[N];

	a_host[0] = A;
	b_host[0] = B;

#ifdef CUDA
	int* pA_device;
	int* pB_device;
	int* pC_device;

	cudaMalloc( &pA_device, N*sizeof(int) );
	cudaMalloc( &pB_device, N*sizeof(int) );
	cudaMalloc( &pC_device, N*sizeof(int) );

	cudaMemcpy( pA_device, a_host, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy( pB_device, b_host, N*sizeof(int), cudaMemcpyHostToDevice);

	add<<<N,1>>>(N, pA_device, pB_device, pC_device );

	cudaMemcpy( c_host, pC_device, N*sizeof(int), cudaMemcpyDeviceToHost);
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
