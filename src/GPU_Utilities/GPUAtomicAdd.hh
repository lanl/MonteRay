/*
 * GPUAtomicAdd.hh
 *
 *  Created on: Mar 20, 2017
 *      Author: jsweezy
 */

#ifndef GPUATOMICADD_HH_
#define GPUATOMICADD_HH_

#include "MonteRayDefinitions.hh"

namespace MonteRay{

#ifdef CUDA

__device__ inline void gpu_atomicAdd_single( float *address, float value ) {
	atomicAdd( address,value);
}

__device__ inline void gpu_atomicAdd_double( double *address, double value ) {
	// From: https://www.sharcnet.ca/help/index.php/CUDA_tips_and_tricks
	unsigned long long oldval, newval, readback;

	oldval = __double_as_longlong(*address);
	newval = __double_as_longlong(__longlong_as_double(oldval) + value);
	while ((readback=atomicCAS((unsigned long long *)address, oldval, newval)) != oldval)
	{
		oldval = readback;
		newval = __double_as_longlong(__longlong_as_double(oldval) + value);
	}
}

__device__ inline void gpu_atomicAdd(gpuTallyType_t *address, gpuTallyType_t value){
#if TALLY_DOUBLEPRECISION < 1
	gpu_atomicAdd_single( address,value);
#else
	gpu_atomicAdd_double( address,value);
#endif /* TALLY_DOUBLEPRECISION < 1 */
}

#endif /* CUDA */

} /* namespace MonteRay*/


#endif /* GPUATOMICADD_HH_ */
