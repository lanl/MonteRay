#ifndef GPUATOMICADD_HH_
#define GPUATOMICADD_HH_

/// DO NOT INCLUDE IN ANOTHER HEADER ONLY CALL FROM T.HH OR .CU FILE

#include <stdio.h>

#include "MonteRayDefinitions.hh"

namespace MonteRay{

CUDA_CALLABLE_MEMBER inline void gpu_atomicAdd_single( float *address, float value ) {
#ifdef __CUDA_ARCH__
    atomicAdd( address,value);
#else
    (*address) += value;
#endif
}

CUDA_CALLABLE_MEMBER inline void gpu_atomicAdd_double( double *address, double value ) {
    //	printf("Debug: MonteRay::GPUAtomicAdd.h::gpu_atomicAdd_double **************\n");
#ifdef __CUDA_ARCH__
    // From: https://www.sharcnet.ca/help/index.php/CUDA_tips_and_tricks
    unsigned long long oldval, newval, readback;

    oldval = __double_as_longlong(*address);
    newval = __double_as_longlong(__longlong_as_double(oldval) + value);
    while ((readback=atomicCAS((unsigned long long *)address, oldval, newval)) != oldval)
    {
        oldval = readback;
        newval = __double_as_longlong(__longlong_as_double(oldval) + value);
    }
#else
    (*address) += value;
#endif
}

CUDA_CALLABLE_MEMBER inline void gpu_atomicAdd(gpuTallyType_t *address, gpuTallyType_t value){
#if TALLY_DOUBLEPRECISION < 1
    gpu_atomicAdd_single( address,value);
#else
    gpu_atomicAdd_double( address,value);
#endif /* TALLY_DOUBLEPRECISION < 1 */
}

} /* namespace MonteRay*/


#endif /* GPUATOMICADD_HH_ */
