#ifndef GPUATOMICADD_HH_
#define GPUATOMICADD_HH_
#include "MonteRayTypes.hh"
namespace MonteRay{

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else

__device__ double atomicAdd(double* address, const double val)
{
  unsigned long long int* address_as_ull =
                                  (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
       assumed = old;
       old = atomicCAS(address_as_ull, assumed,
                       __double_as_longlong(val +
                       __longlong_as_double(assumed)));
       // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
     } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

template <typename T, typename U>
CUDA_CALLABLE_MEMBER inline void gpu_atomicAdd(T* address, U value) {
#ifdef __CUDA_ARCH__
    atomicAdd(address, value);
#else
    (*address) += value;
#endif
}

} /* namespace MonteRay*/
#endif /* GPUATOMICADD_HH_ */
