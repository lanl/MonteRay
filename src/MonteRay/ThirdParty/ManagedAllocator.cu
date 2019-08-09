#include "ManagedAllocator.hh"

#ifdef __CUDACC__
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#endif

namespace MonteRay{

void* Managed::operator new(size_t len) {
#ifdef __CUDACC__
  void *ptr;
  cudaMallocManaged(&ptr, len);
  cudaDeviceSynchronize();
  return ptr;
#else
  return ::operator new(len);
#endif
}

void Managed::operator delete(void *ptr) {
#ifdef __CUDACC__
  cudaDeviceSynchronize();
  cudaFree(ptr);
#else
  ::operator delete(ptr);
#endif
}

} // end namespace MonteRay
