#include "ManagedResource.hh"

#ifdef __CUDACC__
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#endif

namespace MonteRay{

  void* ManagedResource::allocate(size_t n) {
#ifdef __CUDACC__
    void* result = nullptr;

    cudaError_t error = cudaMallocManaged( &result, n, cudaMemAttachGlobal);

    if(error != cudaSuccess) {
      throw thrust::system_error(error, thrust::cuda_category(), "managed_allocator::allocate(): cudaMallocManaged");
    }
    return result;
#else
    return ::operator new(n);
#endif
  }

  void ManagedResource::deallocate(void* ptr, size_t) {
#ifdef __CUDACC__
    cudaError_t error = cudaFree(ptr);

    if(error != cudaSuccess) {
      throw thrust::system_error(error, thrust::cuda_category(), "managed_allocator::deallocate(): cudaFree");
    }
#else
    ::operator delete(ptr);
#endif
  }

}

