#ifndef MANAGEDALLOCATOR_HH_
#define MANAGEDALLOCATOR_HH_

// A C++ allocator based on cudaMallocManaged
//
// By: Jared Hoberock
// From: https://github.com/jaredhoberock/managed_allocator

// TODO: pull in as a git submodule unless modifications are made.

#ifdef __CUDACC__
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>

namespace MonteRay {

class Managed {
public:
  void *operator new(size_t len) {
    void *ptr;
    cudaMallocManaged(&ptr, len);
    cudaDeviceSynchronize();
    return ptr;
  }

  void operator delete(void *ptr) {
    cudaDeviceSynchronize();
    cudaFree(ptr);
  }
};

template<class T>
class managed_allocator
{
  public:
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;

    managed_allocator() {}

    template<class U>
    managed_allocator(const managed_allocator<U>&) {}

    value_type* allocate(size_t n)
    {
      value_type* result = nullptr;

      cudaError_t error = cudaMallocManaged( &result, n*sizeof(T), cudaMemAttachGlobal);

      if(error != cudaSuccess)
      {
        throw thrust::system_error(error, thrust::cuda_category(), "managed_allocator::allocate(): cudaMallocManaged");
      }

      return result;
    }

    void deallocate(value_type* ptr, size_t)
    {
      cudaError_t error = cudaFree(ptr);

      if(error != cudaSuccess)
      {
        throw thrust::system_error(error, thrust::cuda_category(), "managed_allocator::deallocate(): cudaFree");
      }
    }
};

template<class T1, class T2>
bool operator==(const managed_allocator<T1>&, const managed_allocator<T2>&)
{
  return true;
}

template<class T1, class T2>
bool operator!=(const managed_allocator<T1>& lhs, const managed_allocator<T2>& rhs)
{
  return !(lhs == rhs);
}

}
#else
namespace MonteRay {
class Managed {};
}

template<class T>
using managed_allocator = std::allocator<T>;
#endif // end __CUDACC__

#include <vector>
#include <memory>

namespace MonteRay {

#ifdef __CUDACC__
template<class T>
using managed_vector = std::vector<T, managed_allocator<T>>;

#else

template<class T>
using managed_vector = std::vector<T>;

#endif

}


#endif /* MANAGEDALLOCATOR_HH_ */
