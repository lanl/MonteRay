#ifndef MANAGEDALLOCATOR_HH_
#define MANAGEDALLOCATOR_HH_

// A C++ allocator based on cudaMallocManaged
//
// By: Jared Hoberock
// From: https://github.com/jaredhoberock/managed_allocator

// TODO: pull in as a git submodule unless modifications are made.

#include <vector>
#include "ManagedResource.hh"

namespace MonteRay {

class Managed {
public:
  void *operator new(size_t len);
  void operator delete(void *ptr);

  // placment new returns ptr unmodified (cppref)
  void* operator new(size_t, void* ptr){
    return ptr;
  }

  // placement new delete does nothing (cppref)
  void operator delete(void*, void*) noexcept {}
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

    value_type* allocate(size_t n) {
      return static_cast<T*>(ManagedResource().allocate(n*sizeof(T)));
    }

    void deallocate(value_type* ptr, size_t size) {
      ManagedResource().deallocate(static_cast<void*>(ptr), size);
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

template<class T>
using managed_vector = std::vector<T, managed_allocator<T>>;

} // end namespace MonteRay


#endif /* MANAGEDALLOCATOR_HH_ */
