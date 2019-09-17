#include <UnitTest++.h>

#include "MonteRayCopyMemory.t.hh"

SUITE( CopyMemory_tester ) {

#ifdef __CUDACC__
	using namespace MonteRay;

	template<class T>
	class managedAllocator
	{
	  public:
	    using value_type = T;

	    managedAllocator(){}

	    template<class U>
	    managedAllocator(const managedAllocator<U>&) {}

	    value_type* allocate(size_t n)
	    {
	      value_type* result = nullptr;
	      cudaMallocManaged(&result, n*sizeof(T), cudaMemAttachGlobal);
	      cudaDeviceSynchronize();
	      return result;
	    }

	    void deallocate(value_type* ptr, size_t) noexcept
	    {
	      cudaDeviceSynchronize();
	      cudaFree(ptr);
	    }

	    template <typename U>
	      bool operator==(managedAllocator<U> const & rhs) const
	      {
	        return true;
	      }

	    template <typename U>
	      bool operator!=(managedAllocator<U> const & rhs) const
	      {
	        return false;
	      }

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

	template< typename T>
	class MonteRayVector : public std::vector<T, managedAllocator<T>> {


	};

	__global__ void increment_kernel(int *data, size_t n)
	{
	  size_t i = blockDim.x * blockIdx.x + threadIdx.x;

	  if(i < n) {
	    data[i] += 1;
	  }
	}

	TEST( ctor ) {
		MonteRayVector<int> managedVector;
		managedVector.push_back(1);
		managedVector.push_back(3);
		managedVector.push_back(5);
		managedVector.push_back(7);

		CHECK_EQUAL(1,managedVector[0]);
		CHECK_EQUAL(3,managedVector[1]);
		CHECK_EQUAL(5,managedVector[2]);
		CHECK_EQUAL(7,managedVector[3]);

		increment_kernel<<<1,4>>>(managedVector.data(), managedVector.size());
		cudaDeviceSynchronize();

		CHECK_EQUAL(2,managedVector[0]);
		CHECK_EQUAL(4,managedVector[1]);
		CHECK_EQUAL(6,managedVector[2]);
		CHECK_EQUAL(8,managedVector[3]);
	}

#endif

}

