#include <UnitTest++.h>

#ifdef __CUDACC__

#include <thrust/fill.h>
#include <thrust/logical.h>
#include <thrust/execution_policy.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <iostream>

#include "ManagedAllocator.hh"

namespace ManagedAllocator_tester_namespace {

using namespace MonteRay;

template<class T>
using managed_vector = std::vector<T, managed_allocator<T>>;

__global__ void increment_kernel(int *data, size_t n)
{
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i < n)
  {
    data[i] += 1;
  }
}


SUITE( ManagedAllocator_tester ) {

    class setup {
    public:
        setup() {
            vec.resize(n);
            ref.resize(n);
            std::iota(vec.begin(), vec.end(), 0);
            std::iota(ref.begin(), ref.end(), 0);
        }
        const size_t n = 1 << 20;
        managed_vector<int> vec;
        std::vector<int> ref;
    };

    TEST_FIXTURE(setup, test_host ) {
        CHECK_EQUAL(true, std::equal(ref.begin(), ref.end(), vec.begin()) );
    }

    TEST_FIXTURE(setup, test_device ) {

        // we can also use it in a CUDA kernel
        size_t block_size = 256;
        size_t num_blocks = (n + (block_size - 1)) / block_size;

        increment_kernel<<<num_blocks, block_size>>>(vec.data(), vec.size());

        cudaDeviceSynchronize();

        std::for_each( ref.begin(), ref.end(), [](int& x) {x += 1; } );

        CHECK_EQUAL(true, std::equal(ref.begin(), ref.end(), vec.begin()) );
    }

    // Not all Thrust algorithms work - this has been reported by others

    TEST_FIXTURE(setup, test_thrust_host ) {
        // by default, the Thrust algorithm will execute on the host with the managed_vector
        thrust::fill(vec.begin(), vec.end(), 7);

        // fails to compile
        //CHECK_EQUAL(true, thrust::all_of(thrust::host, vec.begin(), vec.end(), [] __host__ __device__ (int x) { return x == 7; }) );

        CHECK_EQUAL(true, all_of( vec.begin(), vec.end(), [] (int x) { return x == 7; } ) );

    }

   TEST_FIXTURE(setup, test_thrust_device ) {
        // test fails on the device

        // to execute on the device, use the thrust::device execution policy
        //thrust::fill(thrust::device, vec.begin(), vec.end(), 13);

        // we need to synchronize before attempting to use the vector on the host
        //cudaDeviceSynchronize();

        // to execute on the host, use the thrust::host execution policy
        //CHECK_EQUAL(true, thrust::all_of(thrust::host, vec.begin(), vec.end(), [] __host__ __device__ (int x) { return x == 13; }) );
        //CHECK_EQUAL(true, all_of( vec.begin(), vec.end(), [] (int x) { return x == 13; } ) );
    }

}

} // end namespace

#endif
