#include <UnitTest++.h>
#include "Array.hh"
#include "GPUErrorCheck.hh"

SUITE( Array_tester ) {
  using namespace MonteRay;

  TEST( Array_ctor ){
  Array<double, 3> arr;
  CHECK_EQUAL(arr.size(), 3);
  CHECK_EQUAL(arr.data(), &arr[0]);
  CHECK_EQUAL(&(*arr.begin()), &arr[0]);
  CHECK_EQUAL(&(*arr.end()), &arr[0] + 3);
  arr = {0.0, 1.0, 2.0};
  CHECK_EQUAL(arr[0], 0.0);
  CHECK_EQUAL(arr[1], 1.0);
  CHECK_EQUAL(arr[2], 2.0);
  Array<double, 3> another_array{0.1, 1.1, 2.1};
  arr = another_array;
  CHECK_EQUAL(arr[0], 0.1);
  CHECK_EQUAL(arr[1], 1.1);
  CHECK_EQUAL(arr[2], 2.1);
  }

  TEST( Array_iters ){
    Array<double, 3> arr{0.1, 1.1, 2.1};
    double result = 0;
    for (auto&& val : arr){
      result += val;
    }
    CHECK_EQUAL(result, 0.1 + 1.1 + 2.1);
  }

#ifdef __CUDACC__
    CUDA_CALLABLE_KERNEL sum_arr(double* result, Array<double, 3> arr){
      for (auto&& val : arr){
        *result += val;
      }
      return;
    }
    TEST( Array_cuda ) {
      double* result;
      cudaMallocManaged(&result, sizeof(double));
      *result = 0;
      Array<double, 3> arr{0.1, 1.1, 2.1};
      sum_arr<<<1,1>>>(result, arr);
      cudaDeviceSynchronize();
      CHECK_EQUAL(*result, 0.1 + 1.1 + 2.1);
    }
#endif
}
