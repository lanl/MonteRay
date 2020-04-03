#include <UnitTest++.h>
#include <tuple>
#include <algorithm>
#ifdef __CUDACC__
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#endif

#include "GPUAtomicAdd.hh"
#include "SimpleVector.hh"

SUITE( GPUAtomicAdd_simple_tests ) {
  class testing{ // because UnitTest++ doesn't play well with GPU lambdas
    public:
    MonteRay::SimpleVector<double> cVec;
    MonteRay::SimpleVector<std::tuple<double, double>> ab;
    testing(){
      ab = MonteRay::SimpleVector<std::tuple<double, double>>(1024, {1, 2});
      cVec = MonteRay::SimpleVector<double>(1, 0.0); // one value that's alloc'd via malloc managed
    }
  };

  TEST_FIXTURE(testing, GPUAtomicAdd){
    auto cData = cVec.data();
    auto func = [ cData ] CUDA_CALLABLE_MEMBER (const std::tuple<double, double>& val) {
      MonteRay::gpu_atomicAdd(cData, std::get<0>(val) + std::get<1>(val));
    };
#ifdef __CUDACC__ 
    thrust::for_each(thrust::device, ab.begin(), ab.end(), func);
    cudaDeviceSynchronize();
    auto& c = cVec[0];
    CHECK_EQUAL(3*1024, c);
    c = 0;
    cudaDeviceSynchronize();
    thrust::for_each(thrust::host, ab.begin(), ab.end(), func);
    CHECK_EQUAL(3*1024, c);
    cudaDeviceSynchronize();
#else
    std::for_each(ab.begin(), ab.end(), func);
    auto val = static_cast<double>(3*1024);
    CHECK_EQUAL(c, val);
#endif
  }

}
