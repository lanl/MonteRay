#include <UnitTest++.h>
#include <tuple>

#include "GPUUtilityFunctions.hh"
#include "SimpleVector.hh"

SUITE( GPUAtomicAdd_simple_tests ) {

  MonteRay::SimpleVector<std::tuple<double, double>> ab(1024, {1, 2});
  MonteRay::SimpleVector<double> cVec(1, 0.0);

   auto func = [ c = cVec.data() ] (const auto& val) {
     gpu_atomicAdd(c, std::get<0>(val) + std::get<1>(val));
   };

   auto c = cVec[0];
#ifdef __CUDACC__ 
   thrust::for_each(thrust::device, ab.begin(), ab.end(), func);
   CHECK_EQUAL(c, 3*1024);
   c = 0;
   thrust::for_each(thrust::host, ab.begin(), ab.end(), func);
   CHECK_EQUAL(c, 3*1024);
#else
   /* std::for_each(ab.begin(), ab.end(), func); */
   auto val = static_cast<double>(3*1024);
   /* CHECK_EQUAL(c, val); */

#endif

}
