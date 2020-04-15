#include <UnitTest++.h>

#include "CudaWrappers.hh"


SUITE(CudaStream_test) {
  TEST(ConstructorAndAssignment){
    MonteRay::cuda::StreamPointer sp;
    auto spCopy = sp;
    CHECK_EQUAL(sp.get(), spCopy.get());
    auto spMove = std::move(spCopy);
    CHECK_EQUAL(sp.get(), spMove.get());
    MonteRay::cuda::StreamPointer spCopyConstructed(sp);
    CHECK_EQUAL(sp.get(), spCopyConstructed.get());
    MonteRay::cuda::StreamPointer spMoveConstructed(std::move(spCopyConstructed));
    CHECK_EQUAL(sp.get(), spMoveConstructed.get());
  }
}

SUITE(CudaEvent_test) {
  TEST(ConstructorAndAssignment){
    MonteRay::cuda::EventPointer ep;
    auto epCopy = ep;
    CHECK_EQUAL(ep.get(), epCopy.get());
    auto epMove = std::move(epCopy);
    CHECK_EQUAL(ep.get(), epMove.get());
    MonteRay::cuda::EventPointer epCopyConstructed(ep);
    CHECK_EQUAL(ep.get(), epCopyConstructed.get());
    MonteRay::cuda::EventPointer epMoveConstructed(std::move(epCopyConstructed));
    CHECK_EQUAL(ep.get(), epMoveConstructed.get());
  }
}


#ifdef __CUDACC__
__global__ void testKernel(int* val){
  atomicAdd(val, 1);
}

__global__ void testWhileKernel(volatile int* checkVal, int* val){
  while (*checkVal == 0){
    // do nothing
  }
  atomicAdd(val, 1);
}

SUITE(CudaStreamAndEventGPU_test){
  TEST(StreamAndEventUsage){
    MonteRay::cuda::StreamPointer pStream1;
    MonteRay::cuda::StreamPointer pStream2;
    MonteRay::cuda::EventPointer pEvent1;
    MonteRay::cuda::EventPointer pEvent2;

    int* val;
    cudaMallocManaged(&val, sizeof(int));
    *val = 0;
    testWhileKernel<<<1, 1, 0, *pStream1>>>(val, val);
    testKernel<<<1, 1, 0, *pStream2>>>(val);
    cudaEventRecord(*pEvent1, *pStream1);
    cudaEventSynchronize(*pEvent1);
    CHECK_EQUAL(2, *val);
    testKernel<<<1, 1, 0, *pStream2>>>(val);
    cudaEventRecord(*pEvent2, *pStream2);
    cudaEventSynchronize(*pEvent2);
    cudaStreamWaitEvent(*pStream2, *pEvent2, 0);
    cudaDeviceSynchronize();
    CHECK_EQUAL(3, *val);

    // check default stream
    cudaStream_t defaultStream = 0;
    MonteRay::cuda::StreamPointer pDefaultStream{MonteRay::cuda::StreamPointer::DefaultStream{}};
    CHECK_EQUAL(defaultStream, *pDefaultStream);
    testKernel<<<1, 1, 0, *pDefaultStream>>>(val);
    cudaDeviceSynchronize();
    CHECK_EQUAL(4, *val);

    cudaFree(val);
    CHECK(pEvent1.get() != pEvent2.get());
    CHECK(pStream1.get() != pStream2.get());
  }
}

#endif
