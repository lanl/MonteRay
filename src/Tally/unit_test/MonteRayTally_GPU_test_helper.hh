#ifndef MONTERAYTALLY_GPU_TEST_HELPER_HH_
#define MONTERAYTALLY_GPU_TEST_HELPER_HH_

#include "MonteRayDefinitions.hh"
#include "GPUUtilityFunctions.hh"
#include "MonteRayTally.hh"
#include "MonteRayTypes.hh"
#include "MonteRay_SingleValueCopyMemory.t.hh"

namespace MonteRayTallyGPUTestHelper {

using namespace MonteRay;

template<typename T>
using resultClass = MonteRay_SingleValueCopyMemory<T>;

CUDA_CALLABLE_KERNEL  kernelGetNumSpatialBins(MonteRayTally* pTally, resultClass<unsigned>* pResult ){
    pResult->v = pTally->getNumSpatialBins();
}

CUDA_CALLABLE_KERNEL  kernelGetNumTimeBins(MonteRayTally* pTally, resultClass<unsigned>* pResult ){
    pResult->v = pTally->getNumTimeBins();
}

CUDA_CALLABLE_KERNEL  kernelScoreByIndex(MonteRayTally* pTally, gpuTallyType_t value, unsigned spatial_index, unsigned time_index=0 ) {
   pTally->scoreByIndex(value, spatial_index, time_index);
}

CUDA_CALLABLE_KERNEL  kernelScore(MonteRayTally* pTally, gpuTallyType_t value, unsigned spatial_index, gpuFloatType_t time = 0.0 ) {
   pTally->score(value, spatial_index, time);
}

CUDA_CALLABLE_KERNEL  kernelGetTally(MonteRayTally* pTally, resultClass<gpuTallyType_t>* pResult, unsigned spatial_index, unsigned time_index=0 ) {
    pResult->v = pTally->getTally( spatial_index, time_index);
}

CUDA_CALLABLE_KERNEL  kernelGetTallySize(MonteRayTally* pTally, resultClass<unsigned>* pResult ) {
    pResult->v = pTally->getTallySize();
}

class MonteRayTallyGPUTester {
public:
    std::unique_ptr<MonteRayTally> pTally;

    MonteRayTallyGPUTester(unsigned default_setup = 1 ){
       if( default_setup == 1 ) setup1();
       if( default_setup == 2 ) setup2();
    }

    void setup1( unsigned default_setup =1 ){
        pTally = std::unique_ptr<MonteRayTally>( new MonteRayTally() );

        std::vector<gpuFloatType_t> timeEdges = { 1.0, 2.0, 10.0, 20.0, 100.0 };
        pTally->setTimeBinEdges(timeEdges);
        pTally->initialize();
        pTally->copyToGPU();
    }

    void setup2(){
         pTally = std::unique_ptr<MonteRayTally>( new MonteRayTally() );

         pTally->initialize();
         pTally->copyToGPU();
     }

    unsigned getNumSpatialBins() {
        using result_t = resultClass<unsigned>;
        std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );

#ifdef __CUDACC__
        pResult->copyToGPU();
        kernelGetNumSpatialBins<<<1,1>>>( pTally->devicePtr, pResult->devicePtr);
        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelGetNumSpatialBins( pTally.get(), pResult.get() );
#endif

        return pResult->v;
    }

    unsigned getNumTimeBins() {
        using result_t = resultClass<unsigned>;
        std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );

#ifdef __CUDACC__
        pResult->copyToGPU();
        kernelGetNumTimeBins<<<1,1>>>( pTally->devicePtr, pResult->devicePtr);
        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelGetNumTimeBins( pTally.get(), pResult.get() );
#endif

        return pResult->v;
    }

    void scoreByIndex(gpuTallyType_t value, unsigned spatial_index, unsigned time_index=0 ) {
#ifdef __CUDACC__
        kernelScoreByIndex<<<1,1>>>( pTally->devicePtr, value, spatial_index, time_index);
        gpuErrchk( cudaPeekAtLastError() );
#else
        kernelScoreByIndex( pTally.get(), value, spatial_index, time_index);
#endif
    }

    void score(gpuTallyType_t value, unsigned spatial_index, gpuFloatType_t time = 0.0 ) {
#ifdef __CUDACC__
        kernelScore<<<1,1>>>( pTally->devicePtr, value, spatial_index, time);
        gpuErrchk( cudaPeekAtLastError() );
#else
        kernelScore( pTally.get(), value, spatial_index, time);
#endif
    }

    gpuTallyType_t getTally(unsigned spatial_index, unsigned time_index=0 ) {
        using result_t = resultClass<gpuTallyType_t>;
        std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );

#ifdef __CUDACC__
        pResult->copyToGPU();
        kernelGetTally<<<1,1>>>( pTally->devicePtr, pResult->devicePtr, spatial_index, time_index );
        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelGetTally( pTally.get(), pResult.get(), spatial_index, time_index );
#endif

        return pResult->v;
    }

    void copyToCPU() {
        pTally->copyToCPU();
    }

    gpuTallyType_t getCPUTally(unsigned spatial_index, unsigned time_index=0 ) {
        return pTally->getTally( spatial_index, time_index );
    }

    unsigned getIndex(unsigned spatial_index, unsigned time_index=0 ) {
        return pTally->getIndex(spatial_index, time_index );
    }

    unsigned getTimeIndex( gpuFloatType_t time ) {
        return pTally->getTimeIndex(time);
    }

    unsigned getGPUTallySize() const {
        using result_t = resultClass<unsigned>;
        std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );

#ifdef __CUDACC__
        pResult->copyToGPU();
        kernelGetTallySize<<<1,1>>>( pTally->devicePtr, pResult->devicePtr );
        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelGetTallySize( pTally.get(), pResult.get() );
#endif

        return pResult->v;
    }

    unsigned getCPUTallySize() const {
        return pTally->getTallySize();
    }

    void setupForParallel() const {
        pTally->setupForParallel();
    }

    void gather() const {
        pTally->gather();
    }
};

}

#endif /* MONTERAYTALLY_GPU_TEST_HELPER_HH_ */
