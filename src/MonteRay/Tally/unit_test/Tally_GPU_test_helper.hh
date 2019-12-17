#ifndef TALLY_GPU_TEST_HELPER_HH_
#define TALLY_GPU_TEST_HELPER_HH_

#include "Tally.hh"
#include "GPUErrorCheck.hh"

namespace TallyGPUTestHelper {

using Tally_t = MonteRay::Tally;

using namespace MonteRay;

CUDA_CALLABLE_KERNEL  kernelScore(Tally_t* const pTally, gpuTallyType_t value, unsigned spatial_index, gpuFloatType_t time = 0.0 ) {
  pTally->score(value, spatial_index, time);
}

class TallyGPUTester : public Managed {
public:
    std::unique_ptr<Tally_t> pTally;

    template <typename TimeEdges>
    TallyGPUTester(int numSpatialBins, TimeEdges&& timeEdges) {
      pTally = std::make_unique<Tally_t>(numSpatialBins, timeEdges);
    }

    void score(gpuTallyType_t value, unsigned spatial_index, gpuFloatType_t time = 0.0 ) {
#ifdef __CUDACC__
        kernelScore<<<1,1>>>( pTally.get(), value, spatial_index, time);
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );
#else
        kernelScore( pTally.get(), value, spatial_index, time);
#endif
    }

    CUDA_CALLABLE_MEMBER
    gpuTallyType_t getTally(int spatial_index, int time_index = 0 ) const {
      return pTally->getTally( spatial_index, time_index);
    }

    void gatherWorkGroup(){
      pTally->gatherWorkGroup();
    }

    void gather(){
      pTally->gather();
    }
};

}

#endif
