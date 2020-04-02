#include <UnitTest++.h>

#include <iostream>
#include <iomanip>

#include "MonteRayDefinitions.hh"
#include "GPUUtilityFunctions.hh"
#include "Tally.hh"

#ifdef __CUDACC__
CUDA_CALLABLE_KERNEL  kernelScore(MonteRay::Tally* const pTally, MonteRay::Tally::TallyFloat value) {
  if (threadIdx.x < pTally->size()){
    pTally->score(value*threadIdx.x, threadIdx.x, time);
  }
}
#endif

SUITE( Tally_ptester ) {

  class Tally_Fixture {
    private:
    using DataFloat = MonteRay::Tally::DataFloat;
    public:
    std::unique_ptr<MonteRay::Tally> pTally;
    Tally_Fixture() {
      int nSpatialBins = 10;
      MonteRay::Vector<DataFloat> energyBins = {2.5e-5, 1.0, 10.0};
      MonteRay::Vector<DataFloat> timeBins = {1.5, 10.0};
      bool useStats = true;

      MonteRay::Tally::Builder builder;
      builder.timeBinEdges(timeBins);
      builder.energyBinEdges(energyBins);
      builder.spatialBins(nSpatialBins);
      builder.useStats(useStats);
      pTally = std::make_unique<MonteRay::Tally>(builder.build());
    }
  };

  TEST_FIXTURE(Tally_Fixture, score_and_gather ) {
      pTally->score(1.0, 0);
      pTally->score(1.0, 9);
      pTally->gather();
      const auto& PA = MonteRay::MonteRayParallelAssistant::getInstance();

      if( PA.getWorldRank() == 0 ) {
          CHECK_CLOSE( 1.0*PA.getWorldSize(), pTally->contribution(0), 1e-6);
          CHECK_CLOSE( 1.0*PA.getWorldSize(), pTally->contribution(9), 1e-6);
      }

      pTally->score(1.0, 0);
      pTally->gather();

      if( PA.getWorldRank() == 0 ) {
        CHECK_CLOSE( 2.0*PA.getWorldSize(), pTally->contribution(0), 1e-6);
        CHECK_CLOSE( 1.0*PA.getWorldSize(), pTally->contribution(9), 1e-6);
      } else {
        CHECK_CLOSE( 0.0, pTally->contribution(0,0), 1e-6);
      }
  }

#ifdef __CUDACC__
  TEST(scoreOnGPUThenGather) {
    constexpr int nBlocks = 100;
    constexpr int nThreadsPerBlock = 256;
    auto pTally = std::make_unique<MonteRay::Tally>(nThreadsPerBlock, true);

    MonteRay::gpuFloatType_t time = 1.5;
    kernelScore<<<nBlocks, nThreadsPerBlock>>>(pTally.get(), 1.0);

    pTally->gather();

    const auto& PA = MonteRay::MonteRayParallelAssistant::getInstance();
    if( PA.getWorldRank() == 0 ) {
      for (size_t i = 0; i < pTally->size(); i++){
        CHECK_CLOSE(static_cast<MonteRay::Tally::TallyFloat>(nBlocks), pTally->contribution(i), 1e-6);
      }
    }
  }
#endif

} // end namespace
