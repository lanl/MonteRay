#include <UnitTest++.h>

#include <iostream>
#include <iomanip>

#include "MonteRayDefinitions.hh"
#include "GPUUtilityFunctions.hh"
#include "Tally.hh"

#ifdef __CUDACC__
CUDA_CALLABLE_KERNEL  kernelScore(MonteRay::Tally* const pTally, MonteRay::Tally::TallyFloat value) {
  if (threadIdx.x < pTally->size()){
    for (int i = 0; i < pTally->nTimeBins(); i++){
      for (int j = 0; j < pTally->nEnergyBins(); j++){
        pTally->score(value, threadIdx.x, j, i);
      }
    }
  }
}
#endif

SUITE( Tally_ptester ) {

  struct TallyFixture {
    using DataFloat = MonteRay::Tally::DataFloat;
    MonteRay::Tally::Builder tallyBuilder;
    TallyFixture() {
      int nSpatialBins = 10;
      MonteRay::Vector<DataFloat> energyBins = {2.5e-5, 1.0, 10.0};
      MonteRay::Vector<DataFloat> timeBins = {1.5, 10.0};
      bool useStats = true;

      tallyBuilder.timeBinEdges(timeBins);
      tallyBuilder.energyBinEdges(energyBins);
      tallyBuilder.spatialBins(nSpatialBins);
      tallyBuilder.useStats(useStats);
    }
  };

  TEST_FIXTURE(TallyFixture, score_and_gather ) {
      auto pTally = std::make_unique<MonteRay::Tally>(tallyBuilder.build());
      pTally->score(1.0, 0);
      pTally->score(1.0, 9);
      pTally->gatherWorkGroup();
      pTally->gather();
      const auto& PA = MonteRay::MonteRayParallelAssistant::getInstance();
      std::cout << " world Size " << PA.getWorldSize() << " world rank " << PA.getWorldRank() << std::endl;
      if( PA.getWorldRank() == 0 ) {
          CHECK_CLOSE( 1.0*PA.getWorldSize(), pTally->contribution(0), 1e-6);
          CHECK_CLOSE( 1.0*PA.getWorldSize(), pTally->contribution(9), 1e-6);
      }

      pTally->score(1.0, 0);
      pTally->gatherWorkGroup();
      pTally->gather();

      if( PA.getWorldRank() == 0 ) {
        CHECK_CLOSE( 2.0*PA.getWorldSize(), pTally->contribution(0), 1e-6);
        CHECK_CLOSE( 1.0*PA.getWorldSize(), pTally->contribution(9), 1e-6);
      } else {
        CHECK_CLOSE( 0.0, pTally->contribution(0,0), 1e-6);
      }
  }

#ifdef __CUDACC__
  TEST_FIXTURE(TallyFixture, scoreOnGPUThenGather) {
    constexpr int nBlocks = 100;
    constexpr int nThreadsPerBlock = 32;
    MonteRay::Vector<DataFloat> binEdges = {0.5, 1.5};
    tallyBuilder.energyBinEdges( binEdges );
    tallyBuilder.timeBinEdges( binEdges );
    tallyBuilder.spatialBins(nThreadsPerBlock);
    auto pTally = std::make_unique<MonteRay::Tally>(tallyBuilder.build());
    CHECK_EQUAL(9*nThreadsPerBlock, pTally->size());

    kernelScore<<<nBlocks, nThreadsPerBlock>>>(pTally.get(), 1.0);
    cudaDeviceSynchronize();
    pTally->gatherWorkGroup(); // first gather all work-group ranks
    pTally->gather();  // now gather between masters of work groups

    const auto& PA = MonteRay::MonteRayParallelAssistant::getInstance();
    if( PA.getWorldRank() == 0 ) {
      for (size_t i = 0; i < pTally->size(); i++){
        if (static_cast<double>(nBlocks) - pTally->contribution(i) > 1.0e-6) std::cout << i << " test \n";
        CHECK_CLOSE(PA.getWorldSize()*static_cast<MonteRay::Tally::TallyFloat>(nBlocks), pTally->contribution(i), 1e-6);
      }
    }
  }
#endif

} // end namespace
