#include <UnitTest++.h>

#include <iostream>
#include <iomanip>
#include <sstream>

#include <vector>

#include "MonteRayDefinitions.hh"
#include "GPUUtilityFunctions.hh"
#include "Tally.hh"
#include "Containers.hh"

SUITE( MonteRayTally_tester ) {
  using DataFloat = MonteRay::Tally::DataFloat;
  using TallyFloat = MonteRay::Tally::TallyFloat;
  
  struct TallyData{
    const MonteRay::Vector<DataFloat> energyBinEdges = {0.0, 2.5e-5, 1.0e1};
    const MonteRay::Vector<DataFloat> timeBinEdges = {0.0, 1.5};
    const int nSpatialBins = 10;
  };

  TEST_FIXTURE(TallyData, DefaultBuilder ){
    auto builder = MonteRay::Tally::Builder{};
    builder.spatialBins(nSpatialBins);
    auto tally = builder.build();
    CHECK_EQUAL(10, tally.size());
    CHECK_EQUAL(1, tally.nEnergyBins());
    CHECK_EQUAL(1, tally.nTimeBins());
  }

  void check_tally_construction(const MonteRay::Tally& tally){
    CHECK_EQUAL(10, tally.nSpatialBins());
    CHECK_EQUAL(120, tally.size());

    CHECK_EQUAL(2, tally.timeBinEdges().size());
    CHECK_EQUAL(0.0, tally.timeBinEdges()[0]);
    CHECK_EQUAL(1.5, tally.timeBinEdges()[1]);
    CHECK_EQUAL(3, tally.nTimeBins());

    CHECK_EQUAL(3, tally.energyBinEdges().size());
    CHECK_EQUAL(0.0, tally.energyBinEdges()[0]);
    CHECK_CLOSE(2.5e-5, tally.energyBinEdges()[1], 1e-6);
    CHECK_EQUAL(1.0e1, tally.energyBinEdges()[2]);
    CHECK_EQUAL(4, tally.nEnergyBins());
    CHECK(tally.useStats());
  }

  TEST_FIXTURE(TallyData, FullBuilderReadAndWrite ){
    auto builder = MonteRay::Tally::Builder{};
    builder.spatialBins(nSpatialBins);
    builder.energyBinEdges(energyBinEdges);
    builder.timeBinEdges(timeBinEdges);
    builder.useStats(true);

    auto tally = builder.build();
    check_tally_construction(tally);

    std::stringstream file;
    tally.write(file);
    auto newTally = MonteRay::Tally::Builder::read(file);
    check_tally_construction(newTally);
  }

  class TallyFixture{
    private:
    MonteRay::Vector<DataFloat> energyBinEdges = {0.0, 2.5e-5, 1.0e1};
    MonteRay::Vector<DataFloat> timeBinEdges = {0.0, 1.5};
    int nSpatialBins = 10;
    bool useStats = true;
    public:
    MonteRay::Tally::Builder builder;
    TallyFixture() {
      builder.spatialBins(nSpatialBins);
      builder.energyBinEdges(energyBinEdges);
      builder.timeBinEdges(timeBinEdges);
      builder.useStats(true);
    }
  };

  TEST_FIXTURE(TallyFixture, GetIndex){
    auto tally = builder.build();
    CHECK_EQUAL(0, tally.getIndex(0, 0, 0));
    CHECK_EQUAL(1, tally.getIndex(1, 0, 0));
    CHECK_EQUAL(tally.nSpatialBins(), tally.getIndex(0, 1, 0));
    CHECK_EQUAL(tally.nSpatialBins() + 2, tally.getIndex(2, 1, 0));
    CHECK_EQUAL(2*(tally.nSpatialBins() * tally.nEnergyBins()), tally.getIndex(0, 0, 2));
  }

  TEST_FIXTURE(TallyFixture, Score){
    TallyFloat score = 7.0;
    int spatialBin = 7;
    DataFloat energy = 5.0;
    DataFloat time = 1.0;

    auto tally = builder.build();
    tally.score(score, spatialBin, energy, time);
    int timeBin = 1;
    int energyBin = 2;
    CHECK_EQUAL(score, tally.contribution(spatialBin, energyBin, timeBin));
  }

  TEST_FIXTURE(TallyFixture, AccumulateWithVarianceAndComputeStatsAndReadAndWrite) {
    auto tally = builder.build();
    tally.score(1.0, 0, 3.0, 1.0);
    CHECK_EQUAL(1.0, tally.contribution(0, 2, 1));
    CHECK_EQUAL(0, tally.nSamples());
    tally.accumulate();
    CHECK_EQUAL(0.0, tally.contribution(0, 2, 1));
    tally.score(0.1, 0, 3.0, 1.0);
    CHECK_EQUAL(1, tally.nSamples());
    tally.accumulate();
    CHECK_EQUAL(2, tally.nSamples());

    for (size_t index = 0; index < tally.size(); index++){
      if (index == tally.getIndex(0, 2, 1)){
        CHECK_EQUAL(1.1, tally.mean(index));
        CHECK_EQUAL(1.01, tally.stdDev(index));
      } else {
        CHECK_EQUAL(0.0, tally.mean(index));
        CHECK_EQUAL(0.0, tally.stdDev(index));
      }
    }

    tally.computeStats();
    for (size_t index = 0; index < tally.size(); index++){
      if (index == tally.getIndex(0, 2, 1)){
        CHECK_EQUAL(1.1/2.0, tally.mean(index));
        CHECK_EQUAL(std::sqrt(1.01/2.0 - (1.1*1.1/4.0)), tally.stdDev(index));
      } else {
        CHECK_EQUAL(0.0, tally.mean(index));
        CHECK_EQUAL(0.0, tally.stdDev(index));
      }
    }

    // Test read and write given the tally state after accumulation
    {
      std::stringstream file;
      tally.score(1.0, 0, 3.0, 1.0);
      CHECK_EQUAL(1.0, tally.contribution(0, 2, 1));
      tally.write(file);
      auto newTally = MonteRay::Tally::Builder::read(file);
      CHECK_EQUAL(1.0, newTally.contribution(0, 2, 1));
      for (size_t index = 0; index < newTally.size(); index++){
        if (index == tally.getIndex(0, 2, 1)){
          CHECK_EQUAL(1.1/2.0, tally.mean(index));
          CHECK_EQUAL(std::sqrt(1.01/2.0 - (1.1*1.1/4.0)), tally.stdDev(index));
        } else {
          CHECK_EQUAL(0.0, tally.mean(index));
          CHECK_EQUAL(0.0, tally.stdDev(index));
        }
      }
    }
  }

  TEST_FIXTURE(TallyFixture, AccumulateWithoutVarianceAndComputeStats) {
    builder.useStats(false);
    auto tally = builder.build();
    tally.score(1.0, 0, 3.0, 1.0);
    CHECK_EQUAL(1.0, tally.contribution(0, 2, 1));
    tally.accumulate();
    CHECK_EQUAL(1.0, tally.contribution(0, 2, 1));
    tally.score(0.1, 0, 3.0, 1.0); tally.accumulate();
    CHECK_EQUAL(1.1, tally.contribution(0, 2, 1));

    for (size_t index = 0; index < tally.size(); index++){
      if (index == tally.getIndex(0, 2, 1)){
        CHECK_EQUAL(1.1, tally.mean(index));
        CHECK_EQUAL(0.0, tally.stdDev(index));
      } else {
        CHECK_EQUAL(0.0, tally.mean(index));
        CHECK_EQUAL(0.0, tally.stdDev(index));
      }
    }

    tally.computeStats();

    for (size_t index = 0; index < tally.size(); index++){
      if (index == tally.getIndex(0, 2, 1)){
        CHECK_EQUAL(1.1/2.0, tally.mean(index));
        CHECK_EQUAL(0.0, tally.stdDev(index));
      } else {
        CHECK_EQUAL(0.0, tally.mean(index));
        CHECK_EQUAL(0.0, tally.stdDev(index));
      }
    }
  }

  TEST_FIXTURE(TallyFixture, Clear){
    auto tally = builder.build();
    tally.score(1.0, 0, 3.0, 1.0);
    CHECK_EQUAL(1.0, tally.contribution(0, 2, 1));
    tally.clear();
    CHECK_EQUAL(0.0, tally.contribution(0, 2, 1));
  }

}
