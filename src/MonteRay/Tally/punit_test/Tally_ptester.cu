#include <UnitTest++.h>

#include <iostream>
#include <iomanip>

#include "MonteRayDefinitions.hh"
#include "GPUUtilityFunctions.hh"
#include "Tally.hh"
#include "../unit_test/Tally_GPU_test_helper.hh"

namespace Tally_ptester_namespace {

SUITE( Tally_ptester ) {

  class Tally_Fixture {
  public:
    const MonteRay::MonteRayParallelAssistant& PA;

    Tally_Fixture() :
      PA( MonteRay::MonteRayParallelAssistant::getInstance() )
    {

    }

  private:
  };

  TEST_FIXTURE(Tally_Fixture, score_and_gather ) {
      std::vector<MonteRay::gpuFloatType_t> timeEdges= { 1.0, 2.0, 10.0, 99.0, 100.0 };
      int nSpatialBins = 1;
      MonteRay::Tally tally(nSpatialBins, timeEdges);

      tally.scoreByIndex(1.0f, 0, 0);

      tally.gatherWorkGroup(); // used for testing only
      tally.gather();

      if( PA.getWorldRank() == 0 ) {
          CHECK_CLOSE( 1.0*PA.getWorldSize(), tally.getTally(0,0), 1e-6);
      }

      tally.scoreByIndex(1.0f, 0, 0);
      tally.gatherWorkGroup(); // used for testing only
      tally.gather();

      if( PA.getWorldRank() == 0 ) {
        CHECK_CLOSE( 2.0*PA.getWorldSize(), tally.getTally(0,0), 1e-6);
      } else {
        CHECK_CLOSE( 0.0, tally.getTally(0,0), 1e-6);
      }
  }

  TEST_FIXTURE(Tally_Fixture, score ) {
      std::vector<MonteRay::gpuFloatType_t> timeEdges= { 1.0, 2.0, 10.0, 99.0, 100.0 };
      MonteRay::Tally tally(1, timeEdges);

      MonteRay::gpuFloatType_t time = 1.5;
      tally.score(1.0f, 0, time);
      tally.gatherWorkGroup(); // used for testing only
      tally.gather();

      if( PA.getWorldRank() == 0 ) {
        CHECK_CLOSE( 0.0*PA.getWorldSize(), tally.getTally(0,0), 1e-6);
        CHECK_CLOSE( 1.0*PA.getWorldSize(), tally.getTally(0,1), 1e-6);
        CHECK_CLOSE( 0.0*PA.getWorldSize(), tally.getTally(0,2), 1e-6);
      } else {
        CHECK_CLOSE( 0.0*PA.getWorldSize(), tally.getTally(0,0), 1e-6);
        CHECK_CLOSE( 0.0*PA.getWorldSize(), tally.getTally(0,1), 1e-6);
        CHECK_CLOSE( 0.0*PA.getWorldSize(), tally.getTally(0,2), 1e-6);
      }

      time = 2.5;
      tally.score(2.0f, 0, time);
      tally.gatherWorkGroup(); // used for testing only
      tally.gather();

      if( PA.getWorldRank() == 0 ) {
        CHECK_CLOSE( 0.0*PA.getWorldSize(), tally.getTally(0,0), 1e-6);
        CHECK_CLOSE( 1.0*PA.getWorldSize(), tally.getTally(0,1), 1e-6);
        CHECK_CLOSE( 2.0*PA.getWorldSize(), tally.getTally(0,2), 1e-6);
      } else {
        CHECK_CLOSE( 0.0*PA.getWorldSize(), tally.getTally(0,0), 1e-6);
        CHECK_CLOSE( 0.0*PA.getWorldSize(), tally.getTally(0,1), 1e-6);
        CHECK_CLOSE( 0.0*PA.getWorldSize(), tally.getTally(0,2), 1e-6);
      }

  }

  TEST_FIXTURE(Tally_Fixture, score_device ) {
    // std::vector doesn't work here for timeEdges as Tally holds a view
    // to the time bin data and doesn't copy the data into a
    // MonteRay::SimpleVector. Do we want to change that? J. Sweezy

    MonteRay::SimpleVector<MonteRay::gpuFloatType_t> timeEdges= { 1.0, 2.0, 10.0, 99.0, 100.0 };
    TallyGPUTestHelper::TallyGPUTester tally(1,timeEdges);

    MonteRay::gpuFloatType_t time = 1.5;
    tally.score(1.0f, 0, time);

    tally.gatherWorkGroup(); // used for testing only
    tally.gather();

    if( PA.getWorldRank() == 0 ) {
      CHECK_CLOSE( 0.0, tally.getTally(0,0), 1e-6);
      CHECK_CLOSE( 1.0*PA.getWorldSize(), tally.getTally(0,1), 1e-6);
      CHECK_CLOSE( 0.0, tally.getTally(0,2), 1e-6);
    } else {
      CHECK_CLOSE( 0.0, tally.getTally(0,0), 1e-6);
      CHECK_CLOSE( 0.0, tally.getTally(0,1), 1e-6);
      CHECK_CLOSE( 0.0, tally.getTally(0,2), 1e-6);
    }

    time = 2.5;
    tally.score(2.0f, 0, time);

    tally.gatherWorkGroup(); // used for testing only
    tally.gather();

    if( PA.getWorldRank() == 0 ) {
      CHECK_CLOSE( 0.0, tally.getTally(0,0), 1e-6);
      CHECK_CLOSE( 1.0*PA.getWorldSize(), tally.getTally(0,1), 1e-6);
      CHECK_CLOSE( 2.0*PA.getWorldSize(), tally.getTally(0,2), 1e-6);
    } else {
      CHECK_CLOSE( 0.0, tally.getTally(0,0), 1e-6);
      CHECK_CLOSE( 0.0, tally.getTally(0,1), 1e-6);
      CHECK_CLOSE( 0.0, tally.getTally(0,2), 1e-6);
    }
  }

}

} // end namespace
