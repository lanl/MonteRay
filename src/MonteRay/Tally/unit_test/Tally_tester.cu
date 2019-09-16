#include <UnitTest++.h>

#include <iostream>
#include <iomanip>

#include "MonteRayDefinitions.hh"
#include "GPUUtilityFunctions.hh"
#include "Tally.hh"

namespace MontRayTally_tester_namespace {

SUITE( MonteRayTally_tester ) {
  TEST( ConstructorWithoutTimeBins ) {
    MonteRay::Tally tally;
    CHECK_EQUAL( 1, tally.numSpatialBins() );
    CHECK_EQUAL( 1, tally.getTallySize() );
  }

  TEST( ConstructorWithTimeBins ) {
    std::vector<MonteRay::gpuFloatType_t> timeEdges = { 1.0, 2.0, 10.0, 99.0, 100.0 };
    int nSpatialBins = 2;
    MonteRay::Tally tally(nSpatialBins, timeEdges);
    CHECK_EQUAL( 6, tally.getNumTimeBins() ); // timeEdges + inf
    CHECK_EQUAL( 6*2, tally.getTallySize() );
  }

  TEST( getIndex ) {
    std::vector<MonteRay::gpuFloatType_t> timeEdges= { 1.0, 2.0, 3.0 };
    MonteRay::Tally tally(4, timeEdges);

    CHECK_EQUAL( 4,  tally.numSpatialBins() );
    CHECK_EQUAL( 4,  tally.getNumTimeBins() );

    // time bin 0
    CHECK_EQUAL( 0, tally.getIndex(0, 0) );
    CHECK_EQUAL( 1, tally.getIndex(1, 0) );
    CHECK_EQUAL( 2, tally.getIndex(2, 0) );
    CHECK_EQUAL( 3, tally.getIndex(3, 0) );

    // time bin 1
    CHECK_EQUAL( 4, tally.getIndex(0, 1) );
    CHECK_EQUAL( 5, tally.getIndex(1, 1) );
    CHECK_EQUAL( 6, tally.getIndex(2, 1) );
    CHECK_EQUAL( 7, tally.getIndex(3, 1) );

    // time bin 2
    CHECK_EQUAL( 8, tally.getIndex(0, 2) );
    CHECK_EQUAL( 9, tally.getIndex(1, 2) );
    CHECK_EQUAL(10, tally.getIndex(2, 2) );
    CHECK_EQUAL(11, tally.getIndex(3, 2) );

    // time bin 3
    CHECK_EQUAL(12, tally.getIndex(0, 3) );
    CHECK_EQUAL(13, tally.getIndex(1, 3) );
    CHECK_EQUAL(14, tally.getIndex(2, 3) );
    CHECK_EQUAL(15, tally.getIndex(3, 3) );
  }

  TEST( getTimeIndex ) {
      std::vector<MonteRay::gpuFloatType_t> timeEdges= { 1.0, 2.0, 10.0, 20.0, 100.0 };
      int nSpatialBins = 1;
      MonteRay::Tally tally(nSpatialBins, timeEdges);

      CHECK_EQUAL( 5, tally.getTimeIndex( 200.0 ) );
      CHECK_EQUAL( 5, tally.getTimeIndex( 100.0 ) );
      CHECK_EQUAL( 4, tally.getTimeIndex( 90.0 ) );
      CHECK_EQUAL( 3, tally.getTimeIndex( 15.0 ) );
      CHECK_EQUAL( 2, tally.getTimeIndex(  5.0 ) );
      CHECK_EQUAL( 1, tally.getTimeIndex(  1.5 ) );
      CHECK_EQUAL( 1, tally.getTimeIndex(  1.0 ) );
      CHECK_EQUAL( 0, tally.getTimeIndex(  0.5 ) );
  }

  TEST( scoreByIndex ) {
      std::vector<MonteRay::gpuFloatType_t> timeEdges= { 1.0, 2.0, 10.0, 99.0, 100.0 };
      int nSpatialBins = 1;
      MonteRay::Tally tally(nSpatialBins, timeEdges);

      tally.scoreByIndex(1.0f, 0, 0);
      CHECK_CLOSE( 1.0, tally.getTally(0,0), 1e-6);

      tally.scoreByIndex(1.0f, 0, 0);
      CHECK_CLOSE( 2.0, tally.getTally(0,0), 1e-6);
  }

  TEST( score ) {
      std::vector<MonteRay::gpuFloatType_t> timeEdges= { 1.0, 2.0, 10.0, 99.0, 100.0 };
      MonteRay::Tally tally(1, timeEdges);

      MonteRay::gpuFloatType_t time = 1.5;
      tally.score(1.0f, 0, time);
      CHECK_CLOSE( 0.0, tally.getTally(0,0), 1e-6);
      CHECK_CLOSE( 1.0, tally.getTally(0,1), 1e-6);

      time = 2.5;
      tally.score(2.0f, 0, time);
      CHECK_CLOSE( 0.0, tally.getTally(0,0), 1e-6);
      CHECK_CLOSE( 1.0, tally.getTally(0,1), 1e-6);
      CHECK_CLOSE( 2.0, tally.getTally(0,2), 1e-6);
  }
}

} // end namespace
