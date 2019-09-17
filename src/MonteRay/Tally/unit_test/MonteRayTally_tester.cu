#include <UnitTest++.h>

#include <iostream>
#include <iomanip>

#include "MonteRayDefinitions.hh"
#include "GPUUtilityFunctions.hh"
#include "MonteRayTally.hh"

namespace MontRayTally_tester_namespace {

using namespace MonteRay;

SUITE( MonteRayTally_tester ) {
    TEST( ctor ) {
        MonteRayTally tally;
        //CHECK(false);
        CHECK_EQUAL( 1, tally.getNumSpatialBins() );
    }

    TEST( setTimeBinEdges ) {
        MonteRayTally tally;
        std::vector<MonteRay::gpuFloatType_t> timeEdges = { 1.0, 2.0, 10.0, 99.0, 100.0 };
        tally.setTimeBinEdges(timeEdges);
        CHECK_EQUAL( 6, tally.getNumTimeBins() ); // timeEdges + inf
    }

    TEST( initialize ) {
        MonteRayTally tally;
        std::vector<MonteRay::gpuFloatType_t> timeEdges= { 1.0, 2.0, 10.0, 99.0, 100.0 };
        tally.setTimeBinEdges(timeEdges);

        tally.initialize();
        CHECK_EQUAL( true, tally.isInitialized() );

        CHECK_EQUAL( 5, tally.getIndex(0, 5) );
        CHECK_EQUAL( 6, tally.getTallySize() );
    }

    TEST( getIndex ) {
        MonteRayTally tally(4);
        std::vector<MonteRay::gpuFloatType_t> timeEdges= { 1.0, 2.0, 3.0 };
        tally.setTimeBinEdges(timeEdges);
        tally.initialize();

        CHECK_EQUAL( 4,  tally.getNumSpatialBins() );
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
        MonteRayTally tally;
        std::vector<MonteRay::gpuFloatType_t> timeEdges= { 1.0, 2.0, 10.0, 20.0, 100.0 };
        tally.setTimeBinEdges(timeEdges);
        tally.initialize();

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
        MonteRayTally tally;
        std::vector<MonteRay::gpuFloatType_t> timeEdges= { 1.0, 2.0, 10.0, 99.0, 100.0 };
        tally.setTimeBinEdges(timeEdges);
        tally.initialize();

        tally.scoreByIndex(1.0f, 0, 0);
        CHECK_CLOSE( 1.0, tally.getTally(0,0), 1e-6);

        tally.scoreByIndex(1.0f, 0, 0);
        CHECK_CLOSE( 2.0, tally.getTally(0,0), 1e-6);
    }

    TEST( score ) {
        MonteRayTally tally;
        std::vector<MonteRay::gpuFloatType_t> timeEdges= { 1.0, 2.0, 10.0, 99.0, 100.0 };
        tally.setTimeBinEdges(timeEdges);
        tally.initialize();

        gpuFloatType_t time = 1.5;
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
