#include <UnitTest++.h>

#include "MonteRay_SpatialGrid_GPU_helper.hh"

namespace MonteRay_SpatialGrid_GPU_tester {

using namespace MonteRay_SpatialGrid_helper;

SUITE( MonteRay_SpatialGrid_GPU_Tests ) {

   	TEST( setup ) {
   		gpuReset();
   	}

	TEST( ctor ) {
		//CHECK(false);
		Grid_t grid;
	}

	TEST_FIXTURE(SpatialGridGPUTester, getNumCells_on_GPU ) {
		//CHECK(false);
		cartesianGrid1_setup();
        CHECK_EQUAL( 1000000, getNumCells() );
    }

	TEST_FIXTURE(SpatialGridGPUTester, getDimension_on_GPU ) {
		cartesianGrid1_setup();
        CHECK_EQUAL( 3, getDimension() );
    }

	TEST_FIXTURE(SpatialGridGPUTester, getCoordinateSystem_on_GPU ) {
		cartesianGrid1_setup();
        CHECK_EQUAL( TransportMeshTypeEnum::Cartesian, getCoordinateSystem() );
    }

	TEST_FIXTURE(SpatialGridGPUTester, isInitialized_on_GPU ) {
		cartesianGrid1_setup();
        CHECK_EQUAL( true, isInitialized() );
    }

	TEST_FIXTURE(SpatialGridGPUTester, getNumberOfGridBins ) {
		cartesianGrid1_setup();
        CHECK_EQUAL( 100, getNumGridBins(0) );
        CHECK_EQUAL( 100, getNumGridBins(1) );
        CHECK_EQUAL( 100, getNumGridBins(2) );
    }

	TEST_FIXTURE(SpatialGridGPUTester, getMinMaxVertices ) {
		cartesianGrid1_setup();
        CHECK_CLOSE( -10.0, getMinVertex(0), 1e-4 );
        CHECK_CLOSE(  10.0, getMaxVertex(0), 1e-4 );
        CHECK_CLOSE( -20.0, getMinVertex(1), 1e-4 );
        CHECK_CLOSE(  20.0, getMaxVertex(1), 1e-4 );
        CHECK_CLOSE( -30.0, getMinVertex(2), 1e-4 );
        CHECK_CLOSE(  30.0, getMaxVertex(2), 1e-4 );
    }

	TEST_FIXTURE(SpatialGridGPUTester, getDelta ) {
		cartesianGrid1_setup();
        CHECK_CLOSE(  0.2, getDelta(0), 1e-4 );
        CHECK_CLOSE(  0.4, getDelta(1), 1e-4 );
        CHECK_CLOSE(  0.6, getDelta(2), 1e-4 );
    }

	TEST_FIXTURE(SpatialGridGPUTester, getVertex ) {
		cartesianGrid1_setup();
        CHECK_CLOSE( -10.0, getVertex(0,0), 1e-4 );
        CHECK_CLOSE(  -9.8, getVertex(0,1), 1e-4 );
        CHECK_CLOSE( -20.0, getVertex(1,0), 1e-4 );
        CHECK_CLOSE( -19.6, getVertex(1,1), 1e-4 );
        CHECK_CLOSE( -30.0, getVertex(2,0), 1e-4 );
        CHECK_CLOSE( -29.4, getVertex(2,1), 1e-4 );
    }

	TEST_FIXTURE(SpatialGridGPUTester, getIndex ) {
		cartesianGrid1_setup();
		unsigned outside = OUTSIDE_INDEX;
		CHECK_EQUAL( outside, getIndex( Position_t(-9999.9,-19.9,-29.9) ) ); // outside neg-x
        CHECK_EQUAL( 0, getIndex( Position_t(-9.9,-19.9,-29.9) ) ); // first index
        CHECK_EQUAL( 1, getIndex( Position_t(-9.7,-19.9,-29.9) ) ); // second index
        CHECK_EQUAL( 1+100, getIndex( Position_t(-9.7,-19.5,-29.9) ) ); // second index of second y slice
        CHECK_EQUAL( 1+100 + 10000, getIndex( Position_t(-9.7,-19.5,-29.3) ) ); // second index of second y slice, of second z slice
    }

	TEST_FIXTURE(SpatialGridGPUTester, getIndex_via_particle ) {
		cartesianGrid1_setup();
		unsigned outside = OUTSIDE_INDEX;
		particle p;
		p.pos = Position_t( -9999.9,-19.9,-29.9 );
		CHECK_EQUAL( outside, getIndex( p ) ); // outside neg-x

		p.pos =  Position_t( -9.9,-19.9,-29.9 );
        CHECK_EQUAL( 0, getIndex( p ) ); // first index

        p.pos =  Position_t( -9.7,-19.9,-29.9 );
        CHECK_EQUAL( 1, getIndex( p ) ); // second index

        p.pos =  Position_t( -9.7,-19.5,-29.9 );
        CHECK_EQUAL( 1+100, getIndex( p ) ); // second index of second y slice

        p.pos =  Position_t( -9.7,-19.5,-29.3 );
        CHECK_EQUAL( 1+100 + 10000, getIndex( p ) ); // second index of second y slice, of second z slice
    }

}

} // end namespace
