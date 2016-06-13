#include <UnitTest++.h>

#include "global.h"
#include "CartesianGrid.h"

SUITE( CartesianGrid_tests ) {
    TEST( ctor ) {
    	CartesianGrid grid;
    	CHECK_EQUAL( 1001, grid.getMaxNumVertices() );
    }
    TEST( setVertices ) {
    	CartesianGrid grid;
    	grid.setVertices(0, 0.0, 10.0, 10);
    	grid.setVertices(1, 20.0, 30.0, 10);
    	grid.setVertices(2, 40.0, 50.0, 10);

    	CHECK_EQUAL(11, grid.getNumVertices(0) );
    	CHECK_EQUAL(10, grid.getNumBins(0) );

    	CHECK_EQUAL(11, grid.getNumVertices(1) );
    	CHECK_EQUAL(10, grid.getNumBins(1) );

    	CHECK_EQUAL(11, grid.getNumVertices(2) );
    	CHECK_EQUAL(10, grid.getNumBins(2) );

    	CHECK_CLOSE(  0.0, grid.getVertex(0,0), 1e-11 );
    	CHECK_CLOSE(  1.0, grid.getVertex(0,1), 1e-11 );
    	CHECK_CLOSE( 10.0, grid.getVertex(0,10), 1e-11 );

    	CHECK_CLOSE( 20.0, grid.getVertex(1,0), 1e-11 );
    	CHECK_CLOSE( 21.0, grid.getVertex(1,1), 1e-11 );
    	CHECK_CLOSE( 30.0, grid.getVertex(1,10), 1e-11 );

    	CHECK_CLOSE( 40.0, grid.getVertex(2,0), 1e-11 );
    	CHECK_CLOSE( 41.0, grid.getVertex(2,1), 1e-11 );
    	CHECK_CLOSE( 50.0, grid.getVertex(2,10), 1e-11 );
    }
    TEST( finalize ) {
    	CartesianGrid grid;
    	grid.setVertices(0, 0.0, 10.0, 10);
    	grid.setVertices(1, 20.0, 30.0, 10);
    	grid.setVertices(2, 40.0, 50.0, 10);
    	grid.finalize();

    	CHECK_CLOSE(  0.0, grid.getVertex(0,0), 1e-11 );
    	CHECK_CLOSE(  1.0, grid.getVertex(0,1), 1e-11 );
    	CHECK_CLOSE( 10.0, grid.getVertex(0,10), 1e-11 );

    	CHECK_CLOSE( 20.0, grid.getVertex(1,0), 1e-11 );
    	CHECK_CLOSE( 21.0, grid.getVertex(1,1), 1e-11 );
    	CHECK_CLOSE( 30.0, grid.getVertex(1,10), 1e-11 );

    	CHECK_CLOSE( 40.0, grid.getVertex(2,0), 1e-11 );
    	CHECK_CLOSE( 41.0, grid.getVertex(2,1), 1e-11 );
    	CHECK_CLOSE( 50.0, grid.getVertex(2,10), 1e-11 );
    }
    TEST( min_max ) {
    	CartesianGrid grid;
    	grid.setVertices(0, 0.0, 10.0, 10);
    	grid.setVertices(1, 20.0, 30.0, 10);
    	grid.setVertices(2, 40.0, 50.0, 10);
    	grid.finalize();

    	CHECK_CLOSE(  0.0, grid.min(0), 1e-11 );
    	CHECK_CLOSE( 10.0, grid.max(0), 1e-11 );

    	CHECK_CLOSE( 20.0, grid.min(1), 1e-11 );
    	CHECK_CLOSE( 30.0, grid.max(1), 1e-11 );

    	CHECK_CLOSE( 40.0, grid.min(2), 1e-11 );
    	CHECK_CLOSE( 50.0, grid.max(2), 1e-11 );
    }

    TEST( getDimIndex ) {
    	CartesianGrid grid;
    	grid.setVertices(0, 0.0, 10.0, 10);
    	grid.setVertices(1, 0.0, 10.0, 10);
    	grid.setVertices(2, 0.0, 10.0, 10);
    	grid.finalize();

    	CHECK_EQUAL( 0, grid.getDimIndex(0, 0.5) );
    	CHECK_EQUAL( -1, grid.getDimIndex(0, -0.5) );
    	CHECK_EQUAL( 10, grid.getDimIndex(0, 10.5) );

    	CHECK_EQUAL( 0, grid.getDimIndex(1, 0.5) );
    	CHECK_EQUAL( -1, grid.getDimIndex(1, -0.5) );
    	CHECK_EQUAL( 10, grid.getDimIndex(1, 10.5) );

    	CHECK_EQUAL( 0, grid.getDimIndex(2, 0.5) );
    	CHECK_EQUAL( -1, grid.getDimIndex(2, -0.5) );
    	CHECK_EQUAL( 10, grid.getDimIndex(2, 10.5) );

    }

    TEST(getIndex){
    	CartesianGrid grid;
    	grid.setVertices(0, 0.0, 10.0, 10);
    	grid.setVertices(1, 0.0, 10.0, 10);
    	grid.setVertices(2, 0.0, 10.0, 10);
    	grid.finalize();

    	Vector3D pos1( 0.5, 0.5, 0.5);
    	CHECK_EQUAL(  0, grid.getIndex( pos1 ) );

    	Vector3D pos2( 0.5, 1.5, 0.5);
    	CHECK_EQUAL(  10, grid.getIndex( pos2 ) );

    	Vector3D pos3( 0.5, 0.5, 1.5);
    	CHECK_EQUAL(  100, grid.getIndex( pos3 ) );
    }

    TEST(isOutside) {
    	CartesianGrid grid;
    	grid.setVertices(0, 0.0, 10.0, 10);
    	grid.setVertices(1, 0.0, 10.0, 10);
    	grid.setVertices(2, 0.0, 10.0, 10);
    	grid.finalize();


    	int indices1[3] = { 0, 0, 0};
    	CHECK_EQUAL( false, grid.isOutside(indices1) );

    	int indices2[3] = { -1, 0, 0};
    	CHECK_EQUAL( true, grid.isOutside(indices2) );

    	int indices3[3] = { 10, 0, 0};
    	CHECK_EQUAL( true, grid.isOutside(indices3) );

    }


    TEST( crossingInside_to_outside_posDir_one_crossing) {
    	CartesianGrid grid;
    	grid.setVertices(0, 0.0, 10.0, 10);
    	grid.setVertices(1, 0.0, 10.0, 10);
    	grid.setVertices(2, 0.0, 10.0, 10);
    	grid.finalize();

    	Vector3D pos( -0.5, 0.5, 0.5 );
    	Vector3D dir( 1, 0, 0);
    	float_t distance = 1.0;

    	int cells[1000];
    	float_t distances[1000];

    	unsigned nDistances = grid.rayTrace( cells, distances, pos, dir, distance, false );

    	CHECK_EQUAL( 1, nDistances);
    	CHECK_EQUAL( 0U, cells[0]);
    	CHECK_CLOSE( 0.5f, distances[0], 1e-11 );
    }

    TEST(crossingInside_to_outside_posDir_two_crossings) {
    	CartesianGrid grid;
    	grid.setVertices(0, 0.0, 10.0, 10);
    	grid.setVertices(1, 0.0, 10.0, 10);
    	grid.setVertices(2, 0.0, 10.0, 10);
    	grid.finalize();

    	Vector3D pos( -0.5, 0.5, 0.5 );
    	Vector3D dir( 1, 0, 0);
    	float_t distance = 2.0;

    	int cells[1000];
    	float_t distances[1000];

    	unsigned nDistances = grid.rayTrace( cells, distances, pos, dir, distance, false );

    	CHECK_EQUAL( 2, nDistances);
    	CHECK_EQUAL( 0U, cells[0]);
    	CHECK_CLOSE( 1.0f, distances[0], 1e-11 );
    	CHECK_EQUAL( 1U, cells[1]);
    	CHECK_CLOSE( 0.5f, distances[1], 1e-11 );
    }
}


