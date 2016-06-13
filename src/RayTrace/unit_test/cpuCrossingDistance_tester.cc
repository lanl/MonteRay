#include <UnitTest++.h>

#include "global.h"
#include "cpuCrossingDistance.h"

SUITE( cpuCrossingDistanceTest ) {
	typedef global::float_t float_t;

    TEST( crossingInside_to_outside_posDir ) {
    	//calc( float_t* vertices, unsigned nVertices, int* cells, float_t* distances, float_t pos, float_t dir, float_t distance, unsigned index );

    	const unsigned nVertices = 5;
    	float_t vertices[nVertices] = { 0.0, 1.0, 2.0, 3.0, 4.0};

    	int cells[nVertices];
    	float_t distances[nVertices];

    	float_t pos = 0.5;
    	float_t dir = 1.0;
    	float_t distance = 10.0;
    	int index = 0U;

    	unsigned nDistances = CrossingDistance::calc( vertices, nVertices, cells, distances, pos, dir, distance, index);
    	CHECK_EQUAL( 5, nDistances);
    	CHECK_EQUAL( 0, cells[0]);
    	CHECK_CLOSE( 0.5f, distances[0], 1e-11 );
    	CHECK_EQUAL( 1, cells[1]);
    	CHECK_CLOSE( 1.5f, distances[1], 1e-11 );
    	CHECK_EQUAL( 2, cells[2]);
    	CHECK_CLOSE( 2.5f, distances[2], 1e-11 );
    	CHECK_EQUAL( 3, cells[3]);
    	CHECK_CLOSE( 3.5f, distances[3], 1e-11 );
    	CHECK_EQUAL( 4, cells[4]);
    	CHECK_CLOSE( 10.0f, distances[4], 1e-11 );
    }

    TEST( crossingInside_to_outside_negDir) {
    	//calc( float_t* vertices, unsigned nVertices, int* cells, float_t* distances, float_t pos, float_t dir, float_t distance, unsigned index );

    	const unsigned nVertices = 5;
    	float_t vertices[nVertices] = { 0.0, 1.0, 2.0, 3.0, 4.0};

    	int cells[nVertices];
    	float_t distances[nVertices];

    	float_t pos = 0.5;
    	float_t dir = -1.0;
    	float_t distance = 10.0;
    	int index = 0U;

    	unsigned nDistances = CrossingDistance::calc( vertices, nVertices, cells, distances, pos, dir, distance, index);
    	CHECK_EQUAL( 2, nDistances);
    	CHECK_EQUAL( 0, cells[0]);
    	CHECK_CLOSE( 0.5f, distances[0], 1e-11 );
    	CHECK_EQUAL( -1, cells[1]);
    	CHECK_CLOSE( 10.0f, distances[1], 1e-11 );
    }
}
