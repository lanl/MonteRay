#include <UnitTest++.h>

#include "GridBins.h"
#include "cpuRayTrace.h"

using namespace MonteRay;

SUITE( GridBins_Tester ) {

	class GridBinsTest {
	public:
		typedef global::float_t float_t;

		GridBinsTest(){
			grid = (GridBins*) malloc( sizeof(GridBins) );
			ctor( grid );
		}

		~GridBinsTest(){
			free( grid );
		}

		GridBins* grid;
	};

	TEST_FIXTURE( GridBinsTest, ctor ) {
		CHECK_EQUAL( 1001, getMaxNumVertices(grid) );
		CHECK_EQUAL( 1001, grid->offset[1] );
	}

	TEST_FIXTURE( GridBinsTest, setVertices ) {
		setVertices(grid, 0,  0.0, 10.0, 10);
		setVertices(grid, 1, 20.0, 30.0, 10);
		setVertices(grid, 2, 40.0, 50.0, 10);

		CHECK_EQUAL(11, getNumVertices(grid, 0) );
		CHECK_EQUAL(10, getNumBins(grid, 0) );

		CHECK_EQUAL(11, getNumVertices(grid, 1) );
		CHECK_EQUAL(10, getNumBins(grid, 1) );

		CHECK_EQUAL(11, getNumVertices(grid, 2) );
		CHECK_EQUAL(10, getNumBins(grid, 2) );

		CHECK_CLOSE(  0.0, getVertex(grid, 0,  0), 1e-11 );
		CHECK_CLOSE(  1.0, getVertex(grid, 0,  1), 1e-11 );
		CHECK_CLOSE( 10.0, getVertex(grid, 0, 10), 1e-11 );

		CHECK_CLOSE( 20.0, getVertex(grid, 1,  0), 1e-11 );
		CHECK_CLOSE( 21.0, getVertex(grid, 1,  1), 1e-11 );
		CHECK_CLOSE( 30.0, getVertex(grid, 1, 10), 1e-11 );

		CHECK_CLOSE( 40.0, getVertex(grid, 2,  0), 1e-11 );
		CHECK_CLOSE( 41.0, getVertex(grid, 2,  1), 1e-11 );
		CHECK_CLOSE( 50.0, getVertex(grid, 2, 10), 1e-11 );
	}

	TEST_FIXTURE( GridBinsTest, finalize ) {
		setVertices(grid, 0, 0.0, 10.0, 10);
		setVertices(grid, 1, 20.0, 30.0, 10);
		setVertices(grid, 2, 40.0, 50.0, 10);
		finalize(grid);

		CHECK_CLOSE(  0.0, getVertex(grid, 0,0), 1e-11 );
		CHECK_CLOSE(  1.0, getVertex(grid, 0,1), 1e-11 );
		CHECK_CLOSE( 10.0, getVertex(grid, 0,10), 1e-11 );

		CHECK_CLOSE( 20.0, getVertex(grid, 1,0), 1e-11 );
		CHECK_CLOSE( 21.0, getVertex(grid, 1,1), 1e-11 );
		CHECK_CLOSE( 30.0, getVertex(grid, 1,10), 1e-11 );

		CHECK_CLOSE( 40.0, getVertex(grid, 2,0), 1e-11 );
		CHECK_CLOSE( 41.0, getVertex(grid, 2,1), 1e-11 );
		CHECK_CLOSE( 50.0, getVertex(grid, 2,10), 1e-11 );
	}

	TEST_FIXTURE(GridBinsTest, numXY)
	{

		setVertices(grid, 0, 0.0, 10.0, 10);
		setVertices(grid, 1, 20.0, 30.0, 10);
		setVertices(grid, 2, 40.0, 50.0, 10);
		finalize(grid);

		CHECK_EQUAL( 100, getNumXY(grid)  );
	}

	TEST_FIXTURE(GridBinsTest, min_max)
	{
		setVertices(grid, 0, 0.0, 10.0, 10);
		setVertices(grid, 1, 20.0, 30.0, 10);
		setVertices(grid, 2, 40.0, 50.0, 10);
		finalize(grid);

		CHECK_CLOSE(  0.0, min(grid, 0), 1e-11 );
		CHECK_CLOSE( 10.0, max(grid, 0), 1e-11 );

		CHECK_CLOSE( 20.0, min(grid, 1), 1e-11 );
		CHECK_CLOSE( 30.0, max(grid, 1), 1e-11 );

		CHECK_CLOSE( 40.0, min(grid, 2), 1e-11 );
		CHECK_CLOSE( 50.0, max(grid, 2), 1e-11 );
	}

	TEST_FIXTURE(GridBinsTest, getDimIndex)
	{
		setVertices(grid, 0, 0.0, 10.0, 10);
		setVertices(grid, 1, 0.0, 10.0, 10);
		setVertices(grid, 2, 0.0, 10.0, 10);
		finalize(grid);

		CHECK_EQUAL( 0, getDimIndex(grid, 0, 0.5) );
		CHECK_EQUAL( -1, getDimIndex(grid, 0, -0.5) );
		CHECK_EQUAL( 10, getDimIndex(grid, 0, 10.5) );

		CHECK_EQUAL( 0, getDimIndex(grid, 1, 0.5) );
		CHECK_EQUAL( -1, getDimIndex(grid, 1, -0.5) );
		CHECK_EQUAL( 10, getDimIndex(grid, 1, 10.5) );

		CHECK_EQUAL( 0, getDimIndex(grid, 2, 0.5) );
		CHECK_EQUAL( -1, getDimIndex(grid, 2, -0.5) );
		CHECK_EQUAL( 10, getDimIndex(grid, 2, 10.5) );
	}

	TEST_FIXTURE(GridBinsTest, getIndex)
	{
		setVertices(grid, 0, 0.0, 10.0, 10);
		setVertices(grid, 1, 0.0, 10.0, 10);
		setVertices(grid, 2, 0.0, 10.0, 10);
		finalize(grid);

		Vector3D pos1( 0.5, 0.5, 0.5);
		CHECK_EQUAL(  0, getIndex(grid,  pos1 ) );

		Vector3D pos2( 0.5, 1.5, 0.5);
		CHECK_EQUAL(  10, getIndex(grid,  pos2 ) );

		Vector3D pos3( 0.5, 0.5, 1.5);
		CHECK_EQUAL(  100, getIndex(grid,  pos3 ) );
	}

	TEST_FIXTURE(GridBinsTest, isOutside)
	{
		setVertices(grid, 0, 0.0, 10.0, 10);
		setVertices(grid, 1, 0.0, 10.0, 10);
		setVertices(grid, 2, 0.0, 10.0, 10);
		finalize(grid);


		int indices1[3] = { 0, 0, 0};
		CHECK( isOutside(grid, indices1) == false );

		int indices2[3] = { -1, 0, 0};
		CHECK( isOutside(grid, indices2) == true );

		int indices3[3] = { 10, 0, 0};
		CHECK( isOutside(grid, indices3) == true );

	}


	class GridBinsRayTraceTest{
	public:
		typedef global::float_t float_t;

		GridBinsRayTraceTest(){
			grid = (GridBins*) malloc( sizeof(GridBins) );
			ctor( grid );
			setVertices(grid, 0, 0.0, 10.0, 10);
			setVertices(grid, 1, 0.0, 10.0, 10);
			setVertices(grid, 2, 0.0, 10.0, 10);
			finalize(grid);
		}

		~GridBinsRayTraceTest(){
			free( grid );
		}

		GridBins* grid;
	};

	TEST_FIXTURE(GridBinsRayTraceTest, getNumCells)
	{
		CHECK_EQUAL( 1000, getNumCells(grid) );
	}

	TEST_FIXTURE(GridBinsRayTraceTest, getCenterPoint)
	{
		Position_t pos;
		getCenterPointByIndex(grid, 0, pos );
		CHECK_CLOSE( 0.5, pos[0], 1e-11 );
		CHECK_CLOSE( 0.5, pos[1], 1e-11 );
		CHECK_CLOSE( 0.5, pos[2], 1e-11 );
	}

	TEST_FIXTURE(GridBinsRayTraceTest, Distance1)
	{
		Position_t pos;
		getCenterPointByIndex(grid, 0, pos );

		Position_t start( 1.5, 0.5, 0.5);

		float_t distance = getDistance( pos, start );
		CHECK_CLOSE( 1.0, distance, 1e-11);
	}

	TEST_FIXTURE(GridBinsRayTraceTest, Distance2)
	{
		Position_t pos;
		getCenterPointByIndex(grid, 0, pos );

		Position_t start( 1.5, 0.5, 1.5);

		float_t distance = getDistance( pos, start );
		CHECK_CLOSE( 1.0f*sqrt(2.0f), distance, 1e-7);
	}

	TEST_FIXTURE(GridBinsRayTraceTest, getCenterPointbyIndices)
	{
		Position_t pos;
		unsigned indices[3];
		indices[0] = 0; indices[1] = 0; indices[2] = 0;
		getCenterPointByIndices(grid, indices, pos );
		CHECK_CLOSE( 0.5, pos[0], 1e-11 );
		CHECK_CLOSE( 0.5, pos[1], 1e-11 );
		CHECK_CLOSE( 0.5, pos[2], 1e-11 );
	}


	TEST_FIXTURE(GridBinsRayTraceTest, crossingInside_to_outside_posDir_one_crossing)
	{
		Vector3D pos( -0.5, 0.5, 0.5 );
		Vector3D dir( 1, 0, 0);
		float_t distance = 1.0;

		int cells[1000];
		float_t distances[1000];

		unsigned nDistances = rayTrace(grid, cells, distances, pos, dir, distance, false );

		CHECK_EQUAL( 1, nDistances);
		CHECK_EQUAL( 0U, cells[0]);
		CHECK_CLOSE( 0.5f, distances[0], 1e-11 );
	}

	TEST_FIXTURE(GridBinsRayTraceTest, crossingInside_to_outside_posDir_two_crossings)
	{
		Vector3D pos( -0.5, 0.5, 0.5 );
		Vector3D dir( 1, 0, 0);
		float_t distance = 2.0;

		int cells[1000];
		float_t distances[1000];

		unsigned nDistances = rayTrace(grid, cells, distances, pos, dir, distance, false );

		CHECK_EQUAL( 2, nDistances);
		CHECK_EQUAL( 0U, cells[0]);
		CHECK_CLOSE( 1.0f, distances[0], 1e-11 );
		CHECK_EQUAL( 1U, cells[1]);
		CHECK_CLOSE( 0.5f, distances[1], 1e-11 );
	}

	class GridBinsRayTraceTest2{
	public:
		typedef global::float_t float_t;

		GridBinsRayTraceTest2(){
			grid = (GridBins*) malloc( sizeof(GridBins) );
			ctor( grid );
			setVertices(grid, 0, -5.0, 5.0, 10);
			setVertices(grid, 1, -5.0, 5.0, 10);
			setVertices(grid, 2, -5.0, 5.0, 10);
			finalize(grid);
		}

		~GridBinsRayTraceTest2(){
			free( grid );
		}

		GridBins* grid;
	};

	TEST_FIXTURE(GridBinsRayTraceTest2, Distance1)
	{
		Vector3D pos( 0.5, 0.5, 0.5 );
		Vector3D dir( 1, 0, 0);
		float_t distance = 1.0;

		int cells[1000];
		float_t distances[1000];

		unsigned nDistances = rayTrace(grid, cells, distances, pos, dir, distance, false );

		unsigned i = MonteRay::getIndex( grid, pos );
		CHECK_EQUAL( 555, i);

		CHECK_EQUAL( 2, nDistances);
		CHECK_EQUAL( 555U, cells[0]);
		CHECK_CLOSE( 0.5f, distances[0], 1e-11 );
	}

}
