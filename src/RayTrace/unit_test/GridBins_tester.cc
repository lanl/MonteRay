#include <UnitTest++.h>

#include "GridBins.h"
#include "cpuRayTrace.h"

using namespace MonteRay;

SUITE( GridBins_Tester ) {

	class GridBinsTest {
	public:

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

		CHECK_EQUAL( true, isRegular(grid,0) );
		CHECK_EQUAL( true, isRegular(grid,1) );
		CHECK_EQUAL( true, isRegular(grid,2) );

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

		Vector3D<double> pos1( 0.5, 0.5, 0.5);
		CHECK_EQUAL(  0, getIndex(grid,  pos1 ) );

		Vector3D<double> pos2( 0.5, 1.5, 0.5);
		CHECK_EQUAL(  10, getIndex(grid,  pos2 ) );

		Vector3D<double> pos3( 0.5, 0.5, 1.5);
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

    TEST_FIXTURE(GridBinsTest, is_getRegular ) {
    	std::vector<double> vertices {-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
     			                        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10};

        setVertices(grid, 0, vertices );

        CHECK_EQUAL( false, isRegular(grid,0) );
    }

    TEST_FIXTURE(GridBinsTest, regular_getDimIndex ) {

        setVertices(grid, 0, -10, 10, 20 );

        CHECK_EQUAL( -1, getDimIndex( grid, 0, -10.5) );
        CHECK_EQUAL( 0, getDimIndex( grid, 0, -9.5) );
        CHECK_EQUAL( 1, getDimIndex( grid, 0, -8.5) );
        CHECK_EQUAL( 18, getDimIndex( grid, 0,  8.5) );
        CHECK_EQUAL( 19, getDimIndex( grid, 0,  9.5) );
        CHECK_EQUAL( 20, getDimIndex( grid, 0,  10.5) );
    }

    TEST_FIXTURE(GridBinsTest, regular_but_force_irregular_getDimIndex_1D ) {

        setVertices(grid, 0, -10, 10, 20 );
        grid->isRegular[0] = false;

        CHECK_EQUAL( -1, getDimIndex( grid, 0, -10.5) );
        CHECK_EQUAL( 0, getDimIndex( grid, 0, -9.5) );
        CHECK_EQUAL( 1, getDimIndex( grid, 0, -8.5) );
        CHECK_EQUAL( 18, getDimIndex( grid, 0,  8.5) );
        CHECK_EQUAL( 19, getDimIndex( grid, 0,  9.5) );
        CHECK_EQUAL( 20, getDimIndex( grid, 0,  10.5) );
    }

    TEST_FIXTURE(GridBinsTest, regular_but_force_irregular_getDimIndex_3D ) {

        setVertices(grid, 0, -10, 10, 10 );
        setVertices(grid, 1, -100, 100, 10 );
        setVertices(grid, 2, -20, 20, 10 );
        finalize(grid);

        CHECK_CLOSE( -10.0, grid->vertices[0], 1e-5 );
        CHECK_CLOSE(  -8.0, grid->vertices[1], 1e-5 );
        CHECK_CLOSE(  -6.0, grid->vertices[2], 1e-5 );
        CHECK_CLOSE(  -4.0, grid->vertices[3], 1e-5 );
        CHECK_CLOSE(  -2.0, grid->vertices[4], 1e-5 );
        CHECK_CLOSE(   0.0, grid->vertices[5], 1e-5 );
        CHECK_CLOSE(   2.0, grid->vertices[6], 1e-5 );
        CHECK_CLOSE(   4.0, grid->vertices[7], 1e-5 );
        CHECK_CLOSE(   6.0, grid->vertices[8], 1e-5 );
        CHECK_CLOSE(   8.0, grid->vertices[9], 1e-5 );
        CHECK_CLOSE(  10.0, grid->vertices[10], 1e-5 );
        CHECK_CLOSE(-100.0, grid->vertices[11], 1e-5 );
        CHECK_CLOSE( -80.0, grid->vertices[12], 1e-5 );
        CHECK_CLOSE( -60.0, grid->vertices[13], 1e-5 );
        CHECK_CLOSE( -40.0, grid->vertices[14], 1e-5 );
        CHECK_CLOSE( -20.0, grid->vertices[15], 1e-5 );
        CHECK_CLOSE(   0.0, grid->vertices[16], 1e-5 );
        CHECK_CLOSE(  20.0, grid->vertices[17], 1e-5 );
        CHECK_CLOSE(  40.0, grid->vertices[18], 1e-5 );
        CHECK_CLOSE(  60.0, grid->vertices[19], 1e-5 );
        CHECK_CLOSE(  80.0, grid->vertices[20], 1e-5 );
        CHECK_CLOSE( 100.0, grid->vertices[21], 1e-5 );
        CHECK_CLOSE( -20.0, grid->vertices[22], 1e-5 );
        CHECK_CLOSE( -16.0, grid->vertices[23], 1e-5 );
        CHECK_CLOSE( -12.0, grid->vertices[24], 1e-5 );
        CHECK_CLOSE(  -8.0, grid->vertices[25], 1e-5 );
        CHECK_CLOSE(  -4.0, grid->vertices[26], 1e-5 );
        CHECK_CLOSE(   0.0, grid->vertices[27], 1e-5 );
        CHECK_CLOSE(   4.0, grid->vertices[28], 1e-5 );
        CHECK_CLOSE(   8.0, grid->vertices[29], 1e-5 );
        CHECK_CLOSE(  12.0, grid->vertices[30], 1e-5 );
        CHECK_CLOSE(  16.0, grid->vertices[31], 1e-5 );
        CHECK_CLOSE(  20.0, grid->vertices[32], 1e-5 );

        CHECK_EQUAL( 0, grid->offset[0]);
        CHECK_EQUAL( 11, grid->offset[1]);
        CHECK_EQUAL( 22, grid->offset[2]);

        grid->isRegular[0] = false;
        grid->isRegular[1] = false;
        grid->isRegular[2] = false;

        CHECK_EQUAL( -1, getDimIndex( grid, 0, -10.5) );
        CHECK_EQUAL( 0, getDimIndex( grid, 0, -9.5) );
        CHECK_EQUAL( 1, getDimIndex( grid, 0, -7.5) );
        CHECK_EQUAL( 8, getDimIndex( grid, 0,  7.5) );
        CHECK_EQUAL( 9, getDimIndex( grid, 0,  9.5) );
        CHECK_EQUAL( 10, getDimIndex( grid, 0,  10.5) );

        CHECK_EQUAL( -1, getDimIndex( grid, 1,-105.0) );
        CHECK_EQUAL(  0, getDimIndex( grid, 1, -95.0) );
        CHECK_EQUAL(  1, getDimIndex( grid, 1, -75.0) );
        CHECK_EQUAL(  8, getDimIndex( grid, 1,  75.0) );
        CHECK_EQUAL(  9, getDimIndex( grid, 1,  95.0) );
        CHECK_EQUAL( 10, getDimIndex( grid, 1, 105.0) );

    }

    TEST_FIXTURE(GridBinsTest, irregular_getDimIndex ) {
    	std::vector<double> vertices {-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
     			                        0,  1,  2,  3.1,  4,  5,  6,  7,  8,  9,  10};

        setVertices(grid, 0, vertices );

        CHECK_EQUAL( -1, getDimIndex( grid, 0, -10.5) );
        CHECK_EQUAL( 0, getDimIndex( grid, 0, -9.5) );
        CHECK_EQUAL( 1, getDimIndex( grid, 0, -8.5) );
        CHECK_EQUAL( 18, getDimIndex( grid, 0,  8.5) );
        CHECK_EQUAL( 19, getDimIndex( grid, 0,  9.5) );
        CHECK_EQUAL( 20, getDimIndex( grid, 0,  10.5) );
    }

    TEST_FIXTURE(GridBinsTest, pwr_vertices_getDimIndex ) {
        //     0      1       2        3      4       5        6       7       8      9
        std::vector<double> vertices  {
          -40.26, -20.26, -10.26, -10.08,  -9.90,  -9.84,  -9.06,  -9.00,  -8.64,  -8.58,
           -7.80,  -7.74,  -7.38,  -7.32,  -6.54,  -6.48,  -6.12,  -6.06,  -5.28,  -5.22,
           -4.86,  -4.80,  -4.02,  -3.96,  -3.60,  -3.54,  -2.76,  -2.70,  -2.34,  -2.28,
           -1.50,  -1.44,  -1.08,  -1.02,  -0.24,  -0.18,   0.18,   0.24,   1.02,   1.08,
            1.44,   1.50,   2.28,   2.34,   2.70,   2.76,   3.54,   3.60,   3.96,   4.02,
            4.80,   4.86,   5.22,   5.28,   6.06,   6.12,   6.48,   6.54,   7.32,   7.38,
            7.74,   7.80,   8.58,   8.64,   9.00,   9.06,   9.84,   9.90,  10.08,  10.26,
           20.26,  40.26};

        std::vector<double> zvertices { -80, -60, -50, 50, 60, 80 };
        setVertices(grid, 0, vertices );
        setVertices(grid, 1, vertices );
        setVertices(grid, 2, zvertices );

        CHECK_EQUAL( 72, vertices.size() );
        CHECK_EQUAL(  0, getDimIndex( grid, 0, -25)  );
        CHECK_EQUAL(  1, getDimIndex( grid, 0, -20)  );
        CHECK_EQUAL(  10, getDimIndex( grid, 0, -7.75)  );
        CHECK_EQUAL(  60, getDimIndex( grid, 0,  7.75)  );
        CHECK_EQUAL(  70, getDimIndex( grid, 0,  21.0)  );
        CHECK_EQUAL(  69, getDimIndex( grid, 0, 11.21220650 )  );
        CHECK_EQUAL(  51, getDimIndex( grid, 0,  5.200257 )  );

        CHECK_EQUAL( 2, getDimIndex( grid, 2, -16.975525) );

     }

	class GridBinsRayTraceTest{
	public:

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
		Vector3D<double> pos( -0.5, 0.5, 0.5 );
		Vector3D<double> dir( 1, 0, 0);
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
		Vector3D<double> pos( -0.5, 0.5, 0.5 );
		Vector3D<double> dir( 1, 0, 0);
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
		Vector3D<double> pos( 0.5, 0.5, 0.5 );
		Vector3D<double> dir( 1, 0, 0);
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

	TEST( gridbins_host_pwr_vertices_isRegular ) {
	        //     0      1       2        3      4       5        6       7       8      9
		std::vector<double> vertices  {
	          -40.26, -20.26, -10.26, -10.08,  -9.90,  -9.84,  -9.06,  -9.00,  -8.64,  -8.58,
	           -7.80,  -7.74,  -7.38,  -7.32,  -6.54,  -6.48,  -6.12,  -6.06,  -5.28,  -5.22,
	           -4.86,  -4.80,  -4.02,  -3.96,  -3.60,  -3.54,  -2.76,  -2.70,  -2.34,  -2.28,
	           -1.50,  -1.44,  -1.08,  -1.02,  -0.24,  -0.18,   0.18,   0.24,   1.02,   1.08,
	            1.44,   1.50,   2.28,   2.34,   2.70,   2.76,   3.54,   3.60,   3.96,   4.02,
	            4.80,   4.86,   5.22,   5.28,   6.06,   6.12,   6.48,   6.54,   7.32,   7.38,
	            7.74,   7.80,   8.58,   8.64,   9.00,   9.06,   9.84,   9.90,  10.08,  10.26,
	           20.26,  40.26};

		std::vector<double> zvertices { -80, -60, -50, 50, 60, 80 };

		GridBinsHost grid;
		grid.setVertices(0, vertices);
		grid.setVertices(1, vertices);
		grid.setVertices(2, zvertices);

		CHECK_EQUAL( false, isRegular(grid.getPtr(),0) );
		CHECK_EQUAL( false, isRegular(grid.getPtr(),1) );
		CHECK_EQUAL( false, isRegular(grid.getPtr(),2) );


	}

}
