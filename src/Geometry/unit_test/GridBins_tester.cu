#include <UnitTest++.h>

#include "GridBins.hh"
#include "HashBins.hh"
#include "GPUUtilityFunctions.hh"

using namespace MonteRay;

SUITE( GridBins_Tester ) {


    TEST( ctor_class_getMaxNumVertices ) {
        //gpuReset();

        GridBins grid;
        CHECK_EQUAL( 1001, grid.getMaxNumVertices() );
        CHECK_EQUAL( 0, grid.getOffset(0) );
        CHECK_EQUAL( 1001, grid.getOffset(1) );
        CHECK_EQUAL( 1001*2, grid.getOffset(2) );
    }

    TEST( class_setVertices ) {
        GridBins grid;
        grid.setVertices(0,  0.0, 10.0, 10);
        grid.setVertices(1, 20.0, 30.0, 10);
        grid.setVertices(2, 40.0, 50.0, 10);

        CHECK_EQUAL(11, grid.getNumVertices(0) );
        CHECK_EQUAL(10, grid.getNumBins(0) );

        CHECK_EQUAL(11, grid.getNumVertices(1) );
        CHECK_EQUAL(10, grid.getNumBins(1) );

        CHECK_EQUAL(11, grid.getNumVertices(2) );
        CHECK_EQUAL(10, grid.getNumBins(2) );

        CHECK_CLOSE( 0.0, grid.getVertex(0, 0), 1e-11 );
        CHECK_CLOSE( 1.0, grid.getVertex(0, 1), 1e-11 );
        CHECK_CLOSE( 10.0, grid.getVertex(0, 10), 1e-11 );

        CHECK_CLOSE( 20.0, grid.getVertex(1, 0), 1e-11 );
        CHECK_CLOSE( 21.0, grid.getVertex(1, 1), 1e-11 );
        CHECK_CLOSE( 30.0, grid.getVertex(1, 10), 1e-11 );

        CHECK_CLOSE( 40.0, grid.getVertex(2, 0), 1e-11 );
        CHECK_CLOSE( 41.0, grid.getVertex(2, 1), 1e-11 );
        CHECK_CLOSE( 50.0, grid.getVertex(2, 10), 1e-11 );

        CHECK_EQUAL( true, grid.isRegular(0) );
        CHECK_EQUAL( true, grid.isRegular(1) );
        CHECK_EQUAL( true, grid.isRegular(2) );
    }

    TEST( class_finalize ) {
        GridBins grid;
        grid.setVertices(0,  0.0, 10.0, 10);
        grid.setVertices(1, 20.0, 30.0, 10);
        grid.setVertices(2, 40.0, 50.0, 10);
        grid.finalize();

        CHECK_CLOSE( 0.0, grid.getVertex(0, 0), 1e-11 );
        CHECK_CLOSE( 1.0, grid.getVertex(0, 1), 1e-11 );
        CHECK_CLOSE( 10.0, grid.getVertex(0, 10), 1e-11 );

        CHECK_CLOSE( 20.0, grid.getVertex(1, 0), 1e-11 );
        CHECK_CLOSE( 21.0, grid.getVertex(1, 1), 1e-11 );
        CHECK_CLOSE( 30.0, grid.getVertex(1, 10), 1e-11 );

        CHECK_CLOSE( 40.0, grid.getVertex(2, 0), 1e-11 );
        CHECK_CLOSE( 41.0, grid.getVertex(2, 1), 1e-11 );
        CHECK_CLOSE( 50.0, grid.getVertex(2, 10), 1e-11 );

        CHECK_EQUAL( true, grid.isRegular(0) );
        CHECK_EQUAL( true, grid.isRegular(1) );
        CHECK_EQUAL( true, grid.isRegular(2) );
    }

    TEST( class_numXY ) {
        GridBins grid;
        grid.setVertices(0,  0.0, 10.0, 10);
        grid.setVertices(1, 20.0, 30.0, 10);
        grid.setVertices(2, 40.0, 50.0, 10);
        grid.finalize();

        CHECK_EQUAL( 100, grid.getNumXY()  );
    }

    TEST( class_min_max ) {
        GridBins grid;
        grid.setVertices(0,  0.0, 10.0, 10);
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

    TEST( class_getDimIndex ) {
        GridBins grid;
        grid.setVertices(0,  0.0, 10.0, 10);
        grid.setVertices(1,  0.0, 10.0, 10);
        grid.setVertices(2,  0.0, 10.0, 10);
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

    TEST( class_getIndex ) {
        GridBins grid;
        grid.setVertices(0,  0.0, 10.0, 10);
        grid.setVertices(1,  0.0, 10.0, 10);
        grid.setVertices(2,  0.0, 10.0, 10);
        grid.finalize();

        Vector3D<gpuRayFloat_t> pos1( 0.5, 0.5, 0.5);
        CHECK_EQUAL(  0, grid.getIndex(pos1 ) );

        Vector3D<gpuRayFloat_t> pos2( 0.5, 1.5, 0.5);
        CHECK_EQUAL(  10, grid.getIndex(pos2 ) );

        Vector3D<gpuRayFloat_t> pos3( 0.5, 0.5, 1.5);
        CHECK_EQUAL(  100, grid.getIndex( pos3 ) );
    }

    TEST( class_isOutside ) {
        GridBins grid;
        grid.setVertices(0,  0.0, 10.0, 10);
        grid.setVertices(1,  0.0, 10.0, 10);
        grid.setVertices(2,  0.0, 10.0, 10);
        grid.finalize();

        int indices1[3] = { 0, 0, 0};
        CHECK( grid.isOutside(indices1) == false );

        int indices2[3] = { -1, 0, 0};
        CHECK( grid.isOutside(indices2) == true );

        int indices3[3] = { 10, 0, 0};
        CHECK( grid.isOutside(indices3) == true );
    }

    TEST( class_isRegular ) {
        GridBins grid;

        std::vector<gpuFloatType_t> vertices {-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
            0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10};

        grid.setVertices(0, vertices );
        CHECK_EQUAL( false, grid.isRegular(0) );

        CHECK_EQUAL( 21, grid.getNumVertices(0) );
        CHECK_CLOSE( -10.0, grid.getVertex(0,0), 1e-6 );
        CHECK_CLOSE( -10.0, grid.vertices[0], 1e-6 );
    }

    TEST( class_regular_getDimIndex ) {
        GridBins grid;
        grid.setVertices(0, -10, 10, 20 );

        CHECK_EQUAL( -1, grid.getDimIndex(0, -10.5) );
        CHECK_EQUAL( 0, grid.getDimIndex( 0, -9.5) );
        CHECK_EQUAL( 1, grid.getDimIndex( 0, -8.5) );
        CHECK_EQUAL( 18, grid.getDimIndex( 0,  8.5) );
        CHECK_EQUAL( 19, grid.getDimIndex( 0,  9.5) );
        CHECK_EQUAL( 20, grid.getDimIndex( 0,  10.5) );
    }

    TEST( classregular_but_force_irregular_getDimIndex_1D ) {
        GridBins grid;

        grid.setVertices(0, -10, 10, 20 );
        grid.regular[0]=false;

        CHECK_EQUAL( -1, grid.getDimIndex(0, -10.5) );
        CHECK_EQUAL( 0, grid.getDimIndex( 0, -9.5) );
        CHECK_EQUAL( 1, grid.getDimIndex( 0, -8.5) );
        CHECK_EQUAL( 18, grid.getDimIndex( 0,  8.5) );
        CHECK_EQUAL( 19, grid.getDimIndex( 0,  9.5) );
        CHECK_EQUAL( 20, grid.getDimIndex( 0,  10.5) );
    }

    TEST(class_regular_but_force_irregular_getDimIndex_3D ) {
        GridBins grid;

        grid.setVertices( 0, -10, 10, 10 );
        grid.setVertices( 1, -100, 100, 10 );
        grid.setVertices( 2, -20, 20, 10 );
        grid.finalize();

        CHECK_CLOSE( -10.0, grid.vertices[0], 1e-5 );
        CHECK_CLOSE(  -8.0, grid.vertices[1], 1e-5 );
        CHECK_CLOSE(  -6.0, grid.vertices[2], 1e-5 );
        CHECK_CLOSE(  -4.0, grid.vertices[3], 1e-5 );
        CHECK_CLOSE(  -2.0, grid.vertices[4], 1e-5 );
        CHECK_CLOSE(   0.0, grid.vertices[5], 1e-5 );
        CHECK_CLOSE(   2.0, grid.vertices[6], 1e-5 );
        CHECK_CLOSE(   4.0, grid.vertices[7], 1e-5 );
        CHECK_CLOSE(   6.0, grid.vertices[8], 1e-5 );
        CHECK_CLOSE(   8.0, grid.vertices[9], 1e-5 );
        CHECK_CLOSE(  10.0, grid.vertices[10], 1e-5 );
        CHECK_CLOSE(-100.0, grid.vertices[11], 1e-5 );
        CHECK_CLOSE( -80.0, grid.vertices[12], 1e-5 );
        CHECK_CLOSE( -60.0, grid.vertices[13], 1e-5 );
        CHECK_CLOSE( -40.0, grid.vertices[14], 1e-5 );
        CHECK_CLOSE( -20.0, grid.vertices[15], 1e-5 );
        CHECK_CLOSE(   0.0, grid.vertices[16], 1e-5 );
        CHECK_CLOSE(  20.0, grid.vertices[17], 1e-5 );
        CHECK_CLOSE(  40.0, grid.vertices[18], 1e-5 );
        CHECK_CLOSE(  60.0, grid.vertices[19], 1e-5 );
        CHECK_CLOSE(  80.0, grid.vertices[20], 1e-5 );
        CHECK_CLOSE( 100.0, grid.vertices[21], 1e-5 );
        CHECK_CLOSE( -20.0, grid.vertices[22], 1e-5 );
        CHECK_CLOSE( -16.0, grid.vertices[23], 1e-5 );
        CHECK_CLOSE( -12.0, grid.vertices[24], 1e-5 );
        CHECK_CLOSE(  -8.0, grid.vertices[25], 1e-5 );
        CHECK_CLOSE(  -4.0, grid.vertices[26], 1e-5 );
        CHECK_CLOSE(   0.0, grid.vertices[27], 1e-5 );
        CHECK_CLOSE(   4.0, grid.vertices[28], 1e-5 );
        CHECK_CLOSE(   8.0, grid.vertices[29], 1e-5 );
        CHECK_CLOSE(  12.0, grid.vertices[30], 1e-5 );
        CHECK_CLOSE(  16.0, grid.vertices[31], 1e-5 );
        CHECK_CLOSE(  20.0, grid.vertices[32], 1e-5 );

        CHECK_EQUAL( 0, grid.offset[0]);
        CHECK_EQUAL( 11, grid.offset[1]);
        CHECK_EQUAL( 22, grid.offset[2]);

        grid.regular[0] = false;
        grid.regular[1] = false;
        grid.regular[2] = false;

        CHECK_EQUAL( -1, grid.getDimIndex( 0, -10.5) );
        CHECK_EQUAL( 0, grid.getDimIndex( 0, -9.5) );
        CHECK_EQUAL( 1, grid.getDimIndex( 0, -7.5) );
        CHECK_EQUAL( 8, grid.getDimIndex( 0,  7.5) );
        CHECK_EQUAL( 9, grid.getDimIndex( 0,  9.5) );
        CHECK_EQUAL( 10, grid.getDimIndex( 0,  10.5) );

        CHECK_EQUAL( -1, grid.getDimIndex( 1,-105.0) );
        CHECK_EQUAL(  0, grid.getDimIndex( 1, -95.0) );
        CHECK_EQUAL(  1, grid.getDimIndex( 1, -75.0) );
        CHECK_EQUAL(  8, grid.getDimIndex( 1,  75.0) );
        CHECK_EQUAL(  9, grid.getDimIndex( 1,  95.0) );
        CHECK_EQUAL( 10, grid.getDimIndex( 1, 105.0) );

    }

    TEST( class_irregular_getDimIndex ) {
        GridBins grid;

        std::vector<gpuFloatType_t> vertices {-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
            0,  1,  2,  3.1,  4,  5,  6,  7,  8,  9,  10};

        grid.setVertices(0, vertices );
        CHECK_EQUAL( false, grid.isRegular(0) );
        CHECK_CLOSE( -10.0, grid.min(0), 1e-6 );
        CHECK_CLOSE( 10.0, grid.max(0), 1e-6 );
        CHECK_EQUAL( 20, grid.getNumBins(0) );

        CHECK_CLOSE( -10.0, grid.getVertex(0,0), 1e-6 );

        CHECK_EQUAL( -1, grid.getDimIndex( 0, -10.5) );
        CHECK_EQUAL( 0, grid.getDimIndex( 0, -9.5) );
        CHECK_EQUAL( 1, grid.getDimIndex( 0, -8.5) );
        CHECK_EQUAL( 18, grid.getDimIndex( 0,  8.5) );
        CHECK_EQUAL( 19, grid.getDimIndex( 0,  9.5) );
        CHECK_EQUAL( 20, grid.getDimIndex( 0,  10.5) );
    }

    TEST(class_pwr_vertices_getDimIndex ) {
        GridBins grid;

        //     0      1       2        3      4       5        6       7       8      9
        std::vector<gpuFloatType_t> vertices  {
            -40.26, -20.26, -10.26, -10.08,  -9.90,  -9.84,  -9.06,  -9.00,  -8.64,  -8.58,
            -7.80,  -7.74,  -7.38,  -7.32,  -6.54,  -6.48,  -6.12,  -6.06,  -5.28,  -5.22,
            -4.86,  -4.80,  -4.02,  -3.96,  -3.60,  -3.54,  -2.76,  -2.70,  -2.34,  -2.28,
            -1.50,  -1.44,  -1.08,  -1.02,  -0.24,  -0.18,   0.18,   0.24,   1.02,   1.08,
            1.44,   1.50,   2.28,   2.34,   2.70,   2.76,   3.54,   3.60,   3.96,   4.02,
            4.80,   4.86,   5.22,   5.28,   6.06,   6.12,   6.48,   6.54,   7.32,   7.38,
            7.74,   7.80,   8.58,   8.64,   9.00,   9.06,   9.84,   9.90,  10.08,  10.26,
            20.26,  40.26};

        std::vector<gpuFloatType_t> zvertices { -80, -60, -50, 50, 60, 80 };
        grid.setVertices( 0, vertices );
        grid.setVertices( 1, vertices );
        grid.setVertices( 2, zvertices );

        CHECK_EQUAL( 72, vertices.size() );
        CHECK_EQUAL(  0, grid.getDimIndex( 0, -25)  );
        CHECK_EQUAL(  1, grid.getDimIndex( 0, -20)  );
        CHECK_EQUAL(  10, grid.getDimIndex( 0, -7.75)  );
        CHECK_EQUAL(  60, grid.getDimIndex( 0,  7.75)  );
        CHECK_EQUAL(  70, grid.getDimIndex( 0,  21.0)  );
        CHECK_EQUAL(  69, grid.getDimIndex( 0, 11.21220650 )  );
        CHECK_EQUAL(  51, grid.getDimIndex( 0,  5.200257 )  );

        CHECK_EQUAL( 2, grid.getDimIndex( 2, -16.975525) );

    }

    TEST( pwr_vertices_read_write ) {
            GridBins write_grid;

            //     0      1       2        3      4       5        6       7       8      9
            std::vector<gpuFloatType_t> vertices  {
                -40.26, -20.26, -10.26, -10.08,  -9.90,  -9.84,  -9.06,  -9.00,  -8.64,  -8.58,
                -7.80,  -7.74,  -7.38,  -7.32,  -6.54,  -6.48,  -6.12,  -6.06,  -5.28,  -5.22,
                -4.86,  -4.80,  -4.02,  -3.96,  -3.60,  -3.54,  -2.76,  -2.70,  -2.34,  -2.28,
                -1.50,  -1.44,  -1.08,  -1.02,  -0.24,  -0.18,   0.18,   0.24,   1.02,   1.08,
                1.44,   1.50,   2.28,   2.34,   2.70,   2.76,   3.54,   3.60,   3.96,   4.02,
                4.80,   4.86,   5.22,   5.28,   6.06,   6.12,   6.48,   6.54,   7.32,   7.38,
                7.74,   7.80,   8.58,   8.64,   9.00,   9.06,   9.84,   9.90,  10.08,  10.26,
                20.26,  40.26};

            std::vector<gpuFloatType_t> zvertices { -80, -60, -50, 50, 60, 80 };
            write_grid.setVertices( 0, vertices );
            write_grid.setVertices( 1, vertices );
            write_grid.setVertices( 2, zvertices );

            write_grid.writeToFile( "Gridbins_write_test1.bin");

            // test file exists
            std::ifstream exists("Gridbins_write_test1.bin");
            CHECK_EQUAL( true, exists.good() );
            exists.close();

            GridBins grid;
            grid.readFromFile( "Gridbins_write_test1.bin");
            CHECK_EQUAL(  0, grid.getDimIndex( 0, -25)  );
            CHECK_EQUAL(  1, grid.getDimIndex( 0, -20)  );
            CHECK_EQUAL(  10, grid.getDimIndex( 0, -7.75)  );
            CHECK_EQUAL(  60, grid.getDimIndex( 0,  7.75)  );
            CHECK_EQUAL(  70, grid.getDimIndex( 0,  21.0)  );
            CHECK_EQUAL(  69, grid.getDimIndex( 0, 11.21220650 )  );
            CHECK_EQUAL(  51, grid.getDimIndex( 0,  5.200257 )  );

            CHECK_EQUAL( 2, grid.getDimIndex( 2, -16.975525) );

        }

    class ClassGridBinsRayTraceTest{
    public:

        ClassGridBinsRayTraceTest(){
            grid.setVertices( 0, 0.0, 10.0, 10);
            grid.setVertices( 1, 0.0, 10.0, 10);
            grid.setVertices( 2, 0.0, 10.0, 10);
            grid.finalize();
        }

        ~ClassGridBinsRayTraceTest(){}

        GridBins grid;
    };

    TEST_FIXTURE(ClassGridBinsRayTraceTest, getNumCells)
    {
        CHECK_EQUAL( 1000, grid.getNumCells() );
    }

    TEST_FIXTURE(ClassGridBinsRayTraceTest, getCenterPoint)
    {
        Position_t pos = grid.getCenterPointByIndex(0 );
        CHECK_CLOSE( 0.5, pos[0], 1e-11 );
        CHECK_CLOSE( 0.5, pos[1], 1e-11 );
        CHECK_CLOSE( 0.5, pos[2], 1e-11 );
    }

    TEST_FIXTURE(ClassGridBinsRayTraceTest, Distance1)
    {
        Position_t pos = grid.getCenterPointByIndex( 0 );

        Position_t start( 1.5, 0.5, 0.5);

        float_t distance = getDistance( pos, start );
        CHECK_CLOSE( 1.0, distance, 1e-11);
    }


    TEST_FIXTURE(ClassGridBinsRayTraceTest, Distance2)
    {
        Position_t pos = grid.getCenterPointByIndex( 0 );

        Position_t start( 1.5, 0.5, 1.5);

        float_t distance = getDistance( pos, start );
        CHECK_CLOSE( 1.0f*sqrt(2.0f), distance, 1e-7);
    }


    TEST_FIXTURE(ClassGridBinsRayTraceTest, getCenterPointbyIndices)
    {
        Position_t pos;
        unsigned indices[3];
        indices[0] = 0; indices[1] = 0; indices[2] = 0;
        pos = grid.getCenterPointByIndices( indices );
        CHECK_CLOSE( 0.5, pos[0], 1e-11 );
        CHECK_CLOSE( 0.5, pos[1], 1e-11 );
        CHECK_CLOSE( 0.5, pos[2], 1e-11 );
    }

    TEST_FIXTURE(ClassGridBinsRayTraceTest, crossingInside_to_outside_posDir_one_crossing)
    {
        Position_t pos( -0.5, 0.5, 0.5 );
        Direction_t dir( 1, 0, 0);
        float_t distance = 1.0;

        int cells[1000];
        gpuRayFloat_t distances[1000];

        unsigned nDistances = grid.rayTrace(cells, distances, pos, dir, distance, false );

        CHECK_EQUAL( 1, nDistances);
        CHECK_EQUAL( 0U, cells[0]);
        CHECK_CLOSE( 0.5f, distances[0], 1e-11 );
    }

    TEST_FIXTURE(ClassGridBinsRayTraceTest, crossingInside_to_outside_posDir_two_crossings)
    {
        Position_t pos( -0.5, 0.5, 0.5 );
        Direction_t dir( 1, 0, 0);
        float_t distance = 2.0;

        int cells[1000];
        gpuRayFloat_t distances[1000];

        unsigned nDistances = grid.rayTrace(cells, distances, pos, dir, distance, false );

        CHECK_EQUAL( 2, nDistances);
        CHECK_EQUAL( 0U, cells[0]);
        CHECK_CLOSE( 1.0f, distances[0], 1e-11 );
        CHECK_EQUAL( 1U, cells[1]);
        CHECK_CLOSE( 0.5f, distances[1], 1e-11 );
    }

    class ClassGridBinsRayTraceTest2{
    public:

        ClassGridBinsRayTraceTest2(){
            grid.setVertices(0, -5.0, 5.0, 10);
            grid.setVertices(1, -5.0, 5.0, 10);
            grid.setVertices(2, -5.0, 5.0, 10);
            grid.finalize();
        }

        ~ClassGridBinsRayTraceTest2(){}

        GridBins grid;
    };

    TEST_FIXTURE(ClassGridBinsRayTraceTest2, Distance1)
    {
        Position_t pos( 0.5, 0.5, 0.5 );
        Direction_t dir( 1, 0, 0);
        float_t distance = 1.0;

        int cells[1000];
        gpuRayFloat_t distances[1000];

        unsigned nDistances = grid.rayTrace( cells, distances, pos, dir, distance, false );

        unsigned i = grid.getIndex( pos );
        CHECK_EQUAL( 555, i);

        CHECK_EQUAL( 2, nDistances);
        CHECK_EQUAL( 555U, cells[0]);
        CHECK_CLOSE( 0.5f, distances[0], 1e-11 );
    }

    TEST( gridbins_pwr_vertices_isRegular ) {
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

        GridBins grid;
        grid.setVertices(0, vertices);
        grid.setVertices(1, vertices);
        grid.setVertices(2, zvertices);

        CHECK_EQUAL( false, grid.isRegular(0) );
        CHECK_EQUAL( false, grid.isRegular(1) );
        CHECK_EQUAL( false, grid.isRegular(2) );
    }

    TEST( getHashPtr ) {
        GridBins grid;
        grid.setVertices(0,  0.0, 10.0, 10);

        //CHECK( false );
        CHECK_EQUAL( 8000, grid.getHashPtr(0)->getNEdges() );
        CHECK_CLOSE( 0.0, grid.getHashPtr(0)->getMin(), 1e-5 );
        CHECK_CLOSE( 10.0, grid.getHashPtr(0)->getMax(), 1e-5 );
    }

    TEST( getHasLowerUpperBins ) {
        GridBins grid;
        grid.setDefaultHashSize( 12 );

        std::vector<gpuFloatType_t> vertices  { -1.0, -0.5, 0.1, 0.2, 3.0, 6.0, 6.1, 6.15, 8.0, 10.0 };

        grid.setVertices(0, vertices);

        //CHECK( false );

        CHECK_EQUAL( 12, grid.getDefaultHashSize() );
        CHECK_EQUAL( 12, grid.getHashPtr(0)->getNEdges() );
        CHECK_CLOSE( -1.0, grid.getHashPtr(0)->getMin(), 1e-5 );
        CHECK_CLOSE( 10.0, grid.getHashPtr(0)->getMax(), 1e-5 );

        unsigned lower_bin;
        unsigned upper_bin;

        grid.getHashLowerUpperBins(0, -0.5, lower_bin, upper_bin);
        CHECK_EQUAL( 0, lower_bin);
        CHECK_EQUAL( 1, upper_bin );

        grid.getHashLowerUpperBins(0, 0.0, lower_bin, upper_bin);
        CHECK_EQUAL( 1, lower_bin);
        CHECK_EQUAL( 3, upper_bin );

        grid.getHashLowerUpperBins(0, 1.0, lower_bin, upper_bin);
        CHECK_EQUAL( 3, lower_bin);
        CHECK_EQUAL( 3, upper_bin );

        grid.getHashLowerUpperBins(0, 9.9, lower_bin, upper_bin);
        CHECK_EQUAL( 8, lower_bin);
        CHECK_EQUAL( 9, upper_bin );
    }

    TEST( hash_getHashLowerUpperBins_and_getDimIndex ) {
        GridBins grid;
        grid.setDefaultHashSize( 201 );

        std::vector<gpuFloatType_t> vertices {-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
            0,  1,  2,  3.1,  4,  5,  6,  7,  8,  9,  10};

        grid.setVertices(0, vertices );

        unsigned lower_bin;
        unsigned upper_bin;

        gpuFloatType_t pos = -10.5;
        CHECK_EQUAL( -1, grid.getDimIndex( 0, pos) );

        pos = -9.5;
        grid.getHashLowerUpperBins(0, pos, lower_bin, upper_bin);
        CHECK_EQUAL( 0, grid.getDimIndex( 0, pos) );
        CHECK_EQUAL( 0, lower_bin);
        CHECK_EQUAL( 0, upper_bin );

        pos = -8.5;
        grid.getHashLowerUpperBins(0, pos, lower_bin, upper_bin);
        CHECK_EQUAL( 1, grid.getDimIndex( 0, pos) );
        CHECK_EQUAL( 1, lower_bin);
        CHECK_EQUAL( 1, upper_bin );

        pos = 8.5;
        grid.getHashLowerUpperBins(0, pos, lower_bin, upper_bin);
        CHECK_EQUAL( 18, grid.getDimIndex( 0, pos) );
        CHECK_EQUAL( 18, lower_bin);
        CHECK_EQUAL( 18, upper_bin );

        pos = 9.5;
        grid.getHashLowerUpperBins(0, pos, lower_bin, upper_bin);
        CHECK_EQUAL( 19, grid.getDimIndex( 0, pos) );
        CHECK_EQUAL( 19, lower_bin);
        CHECK_EQUAL( 19, upper_bin );

        pos = 10.5;
        CHECK_EQUAL( 20, grid.getDimIndex( 0, pos) );
    }

}
