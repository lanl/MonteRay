#include <UnitTest++.h>

#include "MonteRay_SpatialGrid_GPU_helper.hh"

namespace MonteRay_CartesianGrid_crossingDistance_GPU_tests{

using namespace MonteRay;
using namespace MonteRay_SpatialGrid_helper;

SUITE( MonteRay_CartesianGrid_crossingDistance_GPU_Tests) {

    typedef singleDimRayTraceMap_t distances_t;
    typedef singleDimRayTraceMap_t rayTraceMap_t;
    TEST_FIXTURE(SpatialGridGPUTester, CrossingDistance_in_1D_PosXDir ) {
		std::vector<gpuRayFloat_t> vertices= {
		        	-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
		              0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10};

        setCoordinateSystem( TransportMeshTypeEnum::Cartesian );
		setDimension( 3 );
        setGrid( MonteRay_SpatialGrid::CART_X, vertices);
        setGrid( MonteRay_SpatialGrid::CART_Y, vertices);
        setGrid( MonteRay_SpatialGrid::CART_Z, vertices);
        initialize();
        copyToGPU();

        Position_t position ( -9.5, 0.5,  0.5 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 1.0;

        distances_t distances = crossingDistance( 0, position[0], direction[0], distance);

        CHECK_EQUAL( 2, distances.size() );
        CHECK_EQUAL( 0, distances.id(0) );
        CHECK_CLOSE( 0.5, distances.dist(0), 1e-11 );
        CHECK_EQUAL( 1, distances.id(1) );
        CHECK_CLOSE( 1.0, distances.dist(1), 1e-11 );
    }


    TEST_FIXTURE(SpatialGridGPUTester, CrossingDistance_in_1D_NegXDir ) {
		std::vector<gpuRayFloat_t> vertices= {
		        	-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
		              0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10};

        setCoordinateSystem( TransportMeshTypeEnum::Cartesian );
		setDimension( 3 );
        setGrid( MonteRay_SpatialGrid::CART_X, vertices);
        setGrid( MonteRay_SpatialGrid::CART_Y, vertices);
        setGrid( MonteRay_SpatialGrid::CART_Z, vertices);
        initialize();
        copyToGPU();

        Position_t position ( -8.5, 0.5,  0.5 );
        Position_t direction(    -1,   0,    0 );
        gpuRayFloat_t distance = 1.0;

        distances_t distances = crossingDistance( 0, position[0], direction[0], distance);

        CHECK_EQUAL( 2, distances.size() );
        CHECK_EQUAL( 1, distances.id(0) );
        CHECK_CLOSE( 0.5, distances.dist(0), 1e-11 );
        CHECK_EQUAL( 0, distances.id(1) );
        CHECK_CLOSE( 1.0, distances.dist(1), 1e-11 );
    }

    TEST_FIXTURE(SpatialGridGPUTester, Outside_negSide_negDir ) {
		std::vector<gpuRayFloat_t> vertices= {
		        	-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
		              0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10};

        setCoordinateSystem( TransportMeshTypeEnum::Cartesian );
		setDimension( 3 );
        setGrid( MonteRay_SpatialGrid::CART_X, vertices);
        setGrid( MonteRay_SpatialGrid::CART_Y, vertices);
        setGrid( MonteRay_SpatialGrid::CART_Z, vertices);
        initialize();
        copyToGPU();

        Position_t position ( -10.5, 0.5,  0.5 );
        Position_t direction(    -1,   0,    0 );
        gpuRayFloat_t distance = 2.0;

        distances_t distances = crossingDistance( 0, position[0], direction[0], distance);

        CHECK_EQUAL(  0, distances.size() );
    }

    TEST_FIXTURE(SpatialGridGPUTester, Outside_posSide_posDir ) {
		std::vector<gpuRayFloat_t> vertices= {
		        	-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
		              0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10};

        setCoordinateSystem( TransportMeshTypeEnum::Cartesian );
		setDimension( 3 );
        setGrid( MonteRay_SpatialGrid::CART_X, vertices);
        setGrid( MonteRay_SpatialGrid::CART_Y, vertices);
        setGrid( MonteRay_SpatialGrid::CART_Z, vertices);
        initialize();
        copyToGPU();

        Position_t position (  10.5, 0.5,  0.5 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 2.0;

        distances_t distances = crossingDistance( 0, position[0], direction[0], distance);

        CHECK_EQUAL(  0, distances.size() );
    }

    TEST_FIXTURE(SpatialGridGPUTester, Outside_negSide_posDir ) {
		std::vector<gpuRayFloat_t> vertices= {
		        	-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
		              0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10};

        setCoordinateSystem( TransportMeshTypeEnum::Cartesian );
		setDimension( 3 );
        setGrid( MonteRay_SpatialGrid::CART_X, vertices);
        setGrid( MonteRay_SpatialGrid::CART_Y, vertices);
        setGrid( MonteRay_SpatialGrid::CART_Z, vertices);
        initialize();
        copyToGPU();

        Position_t position ( -10.5, 0.5,  0.5 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 2.0;

        distances_t distances = crossingDistance( 0, position[0], direction[0], distance);

        CHECK_EQUAL( 3, distances.size() );
        CHECK_EQUAL( -1, distances.id(0) );
        CHECK_CLOSE( 0.5, distances.dist(0), 1e-11 );
        CHECK_EQUAL( 0, distances.id(1) );
        CHECK_CLOSE( 1.5, distances.dist(1), 1e-11 );
        CHECK_EQUAL( 1, distances.id(2) );
        CHECK_CLOSE( 2.0, distances.dist(2), 1e-11 );
    }

    TEST_FIXTURE(SpatialGridGPUTester, Outside_posSide_negDir ) {
		std::vector<gpuRayFloat_t> vertices= {
		        	-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
		              0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10};

        setCoordinateSystem( TransportMeshTypeEnum::Cartesian );
		setDimension( 3 );
        setGrid( MonteRay_SpatialGrid::CART_X, vertices);
        setGrid( MonteRay_SpatialGrid::CART_Y, vertices);
        setGrid( MonteRay_SpatialGrid::CART_Z, vertices);
        initialize();
        copyToGPU();

        Position_t position (  10.5, 0.5,  0.5 );
        Position_t direction(    -1,   0,    0 );
        gpuRayFloat_t distance = 2.0;

        distances_t distances = crossingDistance( 0, position[0], direction[0], distance);

        CHECK_EQUAL( 3, distances.size() );
        CHECK_EQUAL( 20, distances.id(0) );
        CHECK_CLOSE( 0.5, distances.dist(0), 1e-11 );
        CHECK_EQUAL( 19, distances.id(1) );
        CHECK_CLOSE( 1.5, distances.dist(1), 1e-11 );
        CHECK_EQUAL( 18, distances.id(2) );
        CHECK_CLOSE( 2.0, distances.dist(2), 1e-11 );
    }

    TEST_FIXTURE(SpatialGridGPUTester, Crossing_entire_grid_starting_outside_finish_outside_pos_dir ) {
		std::vector<gpuRayFloat_t> vertices= {
		        	-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
		              0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10};

        setCoordinateSystem( TransportMeshTypeEnum::Cartesian );
		setDimension( 3 );
        setGrid( MonteRay_SpatialGrid::CART_X, vertices);
        setGrid( MonteRay_SpatialGrid::CART_Y, vertices);
        setGrid( MonteRay_SpatialGrid::CART_Z, vertices);
        initialize();
        copyToGPU();

        Position_t position (  -10.5, 0.5,  0.5 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 21.0;

        distances_t distances = crossingDistance( 0, position[0], direction[0], distance);

        CHECK_EQUAL( 22, distances.size() );
        CHECK_EQUAL( -1, distances.id(0) );
        CHECK_CLOSE( 0.5, distances.dist(0), 1e-11 );
        CHECK_EQUAL( 0, distances.id(1) );
        CHECK_CLOSE( 1.5, distances.dist(1), 1e-11 );
        CHECK_EQUAL( 1, distances.id(2) );
        CHECK_CLOSE( 2.5, distances.dist(2), 1e-11 );
        CHECK_EQUAL( 17, distances.id(18) );
        CHECK_CLOSE( 18.5, distances.dist(18), 1e-11 );
        CHECK_EQUAL( 18, distances.id(19) );
        CHECK_CLOSE( 19.5, distances.dist(19), 1e-11 );
        CHECK_EQUAL( 19, distances.id(20) );
        CHECK_CLOSE( 20.5, distances.dist(20), 1e-11 );
        CHECK_EQUAL( 20, distances.id(21) );
        CHECK_CLOSE( 21.0, distances.dist(21), 1e-11 );

    }

    TEST_FIXTURE(SpatialGridGPUTester, Crossing_entire_grid_starting_outside_finish_outside_neg_dir ) {
		std::vector<gpuRayFloat_t> vertices= {
		        	-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
		              0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10};

        setCoordinateSystem( TransportMeshTypeEnum::Cartesian );
		setDimension( 3 );
        setGrid( MonteRay_SpatialGrid::CART_X, vertices);
        setGrid( MonteRay_SpatialGrid::CART_Y, vertices);
        setGrid( MonteRay_SpatialGrid::CART_Z, vertices);
        initialize();
        copyToGPU();

        Position_t position (  10.5, 0.5,  0.5 );
        Position_t direction(   -1,   0,    0 );
        gpuRayFloat_t distance = 21.0;

        distances_t distances = crossingDistance( 0, position[0], direction[0], distance);

        CHECK_EQUAL( 22, distances.size() );
        CHECK_EQUAL( 20, distances.id(0) );
        CHECK_CLOSE( 0.5, distances.dist(0), 1e-11 );
        CHECK_EQUAL( 19, distances.id(1) );
        CHECK_CLOSE( 1.5, distances.dist(1), 1e-11 );
        CHECK_EQUAL( 18, distances.id(2) );
        CHECK_CLOSE( 2.5, distances.dist(2), 1e-11 );
        CHECK_EQUAL( 2, distances.id(18) );
        CHECK_CLOSE( 18.5, distances.dist(18), 1e-11 );
        CHECK_EQUAL( 1, distances.id(19) );
        CHECK_CLOSE( 19.5, distances.dist(19), 1e-11 );
        CHECK_EQUAL( 0, distances.id(20) );
        CHECK_CLOSE( 20.5, distances.dist(20), 1e-11 );
        CHECK_EQUAL( -1, distances.id(21) );
        CHECK_CLOSE( 21.0, distances.dist(21), 1e-11 );
    }

    TEST_FIXTURE(SpatialGridGPUTester, Inside_cross_out_negDir ) {
		std::vector<gpuRayFloat_t> vertices= {
		        	-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
		              0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10};

        setCoordinateSystem( TransportMeshTypeEnum::Cartesian );
		setDimension( 3 );
        setGrid( MonteRay_SpatialGrid::CART_X, vertices);
        setGrid( MonteRay_SpatialGrid::CART_Y, vertices);
        setGrid( MonteRay_SpatialGrid::CART_Z, vertices);
        initialize();
        copyToGPU();

        Position_t position (  -8.5, 0.5,  0.5 );
        Position_t direction(    -1,   0,    0 );
        gpuRayFloat_t distance = 2.0;

        distances_t distances = crossingDistance( 0, position[0], direction[0], distance);

        CHECK_EQUAL( 3, distances.size() );
        CHECK_EQUAL( 1, distances.id(0) );
        CHECK_CLOSE( 0.5, distances.dist(0), 1e-11 );
        CHECK_EQUAL( 0, distances.id(1) );
        CHECK_CLOSE( 1.5, distances.dist(1), 1e-11 );
        CHECK_EQUAL( -1, distances.id(2) );
        CHECK_CLOSE( 2.0, distances.dist(2), 1e-11 );
    }

    TEST_FIXTURE(SpatialGridGPUTester, Inside_cross_out_posDir ) {
		std::vector<gpuRayFloat_t> vertices= {
		        	-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
		              0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10};

        setCoordinateSystem( TransportMeshTypeEnum::Cartesian );
		setDimension( 3 );
        setGrid( MonteRay_SpatialGrid::CART_X, vertices);
        setGrid( MonteRay_SpatialGrid::CART_Y, vertices);
        setGrid( MonteRay_SpatialGrid::CART_Z, vertices);
        initialize();
        copyToGPU();

        Position_t position (  8.5, 0.5,  0.5 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 2.0;

        distances_t distances = crossingDistance( 0, position[0], direction[0], distance);

        CHECK_EQUAL( 3, distances.size() );
        CHECK_EQUAL( 18, distances.id(0) );
        CHECK_CLOSE( 0.5, distances.dist(0), 1e-11 );
        CHECK_EQUAL( 19, distances.id(1) );
        CHECK_CLOSE( 1.5, distances.dist(1), 1e-11 );
        CHECK_EQUAL( 20, distances.id(2) );
        CHECK_CLOSE( 2.0, distances.dist(2), 1e-11 );
    }

    TEST_FIXTURE(SpatialGridGPUTester, crossingDistance_2D_internal_hit_corner_posXDir_posYDir ) {
        setCoordinateSystem( TransportMeshTypeEnum::Cartesian );
		setDimension( 3 );
        setGrid( MonteRay_SpatialGrid::CART_X, -1, 1, 2);
        setGrid( MonteRay_SpatialGrid::CART_Y, -1, 1, 2);
        setGrid( MonteRay_SpatialGrid::CART_Z, -1, 1, 2);
        initialize();
        copyToGPU();

        Position_t position (  -.5, -.5, -.5 );
        Position_t direction(  1.0,  1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 1.0*std::sqrt(2.0);

        unsigned dim = 0;
        distances_t distances = crossingDistance( dim, position[dim], direction[dim], distance);

        CHECK_EQUAL( 2, distances.size() );
        CHECK_EQUAL( 0, distances.id(0) );
        CHECK_CLOSE( (0.5)*std::sqrt(2.0), distances.dist(0), 1e-6 );
        CHECK_EQUAL( 1, distances.id(1) );
        CHECK_CLOSE( (1.0)*std::sqrt(2.0), distances.dist(1), 1e-6 );

        dim = 1;
        distances.clear();
        distances = crossingDistance( dim, position[dim], direction[dim], distance);

        CHECK_EQUAL( 2, distances.size() );
        CHECK_EQUAL( 0, distances.id(0) );
        CHECK_CLOSE( (0.5)*std::sqrt(2.0), distances.dist(0), 1e-6 );
        CHECK_EQUAL( 1, distances.id(1) );
        CHECK_CLOSE( (1.0)*std::sqrt(2.0), distances.dist(1), 1e-6 );

    }

    TEST_FIXTURE(SpatialGridGPUTester, crossingDistance_2D_start_on_an_external_corner_posX_posY ) {
        setCoordinateSystem( TransportMeshTypeEnum::Cartesian );
		setDimension( 3 );
        setGrid( MonteRay_SpatialGrid::CART_X, 0, 3, 3);
        setGrid( MonteRay_SpatialGrid::CART_Y, 0, 3, 3);
        setGrid( MonteRay_SpatialGrid::CART_Z, 0, 3, 3);
        initialize();
        copyToGPU();

        Position_t position (  0.0, 0.0, 0.5 );
        Position_t direction(  1.0,  1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        unsigned dim = 0;
        distances_t distances = crossingDistance( dim, position[dim], direction[dim], distance);

        CHECK_EQUAL( 5, distances.size() );
        CHECK_EQUAL( -1, distances.id(0) );
        CHECK_CLOSE( (0.0)*std::sqrt(2.0), distances.dist(0), 1e-6 );
        CHECK_EQUAL( 0, distances.id(1) );
        CHECK_CLOSE( (1.0)*std::sqrt(2.0), distances.dist(1), 1e-6 );
        CHECK_EQUAL( 1, distances.id(2) );
        CHECK_CLOSE( (2.0)*std::sqrt(2.0), distances.dist(2), 1e-6 );
        CHECK_EQUAL( 2, distances.id(3) );
        CHECK_CLOSE( (3.0)*std::sqrt(2.0), distances.dist(3), 1e-6 );
        CHECK_EQUAL( 3, distances.id(4) );
        CHECK_CLOSE( 10.0, distances.dist(4), 1e-6 );

        dim = 1;
        distances.clear();
        distances = crossingDistance( dim, position[dim], direction[dim], distance);

        CHECK_EQUAL( 5, distances.size() );
        CHECK_EQUAL( -1, distances.id(0) );
        CHECK_CLOSE( (0.0)*std::sqrt(2.0), distances.dist(0), 1e-6 );
        CHECK_EQUAL( 0, distances.id(1) );
        CHECK_CLOSE( (1.0)*std::sqrt(2.0), distances.dist(1), 1e-6 );
        CHECK_EQUAL( 1, distances.id(2) );
        CHECK_CLOSE( (2.0)*std::sqrt(2.0), distances.dist(2), 1e-6 );
        CHECK_EQUAL( 2, distances.id(3) );
        CHECK_CLOSE( (3.0)*std::sqrt(2.0), distances.dist(3), 1e-6 );
        CHECK_EQUAL( 3, distances.id(4) );
        CHECK_CLOSE( 10.0, distances.dist(4), 1e-6 );
    }

    TEST_FIXTURE(SpatialGridGPUTester, crossingDistance_2D_start_on_an_external_corner_negX_negY ) {
    	setCoordinateSystem( TransportMeshTypeEnum::Cartesian );
    	setDimension( 3 );
    	setGrid( MonteRay_SpatialGrid::CART_X, 0, 3, 3);
    	setGrid( MonteRay_SpatialGrid::CART_Y, 0, 3, 3);
    	setGrid( MonteRay_SpatialGrid::CART_Z, 0, 3, 3);
    	initialize();
    	copyToGPU();

    	Position_t position (  3.0,  3.0, 0.5 );
    	Position_t direction( -1.0, -1.0, 0.0 );
    	direction.normalize();
    	gpuRayFloat_t distance = 10.0;

    	unsigned dim = 0;
    	distances_t distances = crossingDistance( dim, position[dim], direction[dim], distance);

        CHECK_EQUAL( 5, distances.size() );
        CHECK_EQUAL( 3, distances.id(0) );
        CHECK_CLOSE( (0.0)*std::sqrt(2.0), distances.dist(0), 1e-6 );
        CHECK_EQUAL( 2, distances.id(1) );
        CHECK_CLOSE( (1.0)*std::sqrt(2.0), distances.dist(1), 1e-6 );
        CHECK_EQUAL( 1, distances.id(2) );
        CHECK_CLOSE( (2.0)*std::sqrt(2.0), distances.dist(2), 1e-6 );
        CHECK_EQUAL( 0, distances.id(3) );
        CHECK_CLOSE( (3.0)*std::sqrt(2.0), distances.dist(3), 1e-6 );
        CHECK_EQUAL( -1, distances.id(4) );
        CHECK_CLOSE( 10.0, distances.dist(4), 1e-6 );

        dim = 1;
        distances.clear();
        distances = crossingDistance( dim, position[dim], direction[dim], distance);

        CHECK_EQUAL( 5, distances.size() );
        CHECK_EQUAL( 3, distances.id(0) );
        CHECK_CLOSE( (0.0)*std::sqrt(2.0), distances.dist(0), 1e-6 );
        CHECK_EQUAL( 2, distances.id(1) );
        CHECK_CLOSE( (1.0)*std::sqrt(2.0), distances.dist(1), 1e-6 );
        CHECK_EQUAL( 1, distances.id(2) );
        CHECK_CLOSE( (2.0)*std::sqrt(2.0), distances.dist(2), 1e-6 );
        CHECK_EQUAL( 0, distances.id(3) );
        CHECK_CLOSE( (3.0)*std::sqrt(2.0), distances.dist(3), 1e-6 );
        CHECK_EQUAL( -1, distances.id(4) );
        CHECK_CLOSE( 10.0, distances.dist(4), 1e-6 );
    }

    TEST_FIXTURE(SpatialGridGPUTester, crossingDistance_2D_start_outside_on_an_external_corner_posX_posY ) {
    	setCoordinateSystem( TransportMeshTypeEnum::Cartesian );
    	setDimension( 3 );
    	setGrid( MonteRay_SpatialGrid::CART_X, 0, 3, 3);
    	setGrid( MonteRay_SpatialGrid::CART_Y, 0, 3, 3);
    	setGrid( MonteRay_SpatialGrid::CART_Z, 0, 3, 3);
    	initialize();
    	copyToGPU();

    	Position_t position ( -1.0, -1.0, 0.5 );
    	Position_t direction(  1.0,  1.0, 0.0 );
    	direction.normalize();
    	gpuRayFloat_t distance = 10.0;

    	unsigned dim = 0;
    	distances_t distances = crossingDistance( dim, position[dim], direction[dim], distance);

    	CHECK_EQUAL( 5, distances.size() );
    	CHECK_EQUAL( -1, distances.id(0) );
    	CHECK_CLOSE( (1.0)*std::sqrt(2.0), distances.dist(0), 1e-6 );
    	CHECK_EQUAL( 0, distances.id(1) );
    	CHECK_CLOSE( (2.0)*std::sqrt(2.0), distances.dist(1), 1e-6 );
    	CHECK_EQUAL( 1, distances.id(2) );
    	CHECK_CLOSE( (3.0)*std::sqrt(2.0), distances.dist(2), 1e-6 );
    	CHECK_EQUAL( 2, distances.id(3) );
    	CHECK_CLOSE( (4.0)*std::sqrt(2.0), distances.dist(3), 1e-6 );
    	CHECK_EQUAL( 3, distances.id(4) );
    	CHECK_CLOSE( 10.0, distances.dist(4), 1e-6 );


    	dim = 1;
    	distances.clear();
    	distances = crossingDistance( dim, position[dim], direction[dim], distance);

    	CHECK_EQUAL( 5, distances.size() );
    	CHECK_EQUAL( -1, distances.id(0) );
    	CHECK_CLOSE( (1.0)*std::sqrt(2.0), distances.dist(0), 1e-6 );
    	CHECK_EQUAL( 0, distances.id(1) );
    	CHECK_CLOSE( (2.0)*std::sqrt(2.0), distances.dist(1), 1e-6 );
    	CHECK_EQUAL( 1, distances.id(2) );
    	CHECK_CLOSE( (3.0)*std::sqrt(2.0), distances.dist(2), 1e-6 );
    	CHECK_EQUAL( 2, distances.id(3) );
    	CHECK_CLOSE( (4.0)*std::sqrt(2.0), distances.dist(3), 1e-6 );
    	CHECK_EQUAL( 3, distances.id(4) );
    	CHECK_CLOSE( 10.0, distances.dist(4), 1e-6 );
     }

    TEST_FIXTURE(SpatialGridGPUTester, crossingDistance_2D_start_outside_an_external_corner_negX_negY ) {
    	setCoordinateSystem( TransportMeshTypeEnum::Cartesian );
    	setDimension( 3 );
    	setGrid( MonteRay_SpatialGrid::CART_X, 0, 3, 3);
    	setGrid( MonteRay_SpatialGrid::CART_Y, 0, 3, 3);
    	setGrid( MonteRay_SpatialGrid::CART_Z, 0, 3, 3);
    	initialize();
    	copyToGPU();

    	Position_t position (  4.0,  4.0, 0.5 );
    	Position_t direction( -1.0, -1.0, 0.0 );
    	direction.normalize();
    	gpuRayFloat_t distance = 10.0;

    	unsigned dim = 0;
    	distances_t distances = crossingDistance( dim, position[dim], direction[dim], distance);

    	CHECK_EQUAL( 5, distances.size() );
    	CHECK_EQUAL( 3, distances.id(0) );
    	CHECK_CLOSE( (1.0)*std::sqrt(2.0), distances.dist(0), 1e-6 );
    	CHECK_EQUAL( 2, distances.id(1) );
    	CHECK_CLOSE( (2.0)*std::sqrt(2.0), distances.dist(1), 1e-6 );
    	CHECK_EQUAL( 1, distances.id(2) );
    	CHECK_CLOSE( (3.0)*std::sqrt(2.0), distances.dist(2), 1e-6 );
    	CHECK_EQUAL( 0, distances.id(3) );
    	CHECK_CLOSE( (4.0)*std::sqrt(2.0), distances.dist(3), 1e-6 );
    	CHECK_EQUAL( -1, distances.id(4) );
    	CHECK_CLOSE( 10.0, distances.dist(4), 1e-6 );

        dim = 1;
        distances.clear();
        distances = crossingDistance( dim, position[dim], direction[dim], distance);

    	CHECK_EQUAL( 5, distances.size() );
    	CHECK_EQUAL( 3, distances.id(0) );
    	CHECK_CLOSE( (1.0)*std::sqrt(2.0), distances.dist(0), 1e-6 );
    	CHECK_EQUAL( 2, distances.id(1) );
    	CHECK_CLOSE( (2.0)*std::sqrt(2.0), distances.dist(1), 1e-6 );
    	CHECK_EQUAL( 1, distances.id(2) );
    	CHECK_CLOSE( (3.0)*std::sqrt(2.0), distances.dist(2), 1e-6 );
    	CHECK_EQUAL( 0, distances.id(3) );
    	CHECK_CLOSE( (4.0)*std::sqrt(2.0), distances.dist(3), 1e-6 );
    	CHECK_EQUAL( -1, distances.id(4) );
    	CHECK_CLOSE( 10.0, distances.dist(4), 1e-6 );
    }

}

}

