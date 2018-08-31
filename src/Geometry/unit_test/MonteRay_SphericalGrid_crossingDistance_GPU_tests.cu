#include <UnitTest++.h>

#include "MonteRay_SpatialGrid_GPU_helper.hh"

namespace MonteRay_SphericalGrid_crossingDistance_tests{

using namespace MonteRay;
using namespace MonteRay_SpatialGrid_helper;

SUITE( SphericalGrid_crossingDistance_GPU_Tests) {
#ifdef __CUDACC__

	TEST( setup ) {
		gpuReset();
	}

	typedef singleDimRayTraceMap_t distances_t;
	typedef singleDimRayTraceMap_t rayTraceMap_t;

	TEST_FIXTURE(SpatialGridGPUTester, CrossingDistance_in_1D_R_inward_from_outside_to_outside ) {
		std::vector<gpuRayFloat_t> vertices= { 1.0, 2.0, 3.0, 5.0 };

		setCoordinateSystem( TransportMeshTypeEnum::Spherical );
		setDimension( 1 );
		setGrid( MonteRay_SpatialGrid::SPH_R, vertices);
		initialize();
		copyToGPU();

		Position_t position (  -6.5, 0.0,  0.0 );
		Position_t direction(    1,   0,    0 );
		gpuRayFloat_t distance = 100.0;

		distances_t distances = crossingDistance( position, direction, distance );

		CHECK_EQUAL( 9, distances.size() );
		CHECK_EQUAL( 4, distances.id(0) );
		CHECK_CLOSE( 1.5, distances.dist(0), 1e-6 );
		CHECK_EQUAL( 3, distances.id(1) );
		CHECK_CLOSE( 3.5, distances.dist(1), 1e-6 );
		CHECK_EQUAL( 2, distances.id(2) );
		CHECK_CLOSE( 4.5, distances.dist(2), 1e-6 );
		CHECK_EQUAL( 1, distances.id(3) );
		CHECK_CLOSE( 5.5, distances.dist(3), 1e-6 );
		CHECK_EQUAL( 0, distances.id(4) );
		CHECK_CLOSE( 7.5, distances.dist(4), 1e-6 );
		CHECK_EQUAL( 1, distances.id(5) );
		CHECK_CLOSE( 8.5, distances.dist(5), 1e-6 );
		CHECK_EQUAL( 2, distances.id(6) );
		CHECK_CLOSE( 9.5, distances.dist(6), 1e-6 );
		CHECK_EQUAL( 3, distances.id(7) );
		CHECK_CLOSE( 11.5, distances.dist(7), 1e-6 );
		CHECK_EQUAL( 4, distances.id(8) );
		CHECK_CLOSE( 100.0, distances.dist(8), 1e-6 );
	}
	TEST_FIXTURE(SpatialGridGPUTester, CrossingDistance_in_1D_R_inward_from_inside ) {
		std::vector<gpuRayFloat_t> vertices= { 1.0, 2.0, 3.0, 5.0 };

		setCoordinateSystem( TransportMeshTypeEnum::Spherical );
		setDimension( 1 );
		setGrid( MonteRay_SpatialGrid::SPH_R, vertices);
		initialize();
		copyToGPU();

		Position_t position (  -4.5, 0.0,  0.0 );
		Position_t direction(     1,   0,    0 );
		gpuRayFloat_t distance = 100.0;

		distances_t distances = crossingDistance( position, direction, distance );

		CHECK_EQUAL( 8, distances.size() );
		CHECK_EQUAL( 3, distances.id(0) );
		CHECK_CLOSE( 1.5, distances.dist(0), 1e-6 );
		CHECK_EQUAL( 2, distances.id(1) );
		CHECK_CLOSE( 2.5, distances.dist(1), 1e-6 );
		CHECK_EQUAL( 1, distances.id(2) );
		CHECK_CLOSE( 3.5, distances.dist(2), 1e-6 );
		CHECK_EQUAL( 0, distances.id(3) );
		CHECK_CLOSE( 5.5, distances.dist(3), 1e-6 );
		CHECK_EQUAL( 1, distances.id(4) );
		CHECK_CLOSE( 6.5, distances.dist(4), 1e-6 );
		CHECK_EQUAL( 2, distances.id(5) );
		CHECK_CLOSE( 7.5, distances.dist(5), 1e-6 );
		CHECK_EQUAL( 3, distances.id(6) );
		CHECK_CLOSE( 9.5, distances.dist(6), 1e-6 );
		CHECK_EQUAL( 4, distances.id(7) );
		CHECK_CLOSE( 100.0, distances.dist(7), 1e-6 );
	}
	TEST_FIXTURE(SpatialGridGPUTester, CrossingDistance_in_1D_R_inward_from_outside_to_inside_stop_inward ) {
		std::vector<gpuRayFloat_t> vertices= { 1.0, 2.0, 3.0, 5.0 };

		setCoordinateSystem( TransportMeshTypeEnum::Spherical );
		setDimension( 1 );
		setGrid( MonteRay_SpatialGrid::SPH_R, vertices);
		initialize();
		copyToGPU();

		Position_t position (  -6.5, 0.0,  0.0 );
		Position_t direction(    1,   0,    0 );
		gpuRayFloat_t distance = 6.0;

		distances_t distances = crossingDistance( position, direction, distance );

		CHECK_EQUAL( 5, distances.size() );
		CHECK_EQUAL( 4, distances.id(0) );
		CHECK_CLOSE( 1.5, distances.dist(0), 1e-6 );
		CHECK_EQUAL( 3, distances.id(1) );
		CHECK_CLOSE( 3.5, distances.dist(1), 1e-6 );
		CHECK_EQUAL( 2, distances.id(2) );
		CHECK_CLOSE( 4.5, distances.dist(2), 1e-6 );
		CHECK_EQUAL( 1, distances.id(3) );
		CHECK_CLOSE( 5.5, distances.dist(3), 1e-6 );
		CHECK_EQUAL( 0, distances.id(4) );
		CHECK_CLOSE( 6.0, distances.dist(4), 1e-6 );
	}
	TEST_FIXTURE(SpatialGridGPUTester, CrossingDistance_in_1D_R_inward_from_outside_to_inside_stop_outward ) {
		std::vector<gpuRayFloat_t> vertices= { 1.0, 2.0, 3.0, 5.0 };

		setCoordinateSystem( TransportMeshTypeEnum::Spherical );
		setDimension( 1 );
		setGrid( MonteRay_SpatialGrid::SPH_R, vertices);
		initialize();
		copyToGPU();

		Position_t position (  -6.5, 0.0,  0.0 );
		Position_t direction(    1,   0,    0 );
		gpuRayFloat_t distance = 9.0;

		distances_t distances = crossingDistance( position, direction, distance );

		CHECK_EQUAL( 7, distances.size() );
		CHECK_EQUAL( 4, distances.id(0) );
		CHECK_CLOSE( 1.5, distances.dist(0), 1e-6 );
		CHECK_EQUAL( 3, distances.id(1) );
		CHECK_CLOSE( 3.5, distances.dist(1), 1e-6 );
		CHECK_EQUAL( 2, distances.id(2) );
		CHECK_CLOSE( 4.5, distances.dist(2), 1e-6 );
		CHECK_EQUAL( 1, distances.id(3) );
		CHECK_CLOSE( 5.5, distances.dist(3), 1e-6 );
		CHECK_EQUAL( 0, distances.id(4) );
		CHECK_CLOSE( 7.5, distances.dist(4), 1e-6 );
		CHECK_EQUAL( 1, distances.id(5) );
		CHECK_CLOSE( 8.5, distances.dist(5), 1e-6 );
		CHECK_EQUAL( 2, distances.id(6) );
		CHECK_CLOSE( 9.0, distances.dist(6), 1e-6 );
	}
	TEST_FIXTURE(SpatialGridGPUTester, CrossingDistance_through_a_single_sphere_in_2D_R_inward_from_inside_to_outside ) {
		std::vector<gpuRayFloat_t> vertices= { 1.0, 2.0, 3.0, 5.0 };

		setCoordinateSystem( TransportMeshTypeEnum::Spherical );
		setDimension( 1 );
		setGrid( MonteRay_SpatialGrid::SPH_R, vertices);
		initialize();
		copyToGPU();

		gpuRayFloat_t y = 3.0 / std::sqrt(2.0 ); // Cross the R=3 sphere at (3/sqrt(2), 3/sqrt(2), 0)
		Position_t position (  -4.0, y,  0.0 );
		Position_t direction(    1,   0,    0 );
		gpuRayFloat_t distance = 9.0;

		distances_t distances = crossingDistance( position, direction, distance );

		CHECK_EQUAL( 4, distances.size() );
		CHECK_EQUAL( 3, distances.id(0) );
		CHECK_CLOSE( 4.0 - y, distances.dist(0), 1e-5 );
		CHECK_EQUAL( 2, distances.id(1) );
		CHECK_CLOSE( 4.0 + y, distances.dist(1), 1e-5 );
		CHECK_EQUAL( 3, distances.id(2) );
		CHECK_CLOSE( 4 + std::sqrt(25.0-4.5), distances.dist(2), 1e-5 ); // Cross the R=5 sphere at 25 = (3/sqrt(2))^2 + x^2 => x = 4.52769
		CHECK_EQUAL( 4, distances.id(3) );
		CHECK_CLOSE( 9.0, distances.dist(3), 1e-6 );
	}
	TEST_FIXTURE(SpatialGridGPUTester, CrossingDistance_tangent_to_first_inner_sphere_posY ) {
		std::vector<gpuRayFloat_t> vertices= { 1.0, 2.0, 3.0, 5.0 };

		setCoordinateSystem( TransportMeshTypeEnum::Spherical );
		setDimension( 1 );
		setGrid( MonteRay_SpatialGrid::SPH_R, vertices);
		initialize();
		copyToGPU();

		gpuRayFloat_t x = -3.5;
		gpuRayFloat_t y = 3.0;

		Position_t position (  x, y, 0.0 );
		Position_t direction(  1, 0,   0 );
		gpuRayFloat_t distance = 9.0;

		distances_t distances = crossingDistance( position, direction, distance );

		CHECK_EQUAL( 2, distances.size() );
		CHECK_EQUAL( 3, distances.id(0) );
		CHECK_CLOSE( 3.5 + std::sqrt(25.0 - y*y), distances.dist(0), 1e-6 );
		CHECK_EQUAL( 4, distances.id(1) );
		CHECK_CLOSE( 9.0, distances.dist(1), 1e-6 );
	}
	TEST_FIXTURE(SpatialGridGPUTester, CrossingDistance_tangent_to_first_inner_sphere_negY ) {
		std::vector<gpuRayFloat_t> vertices= { 1.0, 2.0, 3.0, 5.0 };

		setCoordinateSystem( TransportMeshTypeEnum::Spherical );
		setDimension( 1 );
		setGrid( MonteRay_SpatialGrid::SPH_R, vertices);
		initialize();
		copyToGPU();

		gpuRayFloat_t x = -3.5;
		gpuRayFloat_t y = -3.0;

		Position_t position (  x, y, 0.0 );
		Position_t direction(    1,   0,    0 );
		gpuRayFloat_t distance = 9.0;

		distances_t distances = crossingDistance( position, direction, distance );

		CHECK_EQUAL( 2, distances.size() );
		CHECK_EQUAL( 3, distances.id(0) );
		CHECK_CLOSE( 3.5 + std::sqrt(25.0 - y*y), distances.dist(0), 1e-6 );
		CHECK_EQUAL( 4, distances.id(1) );
		CHECK_CLOSE( 9.0, distances.dist(1), 1e-6 );
	}
	TEST_FIXTURE(SpatialGridGPUTester, CrossingDistance_tangent_to_first_second_sphere_posY ) {
		std::vector<gpuRayFloat_t> vertices= { 1.0, 2.0, 3.0, 5.0 };

		setCoordinateSystem( TransportMeshTypeEnum::Spherical );
		setDimension( 1 );
		setGrid( MonteRay_SpatialGrid::SPH_R, vertices);
		initialize();
		copyToGPU();

		gpuRayFloat_t y = 2.0;
		Position_t position (  -4.0, y,  0.0 );
		Position_t direction(    1,   0,    0 );
		gpuRayFloat_t distance = 9.0;

		distances_t distances = crossingDistance( position, direction, distance );

		CHECK_EQUAL( 4, distances.size() );
		CHECK_EQUAL( 3, distances.id(0) );
		CHECK_CLOSE( 4.0 - std::sqrt(9.0-4.0), distances.dist(0), 1e-5 );
		CHECK_EQUAL( 2, distances.id(1) );
		CHECK_CLOSE( 4.0 + std::sqrt(9.0-4.0), distances.dist(1), 1e-5 );
		CHECK_EQUAL( 3, distances.id(2) );
		CHECK_CLOSE( 4.0 + std::sqrt(25.0-4.0), distances.dist(2), 1e-5 );
		CHECK_EQUAL( 4, distances.id(3) );
		CHECK_CLOSE( 9.0, distances.dist(3), 1e-6 );
	}

	TEST_FIXTURE(SpatialGridGPUTester, CrossingDistance_outward_from_Origin_posX_to_outside ) {
		//        std::cout << "Debug: ---------------------------------------------------------" << std::endl;
		std::vector<gpuRayFloat_t> vertices= { 1.0, 2.0, 3.0, 5.0 };

		setCoordinateSystem( TransportMeshTypeEnum::Spherical );
		setDimension( 1 );
		setGrid( MonteRay_SpatialGrid::SPH_R, vertices);
		initialize();
		copyToGPU();

		Position_t position (  0.0, 0.0,  0.0 );
		Position_t direction(    1,   0,    0 );
		gpuRayFloat_t distance = 9.0;

		distances_t distances = crossingDistance( position, direction, distance );

		CHECK_EQUAL( 5, distances.size() );
		CHECK_EQUAL( 0, distances.id(0) );
		CHECK_CLOSE( 1.0, distances.dist(0), 1e-6 );
		CHECK_EQUAL( 1, distances.id(1) );
		CHECK_CLOSE( 2.0, distances.dist(1), 1e-6 );
		CHECK_EQUAL( 2, distances.id(2) );
		CHECK_CLOSE( 3.0, distances.dist(2), 1e-6 );
		CHECK_EQUAL( 3, distances.id(3) );
		CHECK_CLOSE( 5.0, distances.dist(3), 1e-6 );
		CHECK_EQUAL( 4, distances.id(4) );
		CHECK_CLOSE( 9.0, distances.dist(4), 1e-6 );
	}
	TEST_FIXTURE(SpatialGridGPUTester, CrossingDistance_outward_from_Origin_posX_to_inside ) {
		//        std::cout << "Debug: ---------------------------------------------------------" << std::endl;
		std::vector<gpuRayFloat_t> vertices= { 1.0, 2.0, 3.0, 5.0 };

		setCoordinateSystem( TransportMeshTypeEnum::Spherical );
		setDimension( 1 );
		setGrid( MonteRay_SpatialGrid::SPH_R, vertices);
		initialize();
		copyToGPU();

		Position_t position (  0.0, 0.0,  0.0 );
		Position_t direction(    1,   0,    0 );
		gpuRayFloat_t distance = 4.5;

		distances_t distances = crossingDistance( position, direction, distance );

		CHECK_EQUAL( 4, distances.size() );
		CHECK_EQUAL( 0, distances.id(0) );
		CHECK_CLOSE( 1.0, distances.dist(0), 1e-6 );
		CHECK_EQUAL( 1, distances.id(1) );
		CHECK_CLOSE( 2.0, distances.dist(1), 1e-6 );
		CHECK_EQUAL( 2, distances.id(2) );
		CHECK_CLOSE( 3.0, distances.dist(2), 1e-6 );
		CHECK_EQUAL( 3, distances.id(3) );
		CHECK_CLOSE( 4.5, distances.dist(3), 1e-6 );
	}

	TEST_FIXTURE(SpatialGridGPUTester, CrossingDistance_outward_from_posX_Postion_negX_Direction ) {
		//        std::cout << "Debug: ---------------------------------------------------------" << std::endl;
		std::vector<gpuRayFloat_t> vertices= { 1.0, 2.0, 3.0, 5.0 };

		setCoordinateSystem( TransportMeshTypeEnum::Spherical );
		setDimension( 1 );
		setGrid( MonteRay_SpatialGrid::SPH_R, vertices);
		initialize();
		copyToGPU();

		Position_t position (  3.5, 0.0,  0.0 );
		Position_t direction(   -1,   0,    0 );
		gpuRayFloat_t distance = 9.0;

		distances_t distances = crossingDistance( position, direction, distance );

		CHECK_EQUAL( 8, distances.size() );
		CHECK_EQUAL( 3, distances.id(0) );
		CHECK_CLOSE( 0.5, distances.dist(0), 1e-6 );
		CHECK_EQUAL( 2, distances.id(1) );
		CHECK_CLOSE( 1.5, distances.dist(1), 1e-6 );
		CHECK_EQUAL( 1, distances.id(2) );
		CHECK_CLOSE( 2.5, distances.dist(2), 1e-6 );
		CHECK_EQUAL( 0, distances.id(3) );
		CHECK_CLOSE( 4.5, distances.dist(3), 1e-6 );
		CHECK_EQUAL( 1, distances.id(4) );
		CHECK_CLOSE( 5.5, distances.dist(4), 1e-6 );
		CHECK_EQUAL( 2, distances.id(5) );
		CHECK_CLOSE( 6.5, distances.dist(5), 1e-6 );
		CHECK_EQUAL( 3, distances.id(6) );
		CHECK_CLOSE( 8.5, distances.dist(6), 1e-6 );
		CHECK_EQUAL( 4, distances.id(7) );
		CHECK_CLOSE( 9.0, distances.dist(7), 1e-6 );
	}

	TEST_FIXTURE(SpatialGridGPUTester, CrossingDistance_outward_from_posX_Postion_negX_Direction_not_outside ) {
		// std::cout << "Debug: ---------------------------------------------------------" << std::endl;
		std::vector<gpuRayFloat_t> vertices= { 1.0, 2.0, 3.0, 5.0 };

		setCoordinateSystem( TransportMeshTypeEnum::Spherical );
		setDimension( 1 );
		setGrid( MonteRay_SpatialGrid::SPH_R, vertices);
		initialize();
		copyToGPU();

		Position_t position (  3.5, 0.0,  0.0 );
		Position_t direction(   -1,   0,    0 );
		gpuRayFloat_t distance = 7.5;

		distances_t distances = crossingDistance( position, direction, distance );

		CHECK_EQUAL( 7, distances.size() );
		CHECK_EQUAL( 3, distances.id(0) );
		CHECK_CLOSE( 0.5, distances.dist(0), 1e-6 );
		CHECK_EQUAL( 2, distances.id(1) );
		CHECK_CLOSE( 1.5, distances.dist(1), 1e-6 );
		CHECK_EQUAL( 1, distances.id(2) );
		CHECK_CLOSE( 2.5, distances.dist(2), 1e-6 );
		CHECK_EQUAL( 0, distances.id(3) );
		CHECK_CLOSE( 4.5, distances.dist(3), 1e-6 );
		CHECK_EQUAL( 1, distances.id(4) );
		CHECK_CLOSE( 5.5, distances.dist(4), 1e-6 );
		CHECK_EQUAL( 2, distances.id(5) );
		CHECK_CLOSE( 6.5, distances.dist(5), 1e-6 );
		CHECK_EQUAL( 3, distances.id(6) );
		CHECK_CLOSE( 7.5, distances.dist(6), 1e-6 );
	}

	TEST_FIXTURE(SpatialGridGPUTester, radialCrossingDistances_inside_thru_to_outside ) {
		std::vector<gpuRayFloat_t> vertices= { 1.0, 2.0, 3.0, 5.0 };

		setCoordinateSystem( TransportMeshTypeEnum::Spherical );
		setDimension( 1 );
		setGrid( MonteRay_SpatialGrid::SPH_R, vertices);
		initialize();
		copyToGPU();

		Position_t position (  -4.5, 0.0,  0.0 );
		Position_t direction(    1,   0,    0 );
		gpuRayFloat_t distance = 100.0;

		distances_t distances = crossingDistance( position, direction, distance);

		CHECK_EQUAL( 8, distances.size() );
		CHECK_EQUAL( 3, distances.id(0) );
		CHECK_CLOSE( 1.5, distances.dist(0), 1e-6 );
		CHECK_EQUAL( 2, distances.id(1) );
		CHECK_CLOSE( 2.5, distances.dist(1), 1e-6 );
		CHECK_EQUAL( 1, distances.id(2) );
		CHECK_CLOSE( 3.5, distances.dist(2), 1e-6 );
		CHECK_EQUAL( 0, distances.id(3) );
		CHECK_CLOSE( 5.5, distances.dist(3), 1e-6 );
		CHECK_EQUAL( 1, distances.id(4) );
		CHECK_CLOSE( 6.5, distances.dist(4), 1e-6 );
		CHECK_EQUAL( 2, distances.id(5) );
		CHECK_CLOSE( 7.5, distances.dist(5), 1e-6 );
		CHECK_EQUAL( 3, distances.id(6) );
		CHECK_CLOSE( 9.5, distances.dist(6), 1e-6 );
		CHECK_EQUAL( 4, distances.id(7) );
		CHECK_CLOSE( 100.0, distances.dist(7), 1e-6 );
	}

	TEST_FIXTURE(SpatialGridGPUTester, radialCrossingDistances_inside_misses_inner_cells ) {
		std::vector<gpuRayFloat_t> vertices= { 1.0, 2.0, 3.0, 5.0 };

		setCoordinateSystem( TransportMeshTypeEnum::Spherical );
		setDimension( 1 );
		setGrid( MonteRay_SpatialGrid::SPH_R, vertices);
		initialize();
		copyToGPU();

		Position_t position (  -3.5, 3.1,  0.0 );
		Position_t direction(    1,   0,    0 );
		gpuRayFloat_t distance = 100.0;

		distances_t distances = crossingDistance( position, direction, distance);

		CHECK_EQUAL( 2, distances.size() );
		CHECK_EQUAL( 3, distances.id(0) );
		CHECK_CLOSE( 3.5+std::sqrt(5.0*5.0-3.1*3.1), distances.dist(0), 1e-5 );
		CHECK_EQUAL( 4, distances.id(1) );
		CHECK_CLOSE( 100.0, distances.dist(1), 1e-6 );
	}

	TEST_FIXTURE(SpatialGridGPUTester, radialCrossingDistances_twice_through_a_single_sphere_going_inward_single_crossing_outward  ) {
		std::vector<gpuRayFloat_t> vertices= { 1.0, 2.0, 3.0, 5.0 };

		setCoordinateSystem( TransportMeshTypeEnum::Spherical );
		setDimension( 1 );
		setGrid( MonteRay_SpatialGrid::SPH_R, vertices);
		initialize();
		copyToGPU();

		gpuRayFloat_t y = 3.0 / std::sqrt(2.0 );
		Position_t position (  -4.0, y,  0.0 );
		Position_t direction(    1,   0,    0 );
		gpuRayFloat_t distance = 9.0;

		distances_t distances = crossingDistance( position, direction, distance);

		CHECK_EQUAL( 4, distances.size() );
		CHECK_EQUAL( 3, distances.id(0) );
		CHECK_CLOSE( 4.0 - y, distances.dist(0), 1e-5 );
		CHECK_EQUAL( 2, distances.id(1) );
		CHECK_CLOSE( 4.0 + y, distances.dist(1), 1e-5 );
		CHECK_EQUAL( 3, distances.id(2) );
		CHECK_CLOSE( 4.0 + std::sqrt(5.0*5.0-y*y) , distances.dist(2), 1e-5 );
		CHECK_EQUAL( 4, distances.id(3) );
		CHECK_CLOSE( 9.0, distances.dist(3), 1e-6 );
	}
#endif
}

}

