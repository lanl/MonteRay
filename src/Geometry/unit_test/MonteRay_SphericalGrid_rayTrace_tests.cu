#include <UnitTest++.h>

#include "MonteRay_SphericalGrid.hh"
#include "MonteRay_SpatialGrid.hh"
#include "MonteRayVector3D.hh"
#include "MonteRay_GridBins.hh"
#include "MonteRayDefinitions.hh"

namespace MonteRay_SphericalGrid_rayTrace_tests{

using namespace MonteRay;

SUITE( SphericalGrid_Tests) {

    typedef Vector3D<gpuRayFloat_t> Position_t;
	using GridBins_t = MonteRay_GridBins;

    class gridTestData {
    public:
        enum coord {R};
        gridTestData(){
            std::vector<gpuRayFloat_t> Rvertices = { 1.0, 2.0, 3.0, 5.0 };

            pGridInfo[R] = new GridBins_t();
            pGridInfo[R]->initialize( Rvertices );
        }
        ~gridTestData(){
        	delete pGridInfo[R];
        }

        MonteRay_SpatialGrid::pArrayOfpGridInfo_t pGridInfo;
    };

	typedef singleDimRayTraceMap_t distances_t;
	typedef singleDimRayTraceMap_t rayTraceMap_t;
	typedef rayTraceList_t rayTrace_t;

// ************************ rayTrace Testing ****************************


    TEST( rayTrace_in_RDir_crossingDistances_outside_to_outside ) {
         // std::cout << "Debug: -------------------------------------" << std::endl;

         gridTestData data;
         MonteRay_SphericalGrid grid(1,data.pGridInfo);

         Position_t position ( -6.0, 0.0,  0.0 );
         Position_t direction(   1,   0,    0 );
         gpuRayFloat_t distance = 100.0;

         distances_t distances;
         grid.radialCrossingDistances( distances, position, direction, 4, distance );

         CHECK_EQUAL(   9,  distances.size() );
         CHECK_EQUAL(   4,  distances.id(0) );
         CHECK_CLOSE( 1.0,  distances.dist(0), 1e-11 );
         CHECK_EQUAL(   3,  distances.id(1) );
         CHECK_CLOSE( 3.0,  distances.dist(1), 1e-11 );
         CHECK_EQUAL(   2,  distances.id(2) );
         CHECK_CLOSE( 4.0,  distances.dist(2), 1e-11 );
         CHECK_EQUAL(   1,  distances.id(3) );
         CHECK_CLOSE( 5.0,  distances.dist(3), 1e-11 );
         CHECK_EQUAL(   0,  distances.id(4) );
         CHECK_CLOSE( 7.0,  distances.dist(4), 1e-11 );
         CHECK_EQUAL(   1,  distances.id(5) );
         CHECK_CLOSE( 8.0,  distances.dist(5), 1e-11 );
         CHECK_EQUAL(   2,  distances.id(6) );
         CHECK_CLOSE( 9.0,  distances.dist(6), 1e-11 );
         CHECK_EQUAL(   3,  distances.id(7) );
         CHECK_CLOSE( 11.0,  distances.dist(7), 1e-11 );
         CHECK_EQUAL(   4,  distances.id(8) );
         CHECK_CLOSE( 100.0,  distances.dist(8), 1e-11 );
     }

    TEST( rayTrace_in_RDir_outside_to_outside ) {
        // std::cout << "Debug: -------------------------------------" << std::endl;

        gridTestData data;
        MonteRay_SphericalGrid grid(1,data.pGridInfo);

        Position_t position ( -6.0, 0.0,  0.0 );
        Position_t direction(   1,   0,    0 );
        gpuRayFloat_t distance = 100.0;

        rayTraceList_t distances;
        grid.rayTrace( distances, position, direction, distance );
        //distances_t distances = grid.radialCrossingDistances( position, direction, distance);

        CHECK_EQUAL(   7,  distances.size() );
        CHECK_EQUAL(   3,  distances.id(0));
        CHECK_CLOSE( 2.0,  distances.dist(0), 1e-11 );
        CHECK_EQUAL(   2,  distances.id(1) );
        CHECK_CLOSE( 1.0,  distances.dist(1), 1e-11 );
        CHECK_EQUAL(   1,  distances.id(2) );
        CHECK_CLOSE( 1.0,  distances.dist(2), 1e-11 );
        CHECK_EQUAL(   0,  distances.id(3) );
        CHECK_CLOSE( 2.0,  distances.dist(3), 1e-11 );
        CHECK_EQUAL(   1,  distances.id(4) );
        CHECK_CLOSE( 1.0,  distances.dist(4), 1e-11 );
        CHECK_EQUAL(   2,  distances.id(5) );
        CHECK_CLOSE( 1.0,  distances.dist(5), 1e-11 );
        CHECK_EQUAL(   3,  distances.id(6) );
        CHECK_CLOSE( 2.0,  distances.dist(6), 1e-11 );
    }

    TEST( rayTrace_in_RZDir_outside_to_outside_at_45degrees ) {

        // std::cout << "Debug: -------------------------------------" << std::endl;

        gridTestData data;
        MonteRay_SphericalGrid grid(1,data.pGridInfo);

        Position_t position ( -6.0, 0.0,  -6.0 );
        Position_t direction(   1,   0,    1 );
        direction.normalize();
        gpuRayFloat_t distance = 100.0;

        rayTraceList_t distances;
        grid.rayTrace( distances, position, direction, distance );

        CHECK_EQUAL(   7,  distances.size() );
        CHECK_EQUAL(   3,  distances.id(0) );
        CHECK_CLOSE( 2.0,  distances.dist(0), 1e-11 );
        CHECK_EQUAL(   2,  distances.id(1) );
        CHECK_CLOSE( 1.0,  distances.dist(1), 1e-11 );
        CHECK_EQUAL(   1,  distances.id(2) );
        CHECK_CLOSE( 1.0,  distances.dist(2), 1e-11 );
        CHECK_EQUAL(   0,  distances.id(3) );
        CHECK_CLOSE( 2.0,  distances.dist(3), 1e-11 );
        CHECK_EQUAL(   1,  distances.id(4) );
        CHECK_CLOSE( 1.0,  distances.dist(4), 1e-11 );
        CHECK_EQUAL(   2,  distances.id(5) );
        CHECK_CLOSE( 1.0,  distances.dist(5), 1e-11 );
        CHECK_EQUAL(   3,  distances.id(6) );
        CHECK_CLOSE( 2.0,  distances.dist(6), 1e-11 );
    }

    TEST( rayTrace_in_RZDir_inside_to_outside_at_45degrees ) {
        // std::cout << "Debug: -------------------------------------" << std::endl;

        gridTestData data;
        MonteRay_SphericalGrid grid(1,data.pGridInfo);

        Position_t position ( -2.0, 0.0,  -2.0 );
        Position_t direction(   -1,   0,    -1 );
        direction.normalize();
        gpuRayFloat_t distance = 100.0;

        rayTraceList_t distances;
        grid.rayTrace( distances, position, direction, distance );

        CHECK_EQUAL(   2,  distances.size() );
        CHECK_EQUAL(   2,  distances.id(0) );
        CHECK_CLOSE( 0.171572875254,  distances.dist(0), 1e-11 );
        CHECK_EQUAL(   3,  distances.id(1) );
        CHECK_CLOSE( 2.0,  distances.dist(1), 1e-11 );
    }

    TEST( rayTrace_outside_negR_Position_negR_Direction) {
        // std::cout << "Debug: -------------------------------------" << std::endl;

        gridTestData data;
        MonteRay_SphericalGrid grid(1,data.pGridInfo);

        Position_t position ( -6.0, 0.0, -5.0 );
        Position_t direction(   -1,   0,    0 );
        gpuRayFloat_t distance = 100.0;

        rayTraceList_t distances;
        grid.rayTrace( distances, position, direction, distance );

        CHECK_EQUAL(   0,  distances.size() );
    }

    TEST( rayTrace_outside_to_inside_negR_Position_pos_Direction_wOutsideDistance) {
        // std::cout << "Debug: -------------------------------------" << std::endl;

        gridTestData data;
        MonteRay_SphericalGrid grid(1,data.pGridInfo);

        Position_t position ( -6.0, 0.0, -0.0 );
        Position_t direction(   1.0,   0,    0 );
        gpuRayFloat_t distance = 1.5;

        rayTraceList_t distances;
        grid.rayTrace( distances, position, direction, distance );

        CHECK_EQUAL(   1,  distances.size() );
        CHECK_EQUAL(  3,  distances.id(0) );
        CHECK_CLOSE(  0.5,  distances.dist(0), 1e-11 );
    }

    TEST( rayTrace_stay_outside_negR_Position_neg_Direction_wOutsideDistance) {
        // std::cout << "Debug: -------------------------------------" << std::endl;

        gridTestData data;
        MonteRay_SphericalGrid grid(1,data.pGridInfo);

        Position_t position ( -6.0, 0.0, -0.0 );
        Position_t direction(  -1.0,   0,    0 );
        gpuRayFloat_t distance = 10.0;

        rayTraceList_t distances;
        grid.rayTrace( distances, position, direction, distance );

        CHECK_EQUAL(   0,  distances.size() );
    }
    TEST( rayTrace_stay_outside_negR_Position_pos_Direction_misses_wOutsideDistance) {
        // std::cout << "Debug: -------------------------------------" << std::endl;

        gridTestData data;
        MonteRay_SphericalGrid grid(1,data.pGridInfo);

        Position_t position ( -6.0, 0.0, -6.0 );
        Position_t direction(  1.0,   0,    0 );
        gpuRayFloat_t distance = 10.0;

        rayTraceList_t distances;
        grid.rayTrace( distances, position, direction, distance );

        CHECK_EQUAL(   0,  distances.size() );
    }

}

}