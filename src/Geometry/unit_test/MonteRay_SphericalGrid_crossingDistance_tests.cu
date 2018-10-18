#include <UnitTest++.h>

#include "MonteRay_SphericalGrid.hh"
#include "MonteRay_SpatialGrid.hh"
#include "MonteRayVector3D.hh"
#include "MonteRay_GridBins.hh"
#include "MonteRayDefinitions.hh"
#include "MonteRayCopyMemory.t.hh"

namespace MonteRay_SphericalGrid_crossingDistance_tests{

using namespace MonteRay;

SUITE( SphericalGrid_crossinDistance_Tests) {
    typedef Vector3D<gpuRayFloat_t> Position_t;
    using GridBins_t = MonteRay_SphericalGrid::GridBins_t;
    using pArrayOfpGridInfo_t = MonteRay_SphericalGrid::pArrayOfpGridInfo_t;

    enum coord {R=0,DIM=1};
    class gridTestData {
    public:
        gridTestData(){
            std::vector<gpuRayFloat_t> Rverts = { 1.0, 2.0, 3.0, 5.0 };

            pGridInfo[R] = new GridBins_t();
            pGridInfo[R]->initialize( Rverts );
        }
        ~gridTestData(){ delete pGridInfo[R]; }

        pArrayOfpGridInfo_t pGridInfo;
    };

    class MonteRay_SphericalGrid_tester : public MonteRay_SphericalGrid {
    public:
        MonteRay_SphericalGrid_tester(unsigned d, pArrayOfpGridInfo_t pBins) :
            MonteRay_SphericalGrid(d,pBins) {}

        MonteRay_SphericalGrid_tester(unsigned d, GridBins_t* pBins ) :
            MonteRay_SphericalGrid(d,pBins) {}

        void radialCrossingDistancesSingleDirection( singleDimRayTraceMap_t& rayTraceMap,
                const Position_t& pos,
                const Direction_t& dir,
                gpuRayFloat_t distance,
                bool outward ) const {
            MonteRay_SphericalGrid::radialCrossingDistancesSingleDirection( rayTraceMap, pos, dir, distance, outward );
        }

        void radialCrossingDistances(singleDimRayTraceMap_t& rayTraceMap,
                const Position_t& pos,
                const Direction_t& dir,
                gpuRayFloat_t distance ) const {
            MonteRay_SphericalGrid::radialCrossingDistances( rayTraceMap, pos, dir, distance );
        }

    };

    using SphericalGrid = MonteRay_SphericalGrid_tester;

    typedef singleDimRayTraceMap_t distances_t;
    typedef rayTraceList_t rayTraceList_t;

    TEST( CrossingDistance_in_1D_R_inward_from_outside_to_outside ) {
        gridTestData data;
        SphericalGrid grid(1, data.pGridInfo );

        Position_t position (  -6.5, 0.0,  0.0 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 100.0;

        distances_t distances;
        grid.radialCrossingDistancesSingleDirection( distances, position, direction, distance, false);

        CHECK_EQUAL( 8, distances.size() );
        CHECK_EQUAL( 4, distances.id(0) );
        CHECK_CLOSE( 1.5, distances.dist(0), 1e-11 );
        CHECK_EQUAL( 3, distances.id(1) );
        CHECK_CLOSE( 3.5, distances.dist(1), 1e-11 );
        CHECK_EQUAL( 2, distances.id(2) );
        CHECK_CLOSE( 4.5, distances.dist(2), 1e-11 );
        CHECK_EQUAL( 1, distances.id(3) );
        CHECK_CLOSE( 5.5, distances.dist(3), 1e-11 );
        CHECK_EQUAL( 0, distances.id(4) );
        CHECK_CLOSE( 7.5, distances.dist(4), 1e-11 );
        CHECK_EQUAL( 1, distances.id(5) );
        CHECK_CLOSE( 8.5, distances.dist(5), 1e-11 );
        CHECK_EQUAL( 2, distances.id(6) );
        CHECK_CLOSE( 9.5, distances.dist(6), 1e-11 );
        CHECK_EQUAL( 3, distances.id(7) );
        CHECK_CLOSE( 11.5, distances.dist(7), 1e-11 );
    }
    TEST( CrossingDistance_in_1D_R_inward_from_inside ) {
        gridTestData data;
        SphericalGrid grid(1, data.pGridInfo );

        Position_t position (  -4.5, 0.0,  0.0 );
        Position_t direction(     1,   0,    0 );
        gpuRayFloat_t distance = 100.0;

        distances_t distances;
        grid.radialCrossingDistancesSingleDirection( distances, position, direction, distance, false);

        CHECK_EQUAL( 6, distances.size() );
        CHECK_EQUAL( 3, distances.id(0) );
        CHECK_CLOSE( 1.5, distances.dist(0), 1e-11 );
        CHECK_EQUAL( 2, distances.id(1) );
        CHECK_CLOSE( 2.5, distances.dist(1), 1e-11 );
        CHECK_EQUAL( 1, distances.id(2) );
        CHECK_CLOSE( 3.5, distances.dist(2), 1e-11 );
        CHECK_EQUAL( 0, distances.id(3) );
        CHECK_CLOSE( 5.5, distances.dist(3), 1e-11 );
        CHECK_EQUAL( 1, distances.id(4) );
        CHECK_CLOSE( 6.5, distances.dist(4), 1e-11 );
        CHECK_EQUAL( 2, distances.id(5) );
        CHECK_CLOSE( 7.5, distances.dist(5), 1e-11 );
    }
    TEST( CrossingDistance_in_1D_R_inward_from_outside_to_inside_stop_inward ) {
        gridTestData data;
        SphericalGrid grid(1, data.pGridInfo );

        Position_t position (  -6.5, 0.0,  0.0 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 6.0;

        distances_t distances;
        grid.radialCrossingDistancesSingleDirection( distances, position, direction, distance, false);

        CHECK_EQUAL( 5, distances.size() );
        CHECK_EQUAL( 4, distances.id(0) );
        CHECK_CLOSE( 1.5, distances.dist(0), 1e-11 );
        CHECK_EQUAL( 3, distances.id(1) );
        CHECK_CLOSE( 3.5, distances.dist(1), 1e-11 );
        CHECK_EQUAL( 2, distances.id(2) );
        CHECK_CLOSE( 4.5, distances.dist(2), 1e-11 );
        CHECK_EQUAL( 1, distances.id(3) );
        CHECK_CLOSE( 5.5, distances.dist(3), 1e-11 );
        CHECK_EQUAL( 0, distances.id(4) );
        CHECK_CLOSE( 6.0, distances.dist(4), 1e-11 );
    }
    TEST( CrossingDistance_in_1D_R_inward_from_outside_to_inside_stop_outward ) {
        gridTestData data;
        SphericalGrid grid(1, data.pGridInfo );

        Position_t position (  -6.5, 0.0,  0.0 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 9.0;

        distances_t distances;
        grid.radialCrossingDistancesSingleDirection( distances, position, direction, distance, false);

        CHECK_EQUAL( 7, distances.size() );
        CHECK_EQUAL( 4, distances.id(0) );
        CHECK_CLOSE( 1.5, distances.dist(0), 1e-11 );
        CHECK_EQUAL( 3, distances.id(1) );
        CHECK_CLOSE( 3.5, distances.dist(1), 1e-11 );
        CHECK_EQUAL( 2, distances.id(2) );
        CHECK_CLOSE( 4.5, distances.dist(2), 1e-11 );
        CHECK_EQUAL( 1, distances.id(3) );
        CHECK_CLOSE( 5.5, distances.dist(3), 1e-11 );
        CHECK_EQUAL( 0, distances.id(4) );
        CHECK_CLOSE( 7.5, distances.dist(4), 1e-11 );
        CHECK_EQUAL( 1, distances.id(5) );
        CHECK_CLOSE( 8.5, distances.dist(5), 1e-11 );
        CHECK_EQUAL( 2, distances.id(6) );
        CHECK_CLOSE( 9.0, distances.dist(6), 1e-11 );
    }
    TEST( CrossingDistance_through_a_single_sphere_in_2D_R_inward_from_inside_to_outside ) {
        gridTestData data;
        SphericalGrid grid(1, data.pGridInfo );

        gpuRayFloat_t y = 3.0 / std::sqrt(2.0 );
        Position_t position (  -4.0, y,  0.0 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 9.0;

        distances_t distances;
        grid.radialCrossingDistancesSingleDirection( distances, position, direction, distance, false);

        CHECK_EQUAL( 2, distances.size() );
        CHECK_EQUAL( 3, distances.id(0) );
        CHECK_CLOSE( 4.0 - y, distances.dist(0), 1e-5 );
        CHECK_EQUAL( 2, distances.id(1) );
        CHECK_CLOSE( 4.0 + y, distances.dist(1), 1e-5 );
    }
    TEST( CrossingDistance_tanget_to_first_inner_sphere_posY ) {
        gridTestData data;
        SphericalGrid grid(1, data.pGridInfo );

        gpuRayFloat_t x = -3.5;
        gpuRayFloat_t y = 3.0;

        Position_t position (  x, y, 0.0 );
        Position_t direction(  1, 0,   0 );
        gpuRayFloat_t distance = 9.0;

        distances_t distances;
        grid.radialCrossingDistancesSingleDirection( distances, position, direction, distance, false);

        CHECK_EQUAL( 0, distances.size() );
    }
    TEST( CrossingDistance_tanget_to_first_inner_sphere_negY ) {
        gridTestData data;
        SphericalGrid grid(1, data.pGridInfo );

        gpuRayFloat_t x = -3.5;
        gpuRayFloat_t y = -3.0;

        Position_t position (  x, y, 0.0 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 9.0;

        distances_t distances;
        grid.radialCrossingDistancesSingleDirection( distances, position, direction, distance, false);

        CHECK_EQUAL( 0, distances.size() );
    }
    TEST( CrossingDistance_tanget_to_first_second_sphere_posY ) {
        gridTestData data;
        SphericalGrid grid(1, data.pGridInfo );

        gpuRayFloat_t y = 2.0;
        Position_t position (  -4.0, y,  0.0 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 9.0;

        distances_t distances;
        grid.radialCrossingDistancesSingleDirection( distances, position, direction, distance, false);

        CHECK_EQUAL( 2, distances.size() );
        CHECK_EQUAL( 3, distances.id(0) );
        CHECK_CLOSE( 4.0 - std::sqrt(9.0-4.0), distances.dist(0), 1e-5 );
        CHECK_EQUAL( 2, distances.id(1) );
        CHECK_CLOSE( 4.0 + std::sqrt(9.0-4.0), distances.dist(1), 1e-5 );
    }

    TEST( CrossingDistance_outward_from_Origin_posX_to_outside ) {
        //        std::cout << "Debug: ---------------------------------------------------------" << std::endl;
        gridTestData data;
        SphericalGrid grid(1, data.pGridInfo );

        Position_t position (  0.0, 0.0,  0.0 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 9.0;

        distances_t distances;
        grid.radialCrossingDistancesSingleDirection( distances, position, direction, distance, true);

        CHECK_EQUAL( 5, distances.size() );
        CHECK_EQUAL( 0, distances.id(0) );
        CHECK_CLOSE( 1.0, distances.dist(0), 1e-11 );
        CHECK_EQUAL( 1, distances.id(1) );
        CHECK_CLOSE( 2.0, distances.dist(1), 1e-11 );
        CHECK_EQUAL( 2, distances.id(2) );
        CHECK_CLOSE( 3.0, distances.dist(2), 1e-11 );
        CHECK_EQUAL( 3, distances.id(3) );
        CHECK_CLOSE( 5.0, distances.dist(3), 1e-11 );
        CHECK_EQUAL( 4, distances.id(4) );
        CHECK_CLOSE( 9.0, distances.dist(4), 1e-11 );
    }
    TEST( CrossingDistance_outward_from_Origin_posX_to_inside ) {
        //        std::cout << "Debug: ---------------------------------------------------------" << std::endl;
        gridTestData data;
        SphericalGrid grid(1, data.pGridInfo );

        Position_t position (  0.0, 0.0,  0.0 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 4.5;

        distances_t distances;
        grid.radialCrossingDistancesSingleDirection( distances, position, direction, distance, true);

        CHECK_EQUAL( 4, distances.size() );
        CHECK_EQUAL( 0, distances.id(0) );
        CHECK_CLOSE( 1.0, distances.dist(0), 1e-11 );
        CHECK_EQUAL( 1, distances.id(1) );
        CHECK_CLOSE( 2.0, distances.dist(1), 1e-11 );
        CHECK_EQUAL( 2, distances.id(2) );
        CHECK_CLOSE( 3.0, distances.dist(2), 1e-11 );
        CHECK_EQUAL( 3, distances.id(3) );
        CHECK_CLOSE( 4.5, distances.dist(3), 1e-11 );
    }

    TEST( CrossingDistance_outward_from_posX_Postion_negX_Direction ) {
        //        std::cout << "Debug: ---------------------------------------------------------" << std::endl;
        gridTestData data;
        SphericalGrid grid(1, data.pGridInfo );

        Position_t position (  3.5, 0.0,  0.0 );
        Position_t direction(   -1,   0,    0 );
        gpuRayFloat_t distance = 9.0;

        distances_t distances;
        grid.radialCrossingDistancesSingleDirection( distances, position, direction, distance, true);

        CHECK_EQUAL( 2, distances.size() );
        CHECK_EQUAL( 3, distances.id(0) );
        CHECK_CLOSE( 8.5, distances.dist(0), 1e-11 );
        CHECK_EQUAL( 4, distances.id(1) );
        CHECK_CLOSE( 9.0, distances.dist(1), 1e-11 );
    }

    TEST( CrossingDistance_outward_from_posX_Postion_negX_Direction_not_outside ) {
        // std::cout << "Debug: ---------------------------------------------------------" << std::endl;
        gridTestData data;
        SphericalGrid grid(1, data.pGridInfo );

        Position_t position (  3.5, 0.0,  0.0 );
        Position_t direction(   -1,   0,    0 );
        gpuRayFloat_t distance = 7.5;

        distances_t distances;
        grid.radialCrossingDistancesSingleDirection( distances, position, direction, distance, true);

        CHECK_EQUAL( 1, distances.size() );
        CHECK_EQUAL( 3, distances.id(0) );
        CHECK_CLOSE( 7.5, distances.dist(0), 1e-11 );
    }

    TEST( radialCrossingDistances_inside_thru_to_outside ) {
        gridTestData data;
        SphericalGrid grid(1, data.pGridInfo );

        Position_t position (  -4.5, 0.0,  0.0 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 100.0;

        distances_t distances;
        grid.radialCrossingDistances( distances, position, direction, distance);

        CHECK_EQUAL( 8, distances.size() );
        CHECK_EQUAL( 3, distances.id(0) );
        CHECK_CLOSE( 1.5, distances.dist(0), 1e-11 );
        CHECK_EQUAL( 2, distances.id(1) );
        CHECK_CLOSE( 2.5, distances.dist(1), 1e-11 );
        CHECK_EQUAL( 1, distances.id(2) );
        CHECK_CLOSE( 3.5, distances.dist(2), 1e-11 );
        CHECK_EQUAL( 0, distances.id(3) );
        CHECK_CLOSE( 5.5, distances.dist(3), 1e-11 );
        CHECK_EQUAL( 1, distances.id(4) );
        CHECK_CLOSE( 6.5, distances.dist(4), 1e-11 );
        CHECK_EQUAL( 2, distances.id(5) );
        CHECK_CLOSE( 7.5, distances.dist(5), 1e-11 );
        CHECK_EQUAL( 3, distances.id(6) );
        CHECK_CLOSE( 9.5, distances.dist(6), 1e-11 );
        CHECK_EQUAL( 4, distances.id(7) );
        CHECK_CLOSE( 100.0, distances.dist(7), 1e-11 );
    }

    TEST( radialCrossingDistances_inside_misses_inner_cells ) {
        gridTestData data;
        SphericalGrid grid(1, data.pGridInfo );

        Position_t position (  -3.5, 3.1,  0.0 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 100.0;

        distances_t distances;
        grid.radialCrossingDistances( distances, position, direction, distance);

        CHECK_EQUAL( 2, distances.size() );
        CHECK_EQUAL( 3, distances.id(0) );
        CHECK_CLOSE( 3.5+std::sqrt(5.0*5.0-3.1*3.1), distances.dist(0), 1e-5 );
        CHECK_EQUAL( 4, distances.id(1) );
        CHECK_CLOSE( 100.0, distances.dist(1), 1e-11 );
    }

    TEST( radialCrossingDistances_twice_through_a_single_sphere_going_inward_single_crossing_outward  ) {
        gridTestData data;
        SphericalGrid grid(1, data.pGridInfo );

        gpuRayFloat_t y = 3.0 / std::sqrt(2.0 );
        Position_t position (  -4.0, y,  0.0 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 9.0;

        distances_t distances;
        grid.radialCrossingDistances( distances, position, direction, distance);

        CHECK_EQUAL( 4, distances.size() );
        CHECK_EQUAL( 3, distances.id(0) );
        CHECK_CLOSE( 4.0 - y, distances.dist(0), 1e-5 );
        CHECK_EQUAL( 2, distances.id(1) );
        CHECK_CLOSE( 4.0 + y, distances.dist(1), 1e-5 );
        CHECK_EQUAL( 3, distances.id(2) );
        CHECK_CLOSE( 4.0 + std::sqrt(5.0*5.0-y*y) , distances.dist(2), 1e-5 );
        CHECK_EQUAL( 4, distances.id(3) );
        CHECK_CLOSE( 9.0, distances.dist(3), 1e-11 );
    }

}

}

