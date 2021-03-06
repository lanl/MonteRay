#include <UnitTest++.h>

#include "../CrossingDistanceHelper.hh"
#include "../RayTraceHelper.hh"
#include "MonteRay_CylindricalGrid.hh"

namespace MonteRay_CylindricalGrid_crossingDistance_tests{

using namespace MonteRay;

SUITE( MonteRay_CylindricalGrid_crossingDistance_Tests) {
    using GridBins_t = MonteRay_GridBins;
    using Position_t = MonteRay::Vector3D<gpuRayFloat_t>;

    const gpuFloatType_t s2 = std::sqrt(2.0);

    enum coord {R=0,Z=1,Theta=2,DIM=3};

    inline void checkDistances( const char *file, int line,
            const std::vector<unsigned>& expectedIndex,
            const std::vector<gpuFloatType_t>& expectedDistance, const singleDimRayTraceMap_t& distances )
    {
        char const* const errorFormat = "%s(%d): error: Failure \n";
        if( expectedIndex.size() != expectedDistance.size() ) {
            printf(errorFormat, file, line);
        }
        CHECK_EQUAL( expectedIndex.size(), expectedDistance.size() );

        if( expectedIndex.size() != distances.size() ) {
            printf(errorFormat, file, line);
        }
        CHECK_EQUAL( expectedIndex.size(), distances.size() );

        for( auto i=0; i<distances.size(); ++i ) {
            if( expectedIndex[i] != distances.id(i) ) {
                printf("%s(%d): error: Failure in cell id #%d \n", file, line, i);
            }
            CHECK_EQUAL( expectedIndex   [i], distances.id(i) );

            if( std::abs( expectedDistance[i] - distances.dist(i) ) > 1.0e-5  ) {
                printf("%s(%d): error: Failure in distance #%d \n", file, line, i);
            }
            CHECK_CLOSE( expectedDistance[i], distances.dist(i), 1e-5 );
        }
    }

#define checkDistances(expectedIndex, expectedDistance, distances) { checkDistances(__FILE__, __LINE__, expectedIndex, expectedDistance, distances); }

    using distances_t = singleDimRayTraceMap_t;
    using CylindricalGrid = MonteRay_CylindricalGrid;
    using GridBins = MonteRay_GridBins;

    class CylindricalGridTester{
      public:
        std::unique_ptr<CylindricalGrid> pCyl;
        CylindricalGridTester(){
          std::vector<gpuRayFloat_t> Rverts = { 1.0, 2.0, 3.0, 5.0 };
          std::vector<gpuRayFloat_t> Zverts = { 0.0, 1.0, 2.0, 3.0, 5.0 };
        
          pCyl = std::make_unique<CylindricalGrid>(2, GridBins{Rverts, GridBins::RADIAL}, GridBins{Zverts});
        }
    };

    TEST_FIXTURE(CylindricalGridTester, CrossingDistance_in_1D_R_inward_from_outside_to_outside ) {
        Position_t position (  -6.5, 0.0,  0.5 );
        Position_t direction(    1,   0,    0 );
        gpuFloatType_t distance = 100.0;

        distances_t distances = crossingDistanceOnCPU(pCyl.get(), R, position, direction, distance );

        CHECK_EQUAL( 9,  distances.size() );
        checkDistances( std::vector<unsigned>({4,3,2,1,0,1,2,3,4}),
                std::vector<gpuFloatType_t>({1.5,3.5,4.5,5.5,7.5,8.5,9.5,11.5,distance}),
                distances );
#ifdef __CUDACC__
        auto gpuDistances = crossingDistanceOnGPU(pCyl.get(), R, position, direction, distance);
        compareDistances(distances, gpuDistances);
#endif
    }


    TEST_FIXTURE(CylindricalGridTester, CrossingDistance_in_1D_R_inward_from_outside_to_inside_stop_inward ) {


        Position_t position (  -6.5, 0.0,  0.5 );
        Position_t direction(    1,   0,    0 );
        gpuFloatType_t distance = 6.0;

        distances_t distances = crossingDistanceOnCPU(pCyl.get(), R, position, direction, distance );

        CHECK_EQUAL( 5,  distances.size() );
        checkDistances( std::vector<unsigned>({4,3,2,1,0}),
                std::vector<gpuFloatType_t>({1.5,3.5,4.5,5.5,6.0}),
                distances );
#ifdef __CUDACC__
        auto gpuDistances = crossingDistanceOnGPU(pCyl.get(), R, position, direction, distance);
        compareDistances(distances, gpuDistances);
#endif
    }

    TEST_FIXTURE(CylindricalGridTester, CrossingDistance_in_1D_R_inward_from_outside_to_inside_stop_outward ) {


        Position_t position (  -6.5, 0.0,  0.5 );
        Position_t direction(    1,   0,    0 );
        gpuFloatType_t distance = 9.0;

        distances_t distances = crossingDistanceOnCPU(pCyl.get(), R, position, direction, distance );

        CHECK_EQUAL( 7,  distances.size() );
        checkDistances( std::vector<unsigned>({4,3,2,1,0,1,2}),
                std::vector<gpuFloatType_t>({1.5,3.5,4.5,5.5,7.5,8.5,9.0}),
                distances );
#ifdef __CUDACC__
        auto gpuDistances = crossingDistanceOnGPU(pCyl.get(), R, position, direction, distance);
        compareDistances(distances, gpuDistances);
#endif
    }

    TEST_FIXTURE(CylindricalGridTester, CrossingDistance_through_a_single_cylinder_in_2D_R_inward_from_inside_to_outside ) {


        gpuFloatType_t y = 3.0f / std::sqrt(2.0f );
        gpuFloatType_t last_dist = std::sqrt( 25 - y*y );
        Position_t position (  -4.0, y,  0.5 );
        Position_t direction(    1,   0,    0 );
        gpuFloatType_t distance = 9.0;

        distances_t distances = crossingDistanceOnCPU(pCyl.get(), R, position, direction, distance );

        CHECK_EQUAL( 4,  distances.size() );
        checkDistances( std::vector<unsigned>({3,2,3,4}),
                std::vector<gpuFloatType_t>({4.0f-y,4.0f+y,4.0f+last_dist,9.0}),
                distances );
#ifdef __CUDACC__
        auto gpuDistances = crossingDistanceOnGPU(pCyl.get(), R, position, direction, distance);
        compareDistances(distances, gpuDistances);
#endif
    }


    TEST_FIXTURE(CylindricalGridTester, CrossingDistance_tanget_to_first_inner_cylinder_posY ) {

        gpuFloatType_t x = -3.5;
        gpuFloatType_t y = 3.0;
        gpuFloatType_t last_dist = std::sqrt( 25 - y*y );

        Position_t position (  x, y, 0.5 );
        Position_t direction(  1, 0,   0 );
        gpuFloatType_t distance = 9.0;

        distances_t distances = crossingDistanceOnCPU(pCyl.get(), R, position, direction, distance );

        CHECK_EQUAL( 4, distances.size() );
        CHECK_EQUAL( 3, distances.id(0) );
        CHECK_CLOSE( 3.5, distances.dist(0), 1e-5 );
        CHECK_EQUAL( 2, distances.id(1) );
        CHECK_CLOSE( 3.5, distances.dist(1), 1e-5 );
        CHECK_EQUAL( 3, distances.id(2) );
        CHECK_CLOSE( 7.5, distances.dist(2), 1e-5 );
        CHECK_EQUAL( 4, distances.id(3) );
        CHECK_CLOSE( 9.0, distances.dist(3), 1e-5 );

#ifdef __CUDACC__
        auto gpuDistances = crossingDistanceOnGPU(pCyl.get(), R, position, direction, distance);
        compareDistances(distances, gpuDistances);
#endif

    }

    TEST_FIXTURE(CylindricalGridTester, CrossingDistance_tanget_to_first_inner_cylinder_negY ) {

        gpuFloatType_t x = -3.5;
        gpuFloatType_t y = -3.0;
        gpuFloatType_t last_dist = std::sqrt( 25 - y*y );

        Position_t position (  x, y, 0.5 );
        Position_t direction(    1,   0,    0 );
        gpuFloatType_t distance = 9.0;

        distances_t distances = crossingDistanceOnCPU(pCyl.get(), R, position, direction, distance );

        CHECK_EQUAL( 4, distances.size() );
        CHECK_EQUAL( 3, distances.id(0) );
        CHECK_CLOSE( 3.5, distances.dist(0), 1e-5 );
        CHECK_EQUAL( 2, distances.id(1) );
        CHECK_CLOSE( 3.5, distances.dist(1), 1e-5 );
        CHECK_EQUAL( 3, distances.id(2) );
        CHECK_CLOSE( 7.5, distances.dist(2), 1e-5 );
        CHECK_EQUAL( 4, distances.id(3) );
        CHECK_CLOSE( 9.0, distances.dist(3), 1e-5 );

#ifdef __CUDACC__
        auto gpuDistances = crossingDistanceOnGPU(pCyl.get(), R, position, direction, distance);
        compareDistances(distances, gpuDistances);
#endif
    }

    TEST_FIXTURE(CylindricalGridTester, CrossingDistance_tanget_to_first_second_cylinder_posY ) {

        gpuFloatType_t y = 2.0;
        Position_t position (  -4.0, y,  0.5 );
        Position_t direction(    1,   0,    0 );
        gpuFloatType_t distance = 9.0;

        distances_t distances = crossingDistanceOnCPU(pCyl.get(), R, position, direction, distance );

        CHECK_EQUAL( 6, distances.size() );
        CHECK_EQUAL( 3, distances.id(0) );
        CHECK_CLOSE( 4.0 - std::sqrt(9.0-4.0), distances.dist(0), 1e-5 );
        CHECK_EQUAL( 2, distances.id(1) );
        CHECK_CLOSE( 4.0, distances.dist(1), 1e-5 );
        CHECK_EQUAL( 1, distances.id(2) );
        CHECK_CLOSE( 4.0, distances.dist(2), 1e-5 );
        CHECK_EQUAL( 2, distances.id(3) );
        CHECK_CLOSE( 4.0 + std::sqrt(9.0-4.0), distances.dist(3), 1e-5 );
        CHECK_EQUAL( 3, distances.id(4) );
        CHECK_CLOSE( 4.0 + std::sqrt(25.0-4.0), distances.dist(4), 1e-5 );
        CHECK_EQUAL( 4, distances.id(5) );
        CHECK_CLOSE( distance, distances.dist(5), 1e-5 );
#ifdef __CUDACC__
        auto gpuDistances = crossingDistanceOnGPU(pCyl.get(), R, position, direction, distance);
        compareDistances(distances, gpuDistances);
#endif
    }

    TEST_FIXTURE(CylindricalGridTester, CrossingDistance_outward_from_Origin_posX_to_outside ) {
        //        std::cout << "Debug: ---------------------------------------------------------" << std::endl;

        Position_t position (  0.0, 0.0,  0.5 );
        Position_t direction(    1,   0,    0 );
        gpuFloatType_t distance = 9.0;

        distances_t distances = crossingDistanceOnCPU(pCyl.get(), R, position, direction, distance );

        CHECK_EQUAL( 5, distances.size() );
        CHECK_EQUAL( 0, distances.id(0) );
        CHECK_CLOSE( 1.0, distances.dist(0), 1e-5 );
        CHECK_EQUAL( 1, distances.id(1) );
        CHECK_CLOSE( 2.0, distances.dist(1), 1e-5 );
        CHECK_EQUAL( 2, distances.id(2) );
        CHECK_CLOSE( 3.0, distances.dist(2), 1e-5 );
        CHECK_EQUAL( 3, distances.id(3) );
        CHECK_CLOSE( 5.0, distances.dist(3), 1e-5 );
        CHECK_EQUAL( 4, distances.id(4) );
        CHECK_CLOSE( 9.0, distances.dist(4), 1e-5 );
#ifdef __CUDACC__
        auto gpuDistances = crossingDistanceOnGPU(pCyl.get(), R, position, direction, distance);
        compareDistances(distances, gpuDistances);
#endif
    }

    TEST_FIXTURE(CylindricalGridTester, CrossingDistance_outward_from_Origin_posX_to_inside ) {
        //        std::cout << "Debug: ---------------------------------------------------------" << std::endl;

        Position_t position (  0.0, 0.0,  0.5 );
        Position_t direction(    1,   0,    0 );
        gpuFloatType_t distance = 4.5;

        distances_t distances = crossingDistanceOnCPU(pCyl.get(), R, position, direction, distance );

        CHECK_EQUAL( 4, distances.size() );
        CHECK_EQUAL( 0, distances.id(0) );
        CHECK_CLOSE( 1.0, distances.dist(0), 1e-5 );
        CHECK_EQUAL( 1, distances.id(1) );
        CHECK_CLOSE( 2.0, distances.dist(1), 1e-5 );
        CHECK_EQUAL( 2, distances.id(2) );
        CHECK_CLOSE( 3.0, distances.dist(2), 1e-5 );
        CHECK_EQUAL( 3, distances.id(3) );
        CHECK_CLOSE( 4.5, distances.dist(3), 1e-5 );
#ifdef __CUDACC__
        auto gpuDistances = crossingDistanceOnGPU(pCyl.get(), R, position, direction, distance);
        compareDistances(distances, gpuDistances);
#endif
    }

    TEST_FIXTURE(CylindricalGridTester, CrossingDistance_outward_from_posX_Postion_negX_Direction ) {
        //        std::cout << "Debug: ---------------------------------------------------------" << std::endl;

        Position_t position (  3.5, 0.0,  0.5 );
        Position_t direction(   -1,   0,    0 );
        gpuFloatType_t distance = 9.0;

        distances_t distances = crossingDistanceOnCPU(pCyl.get(), R, position, direction, distance );

        CHECK_EQUAL( 8,  distances.size() );
        checkDistances( std::vector<unsigned>({3,2,1,0,1,2,3,4}),
                std::vector<gpuFloatType_t>({0.5, 1.5, 2.5, 4.5, 5.5, 6.5, 8.5, 9.0}),
                distances );
#ifdef __CUDACC__
        auto gpuDistances = crossingDistanceOnGPU(pCyl.get(), R, position, direction, distance);
        compareDistances(distances, gpuDistances);
#endif
    }

    TEST_FIXTURE(CylindricalGridTester, CrossingDistance_outward_from_posX_Postion_negX_Direction_not_outside ) {
        // std::cout << "Debug: ---------------------------------------------------------" << std::endl;

        Position_t position (  3.5, 0.0,  0.5 );
        Position_t direction(   -1,   0,    0 );
        gpuFloatType_t distance = 7.5;

        distances_t distances = crossingDistanceOnCPU(pCyl.get(), R, position, direction, distance );

        CHECK_EQUAL( 7,  distances.size() );
        checkDistances( std::vector<unsigned>({3,2,1,0,1,2,3}),
                std::vector<gpuFloatType_t>({0.5, 1.5, 2.5, 4.5, 5.5, 6.5, 7.5}),
                distances );
#ifdef __CUDACC__
        auto gpuDistances = crossingDistanceOnGPU(pCyl.get(), R, position, direction, distance);
        compareDistances(distances, gpuDistances);
#endif
    }

    TEST_FIXTURE(CylindricalGridTester, radialCrossingDistances_inside_thru_to_outside ) {

        Position_t position (  -4.5, 0.0,  0.5 );
        Position_t direction(    1,   0,    0 );
        gpuFloatType_t distance = 100.0;

        distances_t distances = crossingDistanceOnCPU(pCyl.get(), R, position, direction, distance );

        CHECK_EQUAL( 8, distances.size() );
        CHECK_EQUAL( 3, distances.id(0) );
        CHECK_CLOSE( 1.5, distances.dist(0), 1e-5 );
        CHECK_EQUAL( 2, distances.id(1) );
        CHECK_CLOSE( 2.5, distances.dist(1), 1e-5 );
        CHECK_EQUAL( 1, distances.id(2) );
        CHECK_CLOSE( 3.5, distances.dist(2), 1e-5 );
        CHECK_EQUAL( 0, distances.id(3) );
        CHECK_CLOSE( 5.5, distances.dist(3), 1e-5 );
        CHECK_EQUAL( 1, distances.id(4) );
        CHECK_CLOSE( 6.5, distances.dist(4), 1e-5 );
        CHECK_EQUAL( 2, distances.id(5) );
        CHECK_CLOSE( 7.5, distances.dist(5), 1e-5 );
        CHECK_EQUAL( 3, distances.id(6) );
        CHECK_CLOSE( 9.5, distances.dist(6), 1e-5 );
        CHECK_EQUAL( 4, distances.id(7) );
        CHECK_CLOSE( 100.0, distances.dist(7), 1e-5 );
#ifdef __CUDACC__
        auto gpuDistances = crossingDistanceOnGPU(pCyl.get(), R, position, direction, distance);
        compareDistances(distances, gpuDistances);
#endif
    }

    TEST_FIXTURE(CylindricalGridTester, radialCrossingDistances_inside_misses_inner_cells ) {

        Position_t position (  -3.5, 3.1,  0.5 );
        Position_t direction(    1,   0,    0 );
        gpuFloatType_t distance = 100.0;

        distances_t distances = crossingDistanceOnCPU(pCyl.get(), R, position, direction, distance );

        CHECK_EQUAL( 2, distances.size() );
        CHECK_EQUAL( 3, distances.id(0) );
        CHECK_CLOSE( 3.5+std::sqrt(5.0*5.0-3.1*3.1), distances.dist(0), 1e-5 );
        CHECK_EQUAL( 4, distances.id(1) );
        CHECK_CLOSE( 100.0, distances.dist(1), 1e-5 );
#ifdef __CUDACC__
        auto gpuDistances = crossingDistanceOnGPU(pCyl.get(), R, position, direction, distance);
        compareDistances(distances, gpuDistances);
#endif
    }

    TEST_FIXTURE(CylindricalGridTester, radialCrossingDistances_twice_through_a_single_cylinder_going_inward_single_crossing_outward  ) {

        gpuFloatType_t y = 3.0 / std::sqrt(2.0 );
        Position_t position (  -4.0, y,  0.5 );
        Position_t direction(    1,   0,    0 );
        gpuFloatType_t distance = 9.0;

        distances_t distances = crossingDistanceOnCPU(pCyl.get(), R, position, direction, distance );

        CHECK_EQUAL( 4, distances.size() );
        CHECK_EQUAL( 3, distances.id(0) );
        CHECK_CLOSE( 4.0 - y, distances.dist(0), 1e-5 );
        CHECK_EQUAL( 2, distances.id(1) );
        CHECK_CLOSE( 4.0 + y, distances.dist(1), 1e-5 );
        CHECK_EQUAL( 3, distances.id(2) );
        CHECK_CLOSE( 4.0 + std::sqrt(5.0*5.0-y*y) , distances.dist(2), 1e-5 );
        CHECK_EQUAL( 4, distances.id(3) );
        CHECK_CLOSE( 9.0, distances.dist(3), 1e-5 );

#ifdef __CUDACC__
        auto gpuDistances = crossingDistanceOnGPU(pCyl.get(), R, position, direction, distance);
        compareDistances(distances, gpuDistances);
#endif
    }

}

}

