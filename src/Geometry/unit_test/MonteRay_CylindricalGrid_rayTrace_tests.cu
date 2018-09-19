#include <UnitTest++.h>

#include "MonteRay_CylindricalGrid.hh"
#include "MonteRay_SpatialGrid.hh"
#include "MonteRayVector3D.hh"
#include "MonteRay_GridBins.hh"
#include "MonteRayDefinitions.hh"
#include "MonteRayCopyMemory.t.hh"

namespace MonteRay_CylindricalGrid_rayTrace_tests{

using namespace MonteRay;

SUITE( MonteRay_CylindricalGrid_rayTrace_Tests) {
    using Grid_t = MonteRay_CylindricalGrid;
    using GridBins_t = MonteRay_GridBins;
    using GridBins_t = Grid_t::GridBins_t;
    using pGridInfo_t = GridBins_t*;
    using pArrayOfpGridInfo_t = Grid_t::pArrayOfpGridInfo_t;
    using Position_t = MonteRay::Vector3D<gpuRayFloat_t>;

    const gpuFloatType_t s2 = std::sqrt(2.0);
    const unsigned OUTSIDE_GRID = MonteRay_GridSystemInterface::OUTSIDE_GRID;

    enum coord {R=0,Z=1,Theta=2,DIM=3};

    class gridTestData {
    public:

        gridTestData(){
            std::vector<gpuRayFloat_t> Rverts = { 1.0, 2.0, 3.0, 5.0 };
            std::vector<gpuRayFloat_t> Zverts = { -6, -3, -1, 1, 3, 6 };

            pGridInfo[R] = new GridBins_t();
            pGridInfo[Z] = new GridBins_t();

            pGridInfo[R]->initialize( Rverts );
            pGridInfo[Z]->initialize( Zverts );

        }
        ~gridTestData(){
            delete pGridInfo[R];
            delete pGridInfo[Z];
        }

        MonteRay_SpatialGrid::pArrayOfpGridInfo_t pGridInfo;
    };

    typedef singleDimRayTraceMap_t distances_t;
    typedef singleDimRayTraceMap_t rayTraceMap_t;
    typedef rayTraceList_t rayTrace_t;
    typedef MonteRay_CylindricalGrid CylindricalGrid;

    inline void checkDistances( const char *file, int line,
            const std::vector<unsigned>& expectedIndex,
            const std::vector<gpuFloatType_t>& expectedDistance, const rayTraceList_t& distances )
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

    // ************************ rayTrace Testing ****************************

    TEST( rayTrace_in_ZDir_outside_to_outside ) {

        gridTestData data;
        CylindricalGrid grid(2, data.pGridInfo);

        Position_t position ( 0.0, 0.0, -7.5 );
        Position_t direction(   0,   0,    1 );
        gpuFloatType_t distance = 100.0;

        rayTraceList_t distances;
        grid.rayTrace( distances, position, direction, distance);

        CHECK_EQUAL( 5,  distances.size() );
        checkDistances( std::vector<unsigned>({0,4,8,12,16}), std::vector<gpuFloatType_t>({3.,2.,2.,2.,3.}), distances );
    }

    TEST( rayTrace_in_ZDir_outside_to_outside_along_radial_vertex ) {

        gridTestData data;
        CylindricalGrid grid(2,data.pGridInfo);

        Position_t position ( 1.0, 0.0, -7.5 );
        Position_t direction(   0,   0,    1 );
        gpuFloatType_t distance = 100.0;

        rayTraceList_t distances;
        grid.rayTrace( distances, position, direction, distance);

        CHECK_EQUAL(   5,  distances.size() );
        checkDistances( std::vector<unsigned>({1,5,9,13,17}), std::vector<gpuFloatType_t>({3.0,2.0,2.0,2.0,3.0}), distances );
    }

    TEST( rayTrace_in_RDir_outside_to_outside ) {
        // std::cout << "Debug: -------------------------------------" << std::endl;

        gridTestData data;
        CylindricalGrid grid(2,data.pGridInfo);

        Position_t position ( -6.0, 0.0,  -5.0 );
        Position_t direction(   1,   0,    0 );
        double distance = 100.0;

        rayTraceList_t distances;
        grid.rayTrace( distances, position, direction, distance);

        CHECK_EQUAL(   7,  distances.size() );
        checkDistances( std::vector<unsigned>({3,2,1,0,1,2,3}), std::vector<gpuFloatType_t>({2.0,1.0,1.0,2.0,1.0,1.0,2.0}), distances );
    }

    TEST( rayTrace_in_RDir_outside_to_outside_along_Z_vertex ) {
        // std::cout << "Debug: -------------------------------------" << std::endl;

        gridTestData data;
        CylindricalGrid grid(2,data.pGridInfo);

        Position_t position ( -6.0, 0.0,  -3.0 );
        Position_t direction(   1,   0,    0 );
        double distance = 100.0;

        rayTraceList_t distances;
        grid.rayTrace( distances, position, direction, distance);

        CHECK_EQUAL(   7,  distances.size() );
        checkDistances( std::vector<unsigned>({7,6,5,4,5,6,7}), std::vector<gpuFloatType_t>({2.0,1.0,1.0,2.0,1.0,1.0,2.0}), distances );
    }

    TEST( rayTrace_in_RZDir_outside_to_outside_at_45degrees_thru_a_corner ) {
        // !!! DO NOT CHANGE WITHOUT DRAWING A PICTURE

        // std::cout << "Debug: -------------------------------------" << std::endl;

        gridTestData data;
        CylindricalGrid grid(2,data.pGridInfo);

        Position_t position ( -6.0, 0.0,  -7.0 );
        Position_t direction(   1,   0,    1 );
        direction.normalize();
        double distance = 100.0;

        rayTraceList_t distances;
        grid.rayTrace( distances, position, direction, distance);

        CHECK_EQUAL(   11,  distances.size() );

        // 3rd entry can be 1 or 6, 8th can be 10 or 13
        checkDistances( std::vector<unsigned>({3,2,6,5,4,8,9,10,14,15,19}), std::vector<gpuFloatType_t>({2*s2,s2,0,s2,s2,s2,s2,0, s2, s2,s2}), distances );
    }

    TEST( rayTrace_in_RZDir_inside_to_outside_at_45degrees_thru_a_corner ) {
        // !!! DO NOT CHANGE WITHOUT DRAWING A PICTURE

        // std::cout << "Debug: -------------------------------------" << std::endl;

        gridTestData data;
        CylindricalGrid grid(2,data.pGridInfo);

        Position_t position ( -2.0, 0.0,  -3.0 );
        Position_t direction(   -1,   0,    -1 );
        direction.normalize();
        double distance = 100.0;

        rayTraceList_t distances;
        grid.rayTrace( distances, position, direction, distance);

        CHECK_EQUAL(   3,  distances.size() );
        checkDistances( std::vector<unsigned>({6,2,3}), std::vector<gpuFloatType_t>({0,s2,2*s2}), distances );
    }

    TEST( rayTrace_outside_negZ_Position_negZ_Direction) {
        // std::cout << "Debug: -------------------------------------" << std::endl;

        gridTestData data;
        CylindricalGrid grid(2,data.pGridInfo);

        Position_t position ( -4.0, 0.0,  -7.0 );
        Position_t direction(   0,   0,    -1 );
        double distance = 100.0;

        rayTraceList_t distances;
        grid.rayTrace( distances, position, direction, distance);

        CHECK_EQUAL(   0,  distances.size() );
    }

    TEST( rayTrace_outside_negR_Position_negR_Direction) {
        // std::cout << "Debug: -------------------------------------" << std::endl;

        gridTestData data;
        CylindricalGrid grid(2,data.pGridInfo);

        Position_t position ( -6.0, 0.0, -5.0 );
        Position_t direction(   -1,   0,    0 );
        double distance = 100.0;

        rayTraceList_t distances;
        grid.rayTrace( distances, position, direction, distance);

        CHECK_EQUAL(   0,  distances.size() );
    }

    TEST( rayTrace_in_RZDir_outside_to_outside_at_45degrees_thru_a_corner_with_outside_distance ) {
        // !!! DO NOT CHANGE WITHOUT DRAWING A PICTURE

        // std::cout << "Debug: -------------------------------------" << std::endl;

        gridTestData data;
        CylindricalGrid grid(2,data.pGridInfo);

        Position_t position ( -6.0, 0.0,  -7.0 );
        Position_t direction(   1,   0,    1 );
        direction.normalize();
        double distance = 100.0;

        rayTraceList_t distances;
        grid.rayTrace( distances, position, direction, distance, true);       //distances_t distances = grid.radialCrossingDistances( position, direction, distance);

        CHECK_EQUAL( 13,  distances.size() );

        // 5th entry can be 1 or 6, 10th can be 10 or 13
        checkDistances( std::vector<unsigned>( {OUTSIDE_GRID,OUTSIDE_GRID,3,2,6,5,4,8,9,10,14,15,19}), std::vector<gpuFloatType_t>({s2,0,2*s2,s2,0,s2,s2,s2,s2,0,s2,s2,s2}), distances );
    }

}

}

