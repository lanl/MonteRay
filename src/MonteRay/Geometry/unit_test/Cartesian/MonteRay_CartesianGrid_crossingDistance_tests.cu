#include <UnitTest++.h>

#include "MonteRay_CartesianGrid.t.hh"
#include "MonteRay_SpatialGrid.hh"
#include "MonteRayVector3D.hh"
#include "MonteRay_GridBins.hh"
#include "MonteRayDefinitions.hh"
#include "ThirdParty/Array.hh"

namespace MonteRay_CartesianGrid_crossingDistance_tests{

using namespace MonteRay;

SUITE( MonteRay_CartesianGrid_crossingDistance_Tests) {
    typedef Vector3D<gpuRayFloat_t> Position_t;
    using GridBins_t = MonteRay_GridBins;
    enum coord {X,Y,Z,DIM};

    class CartesianGridTester {
    public:

        CartesianGridTester(){
            std::vector<gpuRayFloat_t> vertices{
                -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
                  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10};

            pCart = std::make_unique<MonteRay_CartesianGrid>(3, 
              std::array<MonteRay_GridBins, 3>{
                MonteRay_GridBins{vertices},
                MonteRay_GridBins{vertices},
                MonteRay_GridBins{vertices}
              }
            );

        }

        std::unique_ptr<MonteRay_CartesianGrid> pCart;
    };

    typedef singleDimRayTraceMap_t distances_t;
    typedef singleDimRayTraceMap_t rayTraceMap_t;
    TEST_FIXTURE(CartesianGridTester, CrossingDistance_in_1D_PosXDir ) {
        auto& cart = *pCart;

        Position_t position ( -9.5, 0.5,  0.5 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 1.0;

        const unsigned dim = 0; const unsigned threadID=0;
        RayWorkInfo rayInfo(1,true);
        cart.crossingDistance(dim, threadID, rayInfo, position[dim], direction[dim], distance);
        singleDimRayTraceMap_t distances( rayInfo, 0, dim );

        CHECK_EQUAL( 2, distances.size() );
        CHECK_EQUAL( 0, distances.id(0) );
        CHECK_CLOSE( 0.5, distances.dist(0), 1e-6 );
        CHECK_EQUAL( 1, distances.id(1) );
        CHECK_CLOSE( 1.0, distances.dist(1), 1e-6 );
    }


    TEST_FIXTURE(CartesianGridTester,  CrossingDistance_in_1D_NegXDir ) {
        auto& cart = *pCart;

        Position_t position ( -8.5, 0.5,  0.5 );
        Position_t direction(    -1,   0,    0 );
        gpuRayFloat_t distance = 1.0;

        const unsigned dim = 0; const unsigned threadID=0;
        RayWorkInfo rayInfo(1,true);
        cart.crossingDistance(dim, threadID, rayInfo, position[dim], direction[dim], distance);
        singleDimRayTraceMap_t distances( rayInfo, 0, dim );

        CHECK_EQUAL( 2, distances.size() );
        CHECK_EQUAL( 1, distances.id(0) );
        CHECK_CLOSE( 0.5, distances.dist(0), 1e-6 );
        CHECK_EQUAL( 0, distances.id(1) );
        CHECK_CLOSE( 1.0, distances.dist(1), 1e-6 );
    }

    TEST_FIXTURE(CartesianGridTester,  Outside_negSide_negDir ) {
        auto& cart = *pCart;

        Position_t position ( -10.5, 0.5,  0.5 );
        Position_t direction(    -1,   0,    0 );
        gpuRayFloat_t distance = 2.0;

        const unsigned dim = 0; const unsigned threadID=0;
        RayWorkInfo rayInfo(1,true);
        cart.crossingDistance(dim, threadID, rayInfo, position[dim], direction[dim], distance);
        singleDimRayTraceMap_t distances( rayInfo, 0, dim );

        CHECK_EQUAL(  0, distances.size() );
    }

    TEST_FIXTURE(CartesianGridTester,  Outside_posSide_posDir ) {
        auto& cart = *pCart;

        Position_t position (  10.5, 0.5,  0.5 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 2.0;

        const unsigned dim = 0; const unsigned threadID=0;
        RayWorkInfo rayInfo(1,true);
        cart.crossingDistance(dim, threadID, rayInfo, position[dim], direction[dim], distance);
        singleDimRayTraceMap_t distances( rayInfo, 0, dim );

        CHECK_EQUAL(  0, distances.size() );
    }

    TEST_FIXTURE(CartesianGridTester,  Outside_negSide_posDir ) {
        auto& cart = *pCart;

        Position_t position ( -10.5, 0.5,  0.5 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 2.0;

        const unsigned dim = 0; const unsigned threadID=0;
        RayWorkInfo rayInfo(1,true);
        cart.crossingDistance(dim, threadID, rayInfo, position[dim], direction[dim], distance);
        singleDimRayTraceMap_t distances( rayInfo, 0, dim );

        CHECK_EQUAL( 3, distances.size() );
        CHECK_EQUAL( -1, distances.id(0) );
        CHECK_CLOSE( 0.5, distances.dist(0), 1e-6 );
        CHECK_EQUAL( 0, distances.id(1) );
        CHECK_CLOSE( 1.5, distances.dist(1), 1e-6 );
        CHECK_EQUAL( 1, distances.id(2) );
        CHECK_CLOSE( 2.0, distances.dist(2), 1e-6 );
    }

    TEST_FIXTURE(CartesianGridTester,  Outside_posSide_negDir ) {
        auto& cart = *pCart;

        Position_t position (  10.5, 0.5,  0.5 );
        Position_t direction(    -1,   0,    0 );
        gpuRayFloat_t distance = 2.0;

        const unsigned dim = 0; const unsigned threadID=0;
        RayWorkInfo rayInfo(1,true);
        cart.crossingDistance(dim, threadID, rayInfo, position[dim], direction[dim], distance);
        singleDimRayTraceMap_t distances( rayInfo, 0, dim );
        
        CHECK_EQUAL( 3, distances.size() );
        CHECK_EQUAL( 20, distances.id(0) );
        CHECK_CLOSE( 0.5, distances.dist(0), 1e-6 );
        CHECK_EQUAL( 19, distances.id(1) );
        CHECK_CLOSE( 1.5, distances.dist(1), 1e-6 );
        CHECK_EQUAL( 18, distances.id(2) );
        CHECK_CLOSE( 2.0, distances.dist(2), 1e-6 );
    }

    TEST_FIXTURE(CartesianGridTester,  Crossing_entire_grid_starting_outside_finish_outside_pos_dir ) {
        auto& cart = *pCart;

        Position_t position (  -10.5, 0.5,  0.5 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 21.0;

        const unsigned dim = 0; const unsigned threadID=0;
        RayWorkInfo rayInfo(1,true);
        cart.crossingDistance(dim, threadID, rayInfo, position[dim], direction[dim], distance);
        singleDimRayTraceMap_t distances( rayInfo, 0, dim );

        CHECK_EQUAL( 22, distances.size() );
        CHECK_EQUAL( -1, distances.id(0) );
        CHECK_CLOSE( 0.5, distances.dist(0), 1e-6 );
        CHECK_EQUAL( 0, distances.id(1) );
        CHECK_CLOSE( 1.5, distances.dist(1), 1e-6 );
        CHECK_EQUAL( 1, distances.id(2) );
        CHECK_CLOSE( 2.5, distances.dist(2), 1e-6 );
        CHECK_EQUAL( 17, distances.id(18) );
        CHECK_CLOSE( 18.5, distances.dist(18), 1e-6 );
        CHECK_EQUAL( 18, distances.id(19) );
        CHECK_CLOSE( 19.5, distances.dist(19), 1e-6 );
        CHECK_EQUAL( 19, distances.id(20) );
        CHECK_CLOSE( 20.5, distances.dist(20), 1e-6 );
        CHECK_EQUAL( 20, distances.id(21) );
        CHECK_CLOSE( 21.0, distances.dist(21), 1e-6 );

    }

    TEST_FIXTURE(CartesianGridTester,  Crossing_entire_grid_starting_outside_finish_outside_neg_dir ) {
        auto& cart = *pCart;

        Position_t position (  10.5, 0.5,  0.5 );
        Position_t direction(   -1,   0,    0 );
        gpuRayFloat_t distance = 21.0;

        const unsigned dim = 0; const unsigned threadID=0;
        RayWorkInfo rayInfo(1,true);
        cart.crossingDistance(dim, threadID, rayInfo, position[dim], direction[dim], distance);
        singleDimRayTraceMap_t distances( rayInfo, 0, dim );

        CHECK_EQUAL( 22, distances.size() );
        CHECK_EQUAL( 20, distances.id(0) );
        CHECK_CLOSE( 0.5, distances.dist(0), 1e-6 );
        CHECK_EQUAL( 19, distances.id(1) );
        CHECK_CLOSE( 1.5, distances.dist(1), 1e-6 );
        CHECK_EQUAL( 18, distances.id(2) );
        CHECK_CLOSE( 2.5, distances.dist(2), 1e-6 );
        CHECK_EQUAL( 2, distances.id(18) );
        CHECK_CLOSE( 18.5, distances.dist(18), 1e-6 );
        CHECK_EQUAL( 1, distances.id(19) );
        CHECK_CLOSE( 19.5, distances.dist(19), 1e-6 );
        CHECK_EQUAL( 0, distances.id(20) );
        CHECK_CLOSE( 20.5, distances.dist(20), 1e-6 );
        CHECK_EQUAL( -1, distances.id(21) );
        CHECK_CLOSE( 21.0, distances.dist(21), 1e-6 );
    }

    TEST_FIXTURE(CartesianGridTester,  Inside_cross_out_negDir ) {
        auto& cart = *pCart;

        Position_t position (  -8.5, 0.5,  0.5 );
        Position_t direction(    -1,   0,    0 );
        gpuRayFloat_t distance = 2.0;

        const unsigned dim = 0; const unsigned threadID=0;
        RayWorkInfo rayInfo(1,true);
        cart.crossingDistance(dim, threadID, rayInfo, position[dim], direction[dim], distance);
        singleDimRayTraceMap_t distances( rayInfo, 0, dim );

        CHECK_EQUAL( 3, distances.size() );
        CHECK_EQUAL( 1, distances.id(0) );
        CHECK_CLOSE( 0.5, distances.dist(0), 1e-6 );
        CHECK_EQUAL( 0, distances.id(1) );
        CHECK_CLOSE( 1.5, distances.dist(1), 1e-6 );
        CHECK_EQUAL( -1, distances.id(2) );
        CHECK_CLOSE( 2.0, distances.dist(2), 1e-6 );
    }

    TEST_FIXTURE(CartesianGridTester,  Inside_cross_out_posDir ) {
        auto& cart = *pCart;

        Position_t position (  8.5, 0.5,  0.5 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 2.0;

        const unsigned dim = 0; const unsigned threadID=0;
        RayWorkInfo rayInfo(1,true);
        cart.crossingDistance(dim, threadID, rayInfo, position[dim], direction[dim], distance);
        singleDimRayTraceMap_t distances( rayInfo, 0, dim );

        CHECK_EQUAL( 3, distances.size() );
        CHECK_EQUAL( 18, distances.id(0) );
        CHECK_CLOSE( 0.5, distances.dist(0), 1e-6 );
        CHECK_EQUAL( 19, distances.id(1) );
        CHECK_CLOSE( 1.5, distances.dist(1), 1e-6 );
        CHECK_EQUAL( 20, distances.id(2) );
        CHECK_CLOSE( 2.0, distances.dist(2), 1e-6 );
    }

    TEST( crossingDistance_2D_internal_hit_corner_posXDir_posYDir ) {
        Position_t position (  -.5, -.5, -.5 );
        Position_t direction(  1.0,  1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 1.0*std::sqrt(2.0);

        MonteRay_CartesianGrid cart(3, 
            MonteRay_GridBins{-1, 1, 2},
            MonteRay_GridBins{-1, 1, 2},
            MonteRay_GridBins{-1, 1, 2});

        unsigned dim = 0;
        distances_t distances;
        CHECK_EQUAL( 3, cart.getDimension() );
        CHECK_EQUAL( 0, cart.getDimIndex(0, -0.5) );
        CHECK_EQUAL( 0, cart.getDimIndex(1, -0.5) );
        CHECK_EQUAL( 0, cart.getDimIndex(2, -0.5) );
        CHECK_EQUAL( 1, cart.getDimIndex(0, 0.5) );
        CHECK_EQUAL( 1, cart.getDimIndex(1, 0.5) );
        CHECK_EQUAL( 1, cart.getDimIndex(2, 0.5) );
        CHECK_EQUAL( 2, cart.getNumBins(0) );
        CHECK_EQUAL( 2, cart.getNumBins(1) );
        CHECK_EQUAL( 2, cart.getNumBins(2) );

        //cart.crossingDistance( distances, dim, position, direction, distance);

        unsigned threadID=0;
        RayWorkInfo rayInfo(1,true);
        cart.crossingDistance(dim, threadID, rayInfo, position[dim], direction[dim], distance);

        {
            singleDimRayTraceMap_t distances( rayInfo, 0, dim );

            CHECK_EQUAL( 2, distances.size() );
            CHECK_EQUAL( 0, distances.id(0) );
            CHECK_CLOSE( (0.5)*std::sqrt(2.0), distances.dist(0), 1e-6 );
            CHECK_EQUAL( 1, distances.id(1) );
            CHECK_CLOSE( (1.0)*std::sqrt(2.0), distances.dist(1), 1e-6 );
        }

        dim = 1;
        cart.crossingDistance(dim, threadID, rayInfo, position[dim], direction[dim], distance);

        {
            singleDimRayTraceMap_t distances( rayInfo, 0, dim );

            CHECK_EQUAL( 2, distances.size() );
            CHECK_EQUAL( 0, distances.id(0) );
            CHECK_CLOSE( (0.5)*std::sqrt(2.0), distances.dist(0), 1e-6 );
            CHECK_EQUAL( 1, distances.id(1) );
            CHECK_CLOSE( (1.0)*std::sqrt(2.0), distances.dist(1), 1e-6 );
        }
    }


    TEST( crossingDistance_2D_start_on_an_external_corner_posX_posY ) {
        Position_t position (  0.0, 0.0, 0.5 );
        Position_t direction(  1.0,  1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        MonteRay_CartesianGrid cart(3, 
            MonteRay_GridBins{0, 3, 3},
            MonteRay_GridBins{0, 3, 3},
            MonteRay_GridBins{0, 3, 3});

        unsigned dim = 0;
        unsigned threadID=0;
        RayWorkInfo rayInfo(1,true);
        cart.crossingDistance(dim, threadID, rayInfo, position[dim], direction[dim], distance);

        {
            singleDimRayTraceMap_t distances( rayInfo, 0, dim );
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

        dim = 1;
        cart.crossingDistance(dim, threadID, rayInfo, position[dim], direction[dim], distance);

        {
            singleDimRayTraceMap_t distances( rayInfo, 0, dim );

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

    }

    TEST( crossingDistance_2D_start_on_an_external_corner_negX_negY ) {
        Position_t position (  3.0,  3.0, 0.5 );
        Position_t direction( -1.0, -1.0, 0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        MonteRay_CartesianGrid cart(3, 
            MonteRay_GridBins{0, 3, 3},
            MonteRay_GridBins{0, 3, 3},
            MonteRay_GridBins{0, 3, 3});

        unsigned dim = 0;

        unsigned threadID=0;
        RayWorkInfo rayInfo(1,true);
        cart.crossingDistance(dim, threadID, rayInfo, position[dim], direction[dim], distance);

        {
            singleDimRayTraceMap_t distances( rayInfo, 0, dim );

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

        dim = 1;
        cart.crossingDistance(dim, threadID, rayInfo, position[dim], direction[dim], distance);
        {
            singleDimRayTraceMap_t distances( rayInfo, 0, dim );

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

    }

    TEST( crossingDistance_2D_start_outside_on_an_external_corner_posX_posY ) {
        Position_t position ( -1.0, -1.0, 0.5 );
        Position_t direction(  1.0,  1.0, 0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        MonteRay_CartesianGrid cart(3, 
            MonteRay_GridBins{0, 3, 3},
            MonteRay_GridBins{0, 3, 3},
            MonteRay_GridBins{0, 3, 3});


        unsigned dim = 0;
        unsigned threadID=0;
        RayWorkInfo rayInfo(1,true);
        cart.crossingDistance(dim, threadID, rayInfo, position[dim], direction[dim], distance);

        {
            singleDimRayTraceMap_t distances( rayInfo, 0, dim );

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

        dim = 1;
        cart.crossingDistance(dim, threadID, rayInfo, position[dim], direction[dim], distance);

        {
            singleDimRayTraceMap_t distances( rayInfo, 0, dim );
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
    }

    TEST( crossingDistance_2D_start_outside_an_external_corner_negX_negY ) {
        Position_t position (  4.0,  4.0, 0.5 );
        Position_t direction( -1.0, -1.0, 0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;


        MonteRay_CartesianGrid cart(3, 
            MonteRay_GridBins{0, 3, 3},
            MonteRay_GridBins{0, 3, 3},
            MonteRay_GridBins{0, 3, 3});

        unsigned dim = 0;
        unsigned threadID=0;
        RayWorkInfo rayInfo(1,true);
        cart.crossingDistance(dim, threadID, rayInfo, position[dim], direction[dim], distance);

        {
            singleDimRayTraceMap_t distances( rayInfo, 0, dim );
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

        dim = 1;
        cart.crossingDistance(dim, threadID, rayInfo, position[dim], direction[dim], distance);

        {
            singleDimRayTraceMap_t distances( rayInfo, 0, dim );
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

}

