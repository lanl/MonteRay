#include <UnitTest++.h>

#include "MonteRay_CartesianGrid.t.hh"
#include "MonteRay_SpatialGrid.hh"
#include "MonteRayVector3D.hh"
#include "MonteRay_GridBins.hh"
#include "MonteRayDefinitions.hh"
#include "MonteRayCopyMemory.t.hh"
#include "RayWorkInfo.hh"

namespace MonteRay_CartesianGrid_rayTrace_tests{

using namespace MonteRay;

SUITE( MonteRay_CartesianGrid_rayTrace_Tests) {

    typedef Vector3D<gpuRayFloat_t> Position_t;
    using GridBins_t = MonteRay_GridBins;
    enum coord {X,Y,Z,DIM};

    class gridTestData {
    public:

        gridTestData(){
            std::vector<gpuRayFloat_t> vertices{
                -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
                0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10};

            pGridInfo[X] = new GridBins_t();
            pGridInfo[Y] = new GridBins_t();
            pGridInfo[Z] = new GridBins_t();

            pGridInfo[X]->initialize( vertices );
            pGridInfo[Y]->initialize( vertices );
            pGridInfo[Z]->initialize( vertices );

        }
        ~gridTestData(){
            delete pGridInfo[X];
            delete pGridInfo[Y];
            delete pGridInfo[Z];
        }

        MonteRay_SpatialGrid::pArrayOfpGridInfo_t pGridInfo;
    };

    using CartesianGrid = MonteRay_CartesianGrid;

    TEST( CrossingDistance_in_1D_PosXDir ) {
        //CHECK(false);
        gridTestData data;
        CartesianGrid cart(3,data.pGridInfo);

        Position_t position ( -9.5, -9.5,  -9.5 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 1.0;


        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 2, distances.size() );
        CHECK_EQUAL( 0, distances.id(0) );
        CHECK_CLOSE( 0.5, distances.dist(0), 1e-6 );
        CHECK_EQUAL( 1, distances.id(1) );
        CHECK_CLOSE( 0.5, distances.dist(1), 1e-6 );
    }


    TEST( rayTrace_in_1D_NegXDir ) {
        gridTestData data;
        CartesianGrid cart(3,data.pGridInfo);

        Position_t position ( -8.5, -9.5,  -9.5 );
        Position_t direction(    -1,   0,    0 );
        gpuRayFloat_t distance = 1.0;

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 2, distances.size() );
        CHECK_EQUAL( 1, distances.id(0) );
        CHECK_CLOSE( 0.5, distances.dist(0), 1e-6 );
        CHECK_EQUAL( 0, distances.id(1) );
        CHECK_CLOSE( 0.5, distances.dist(1), 1e-6 );
    }

    TEST( rayTrace_Outside_negSide_negDir ) {
        gridTestData data;
        CartesianGrid cart(3,data.pGridInfo);

        Position_t position ( -10.5, 0.5,  0.5 );
        Position_t direction(    -1,   0,    0 );
        gpuRayFloat_t distance = 2.0;

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL(  0,  distances.size() );
    }

    TEST( rayTrace_Outside_posSide_posDir ) {
        gridTestData data;
        CartesianGrid cart(3,data.pGridInfo);

        Position_t position (  10.5, 0.5,  0.5 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 2.0;

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL(  0, distances.size() );
    }

    TEST( rayTrace_Outside_negSide_posDir ) {
        gridTestData data;
        CartesianGrid cart(3,data.pGridInfo);

        Position_t position ( -10.5, -9.5,  -9.5 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 2.0;

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL(  2, distances.size() );

        CHECK_CLOSE(  0,   distances.id(0), 1e-6 );
        CHECK_CLOSE( 1.0 , distances.dist(0), 1e-6 );
        CHECK_CLOSE(  1,   distances.id(1), 1e-6 );
        CHECK_CLOSE( 0.5 , distances.dist(1), 1e-6 );
    }

    TEST( rayTrace_Outside_posSide_negDir ) {
        // std::cout << "Debug: ---------------------------------------" << std::endl;
        gridTestData data;
        CartesianGrid cart(3,data.pGridInfo);

        Position_t position (  10.5, -9.5,  -9.5 );
        Position_t direction(    -1,   0,    0 );
        gpuRayFloat_t distance = 2.0;

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL(  2, distances.size() );
        CHECK_CLOSE( 19,   distances.id(0), 1e-6 );
        CHECK_CLOSE( 1.0 , distances.dist(0), 1e-6 );
        CHECK_CLOSE( 18,   distances.id(1), 1e-6 );
        CHECK_CLOSE( 0.5 , distances.dist(1), 1e-6 );
    }

    TEST( rayTrace_Crossing_entire_grid_starting_outside_finish_outside_pos_dir ) {
        gridTestData data;
        CartesianGrid cart(3,data.pGridInfo);

        Position_t position (  -10.5, -9.5,  -9.5 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 21.0;

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL(  20, distances.size() );
        CHECK_CLOSE(  0,  distances.id(0), 1e-6 );
        CHECK_CLOSE( 1.0, distances.dist(0), 1e-6 );
        CHECK_CLOSE(  1,  distances.id(1), 1e-6 );
        CHECK_CLOSE( 1.0, distances.dist(1), 1e-6 );
        CHECK_CLOSE( 17,   distances.id(17), 1e-6 );
        CHECK_CLOSE( 1.0, distances.dist(17), 1e-6 );
        CHECK_CLOSE( 18,   distances.id(18), 1e-6 );
        CHECK_CLOSE( 1.0, distances.dist(18), 1e-6 );
        CHECK_CLOSE( 19,   distances.id(19), 1e-6 );
        CHECK_CLOSE( 1.0, distances.dist(19), 1e-6 );
    }
    TEST( rayTrace_Inside_cross_out_negDir ) {
        gridTestData data;
        CartesianGrid cart(3,data.pGridInfo);

        Position_t position (  -8.5, -9.5,  -9.5 );
        Position_t direction(    -1,   0,    0 );
        gpuRayFloat_t distance = 2.0;

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL(  2, distances.size() );
        CHECK_CLOSE(   1, distances.id(0), 1e-6 );
        CHECK_CLOSE( 0.5, distances.dist(0), 1e-6 );
        CHECK_CLOSE(   0, distances.id(1), 1e-6 );
        CHECK_CLOSE( 1.0, distances.dist(1), 1e-6 );
    }
    TEST( rayTrace_cross_out_posDir ) {
        gridTestData data;
        CartesianGrid cart(3,data.pGridInfo);

        Position_t position (  8.5, -9.5, -9.5 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 2.0;

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL(  2, distances.size() );
        CHECK_CLOSE(  18, distances.id(0), 1e-6 );
        CHECK_CLOSE( 0.5, distances.dist(0), 1e-6 );
        CHECK_CLOSE(  19, distances.id(1), 1e-6 );
        CHECK_CLOSE( 1.0, distances.dist(1), 1e-6 );
    }

    class gridInfoTestData {
    public:
        gridInfoTestData(){

            data[X] = new GridBins_t();
            data[Y] = new GridBins_t();
            data[Z] = new GridBins_t();

        }
        ~gridInfoTestData(){
            delete data[X];
            delete data[Y];
            delete data[Z];
        }

        void initialize( coord d, gpuRayFloat_t min, gpuRayFloat_t max, unsigned nBins ){
            data[d]->initialize( min, max, nBins );
        }

        MonteRay_SpatialGrid::pArrayOfpGridInfo_t data;
    };


    TEST( rayTrace_2D_internal_to_external_posX_posY ) {
        // std::cout << "Debug: ---------------------------------------" << std::endl;

        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X, -1, 1, 2);
        pGridInfo.initialize( Y, -1, 1, 2);
        pGridInfo.initialize( Z, -1, 1, 2);

        Position_t position (  -0.5, -.25, -0.5 );
        Position_t direction(    1,   1,    0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 3, distances.size() );
        CHECK_CLOSE( 0, distances.id(0), 1e-6 );
        CHECK_CLOSE( 0.25*std::sqrt(2.0), distances.dist(0), 1e-6 );
        CHECK_CLOSE( 2, distances.id(1), 1e-6 );
        CHECK_CLOSE( (0.25)*std::sqrt(2.0), distances.dist(1), 1e-6 );
        CHECK_CLOSE( 3, distances.id(2), 1e-6 );
        CHECK_CLOSE( (0.75)*std::sqrt(2.0), distances.dist(2), 1e-6 );
    }

    TEST( rayTrace_2D_internal_to_external_negX_negY ) {
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,-1, 1, 2);
        pGridInfo.initialize( Y,-1, 1, 2);
        pGridInfo.initialize( Z,-1, 1, 2);

        Position_t position (  0.25, 0.5, -0.5 );
        Position_t direction(  -1.0, -1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 3, distances.size() );
        CHECK_CLOSE( 3, distances.id(0), 1e-6 );
        CHECK_CLOSE( 0.25*std::sqrt(2.0), distances.dist(0), 1e-6 );
        CHECK_CLOSE( 2, distances.id(1), 1e-6 );
        CHECK_CLOSE( (0.25)*std::sqrt(2.0), distances.dist(1), 1e-6 );
        CHECK_CLOSE( 0, distances.id(2), 1e-6 );
        CHECK_CLOSE( (0.75)*std::sqrt(2.0), distances.dist(2), 1e-6 );
    }

    TEST( rayTrace_2D_internal_to_internal_posX_posY ) {
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,-1, 1, 2);
        pGridInfo.initialize( Y,-1, 1, 2);
        pGridInfo.initialize( Z,-1, 1, 2);

        Position_t position (  -0.5, -.25, -0.5 );
        Position_t direction(    1,   1,    0 );
        direction.normalize();
        gpuRayFloat_t distance = (0.5+0.25+0.25)*std::sqrt(2.0);

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 3, distances.size() );
        CHECK_CLOSE( 0, distances.id(0), 1e-6 );
        CHECK_CLOSE( 0.25*std::sqrt(2.0), distances.dist(0), 1e-6 );
        CHECK_CLOSE( 2, distances.id(1), 1e-6 );
        CHECK_CLOSE( (0.25)*std::sqrt(2.0), distances.dist(1), 1e-6 );
        CHECK_CLOSE( 3, distances.id(2), 1e-6 );
        CHECK_CLOSE( (0.5)*std::sqrt(2.0), distances.dist(2), 1e-6 );
    }
    TEST( rayTrace_2D_internal_to_internal_negX_negY ) {
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,-1, 1, 2);
        pGridInfo.initialize( Y,-1, 1, 2);
        pGridInfo.initialize( Z,-1, 1, 2);

        Position_t position (  0.25, 0.5, -0.5 );
        Position_t direction(  -1.0, -1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = (0.5+0.25+0.25)*std::sqrt(2.0);

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 3, distances.size() );
        CHECK_CLOSE( 3, distances.id(0), 1e-6 );
        CHECK_CLOSE( 0.25*std::sqrt(2.0), distances.dist(0), 1e-6 );
        CHECK_CLOSE( 2, distances.id(1), 1e-6 );
        CHECK_CLOSE( (0.25)*std::sqrt(2.0), distances.dist(1), 1e-6 );
        CHECK_CLOSE( 0, distances.id(2), 1e-6 );
        CHECK_CLOSE( (0.5)*std::sqrt(2.0), distances.dist(2), 1e-6 );
    }

    TEST( rayTrace_2D_external_to_external_posX_posY ) {
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,-1, 1, 2);
        pGridInfo.initialize( Y,-1, 1, 2);
        pGridInfo.initialize( Z,-1, 1, 2);

        Position_t position (  -1.5, -1.25, -0.5 );
        Position_t direction(  1.0, 1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 3, distances.size() );
        CHECK_CLOSE( 0, distances.id(0), 1e-6 );
        CHECK_CLOSE( (0.75)*std::sqrt(2.0), distances.dist(0), 1e-6 );
        CHECK_CLOSE( 2, distances.id(1), 1e-6 );
        CHECK_CLOSE( (0.25)*std::sqrt(2.0), distances.dist(1), 1e-6 );
        CHECK_CLOSE( 3, distances.id(2), 1e-6 );
        CHECK_CLOSE( (0.75)*std::sqrt(2.0), distances.dist(2), 1e-6 );
    }
    TEST( rayTrace_2D_external_to_external_negX_negY ) {
        // std::cout << "Debug: ---------------------------------------" << std::endl;

        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,-1, 1, 2);
        pGridInfo.initialize( Y,-1, 1, 2);
        pGridInfo.initialize( Z,-1, 1, 2);

        Position_t position (  1.25, 1.50, -0.5 );
        Position_t direction(  -1.0, -1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 3, distances.size() );
        CHECK_CLOSE( 3, distances.id(0), 1e-6 );
        CHECK_CLOSE( (0.75)*std::sqrt(2.0), distances.dist(0), 1e-6 );
        CHECK_CLOSE( 2, distances.id(1), 1e-6 );
        CHECK_CLOSE( (0.25)*std::sqrt(2.0), distances.dist(1), 1e-6 );
        CHECK_CLOSE( 0, distances.id(2), 1e-6 );
        CHECK_CLOSE( (0.75)*std::sqrt(2.0), distances.dist(2), 1e-6 );
    }

    TEST( rayTrace_2D_external_miss_posXDir ) {
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,-1, 1, 2);
        pGridInfo.initialize( Y,-1, 1, 2);
        pGridInfo.initialize( Z,-1, 1, 2);

        Position_t position (  -1.5, -.5, -1.5 );
        Position_t direction(  1.0,  0.0,  0.0 );
        gpuRayFloat_t distance = 10.0;

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 0, distances.size() );
    }

    TEST( rayTrace_2D_external_miss_negXDir ) {
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,-1, 1, 2);
        pGridInfo.initialize( Y,-1, 1, 2);
        pGridInfo.initialize( Z,-1, 1, 2);

        Position_t position (  1.5, -.5, -1.5 );
        Position_t direction(  1.0,  0.0,  0.0 );
        gpuRayFloat_t distance = 10.0;

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 0, distances.size() );
    }

    TEST( rayTrace_2D_internal_hit_corner_posXDir_posYDir ) {
        //         std::cout << "Debug: ---------------------------------------" << std::endl;
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,-1, 1, 2);
        pGridInfo.initialize( Y,-1, 1, 2);
        pGridInfo.initialize( Z,-1, 1, 2);

        Position_t position (  -.5, -.5, -.5 );
        Position_t direction(  1.0,  1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 1.0*std::sqrt(2.0);

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 3, distances.size() );
        CHECK_CLOSE( 0, distances.id(0), 1e-6 );
        CHECK_CLOSE( (0.5)*std::sqrt(2.0), distances.dist(0), 1e-6 );
        CHECK_CLOSE( 3, distances.id(2), 1e-6 );
        CHECK_CLOSE( (0.5)*std::sqrt(2.0), distances.dist(2), 1e-6 );
    }

    TEST( rayTrace_2D_internal_hit_corner_negXDir_negYDir ) {
        //         std::cout << "Debug: ---------------------------------------" << std::endl;
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,-1, 1, 2);
        pGridInfo.initialize( Y,-1, 1, 2);
        pGridInfo.initialize( Z,-1, 1, 2);

        Position_t position (  .5, .5, -.5 );
        Position_t direction(  -1.0,  -1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 1.0*std::sqrt(2.0);

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 3, distances.size() );
        CHECK_CLOSE( 3, distances.id(0), 1e-6 );
        CHECK_CLOSE( (0.5)*std::sqrt(2.0), distances.dist(0), 1e-6 );
        CHECK_CLOSE( 0, distances.id(2), 1e-6 );
        CHECK_CLOSE( (0.5)*std::sqrt(2.0), distances.dist(2), 1e-6 );
    }

    TEST( rayTrace_2D_posX_start_on_internal_gridline ) {
        //        std::cout << "Debug: ---------------------------------------" << std::endl;
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,-1, 1, 2);
        pGridInfo.initialize( Y,-1, 1, 2);
        pGridInfo.initialize( Z,-1, 1, 2);

        Position_t position (   0.0, -.5, -.5 );
        Position_t direction(   1.0,  0.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 1.5;

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 1, distances.size() );
        CHECK_CLOSE( 1, distances.id(0), 1e-6 );
        CHECK_CLOSE( 1.0, distances.dist(0), 1e-6 );
    }

    TEST( rayTrace_2D_negX_start_on_internal_gridline ) {
        //         std::cout << "Debug: ---------------------------------------" << std::endl;
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,-1, 1, 2);
        pGridInfo.initialize( Y,-1, 1, 2);
        pGridInfo.initialize( Z,-1, 1, 2);

        Position_t position (   0.0, -.5, -.5 );
        Position_t direction(  -1.0,  0.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 1.5;

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 2, distances.size() );
        CHECK_CLOSE( 0, distances.id(1), 1e-6 );
        CHECK_CLOSE( 1.0, distances.dist(1), 1e-6 );
    }

    TEST( rayTrace_2D_posX_start_on_external_boundary_gridline ) {
        //         std::cout << "Debug: ---------------------------------------" << std::endl;
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,-1, 1, 2);
        pGridInfo.initialize( Y,-1, 1, 2);
        pGridInfo.initialize( Z,-1, 1, 2);

        Position_t position ( -1.0, -.5, -.5 );
        Position_t direction(  1.0,  0.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 1.5;

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 2, distances.size() );
        CHECK_CLOSE( 0, distances.id(0), 1e-6 );
        CHECK_CLOSE( 1.0, distances.dist(0), 1e-6 );
        CHECK_CLOSE( 1, distances.id(1), 1e-6 );
        CHECK_CLOSE( 0.5, distances.dist(1), 1e-6 );
    }

    TEST( rayTrace_2D_negX_start_on_external_boundary_gridline ) {
        //         std::cout << "Debug: ---------------------------------------" << std::endl;
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,-1, 1, 2);
        pGridInfo.initialize( Y,-1, 1, 2);
        pGridInfo.initialize( Z,-1, 1, 2);

        Position_t position (  1.0, -.5, -.5 );
        Position_t direction( -1.0,  0.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 1.5;

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 2, distances.size() );
        CHECK_CLOSE( 1, distances.id(0), 1e-6 );
        CHECK_CLOSE( 1.0, distances.dist(0), 1e-6 );
        CHECK_CLOSE( 0, distances.id(1), 1e-6 );
        CHECK_CLOSE( 0.5, distances.dist(1), 1e-6 );
    }

    TEST( rayTrace_2D_start_on_an_internal_corner_posX_posY ) {
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,0, 3, 3);
        pGridInfo.initialize( Y,0, 3, 3);
        pGridInfo.initialize( Z,0, 3, 3);

        Position_t position (  1.0, 1.0, 0.5 );
        Position_t direction(  1.0, 1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 3, distances.size() );
        CHECK_CLOSE( 4, distances.id(0), 1e-6 );
        CHECK_CLOSE( 1.0*std::sqrt(2.0), distances.dist(0), 1e-6 );
        CHECK_CLOSE( 5, distances.id(1), 1e-6 );
        CHECK_CLOSE( 0, distances.dist(1), 1e-6 );
        CHECK_CLOSE( 8, distances.id(2), 1e-6 );
        CHECK_CLOSE( 1.0*std::sqrt(2.0), distances.dist(2), 1e-6 );
    }


    const gpuRayFloat_t s2 = std::sqrt(2.0);
    void checkDistances( const std::vector<unsigned>& expectedIndex,
            const std::vector<gpuRayFloat_t>& expectedDistance, const rayTraceList_t& distances )
    {
        CHECK_EQUAL( expectedIndex.size(), expectedDistance.size() );
        CHECK_EQUAL( expectedIndex.size(), distances.size() );
        for( auto i=0; i<distances.size(); ++i ) {
            CHECK_EQUAL( expectedIndex   [i], distances.id(i) );
            CHECK_CLOSE( expectedDistance[i], distances.dist(i), 1e-6 );
        }
    }


    TEST( rayTrace_2D_start_on_an_internal_corner_negX_negY ) {
        //         std::cout << "Debug: ---------------------------------------" << std::endl;
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,0, 3, 3);
        pGridInfo.initialize( Y,0, 3, 3);
        pGridInfo.initialize( Z,0, 3, 3);

        Position_t position (  2.0, 2.0, 0.5 );
        Position_t direction(  -1.0, -1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 5, distances.size() );
        std::vector<unsigned> expectedIndex{ 8, 7, 4, 3, 0 };
        std::vector<gpuRayFloat_t> expectedDistance{ 0, 0, s2, 0, s2 };
        checkDistances( expectedIndex, expectedDistance, distances );
    }

    TEST( rayTrace_2D_start_on_an_internal_corner_posX_negY ) {
        //         std::cout << "Debug: ---------------------------------------" << std::endl;
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,0, 3, 3);
        pGridInfo.initialize( Y,0, 3, 3);
        pGridInfo.initialize( Z,0, 3, 3);

        Position_t position (   1.0, 2.0, 0.5 );
        Position_t direction(   1.0, -1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 4, distances.size() );
        checkDistances( {7,4,5,2}, {0,s2,0,s2}, distances );
    }
    TEST( rayTrace_2D_start_on_an_internal_corner_negX_posY ) {
        //         std::cout << "Debug: ---------------------------------------" << std::endl;
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,0, 3, 3);
        pGridInfo.initialize( Y,0, 3, 3);
        pGridInfo.initialize( Z,0, 3, 3);

        Position_t position (   2.0, 1.0, 0.5 );
        Position_t direction(  -1.0, 1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 4, distances.size() );
        checkDistances( {5,4,3,6}, {0,s2,0,s2}, distances );
    }

    TEST( rayTrace_2D_start_on_an_external_corner_posX_posY ) {
        //        std::cout << "Debug: ---------------------------------------" << std::endl;
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,0, 3, 3);
        pGridInfo.initialize( Y,0, 3, 3);
        pGridInfo.initialize( Z,0, 3, 3);

        Position_t position (   0.0, 0.0, 0.5 );
        Position_t direction(   1.0, 1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 5, distances.size() );
        checkDistances( {0,1,4,5,8}, {s2,0,s2,0,s2}, distances );
    }
    TEST( rayTrace_2D_start_on_an_external_corner_negX_negY ) {
        //         std::cout << "Debug: ---------------------------------------" << std::endl;
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,0, 3, 3);
        pGridInfo.initialize( Y,0, 3, 3);
        pGridInfo.initialize( Z,0, 3, 3);

        Position_t position (   3.0,  3.0, 0.5 );
        Position_t direction(  -1.0, -1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 5, distances.size() );
        checkDistances( {8,7,4,3,0}, {s2,0,s2,0,s2}, distances );
    }
    TEST( rayTrace_2D_start_on_an_external_corner_negX_posY ) {
        //         std::cout << "Debug: ---------------------------------------" << std::endl;
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,0, 3, 3);
        pGridInfo.initialize( Y,0, 3, 3);
        pGridInfo.initialize( Z,0, 3, 3);

        Position_t position (   3.0,  0.0, 0.5 );
        Position_t direction(  -1.0,  1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 5, distances.size() );
        checkDistances( {2,1,4,3,6}, {s2,0,s2,0,s2}, distances );
    }
    TEST( rayTrace_2D_start_on_an_external_corner_posX_negY ) {
        //         std::cout << "Debug: ---------------------------------------" << std::endl;
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,0, 3, 3);
        pGridInfo.initialize( Y,0, 3, 3);
        pGridInfo.initialize( Z,0, 3, 3);

        Position_t position (   0.0,  3.0, 0.5 );
        Position_t direction(   1.0, -1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 5, distances.size() );
        checkDistances( {6,7,4,5,2}, {s2,0,s2,0,s2}, distances );
    }
    const gpuRayFloat_t s3 = std::sqrt(3.0);
    TEST( rayTrace_3D_start_on_an_external_corner_posX_posY_posZ ) {
        //         std::cout << "Debug: ---------------------------------------" << std::endl;
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,0, 3, 3);
        pGridInfo.initialize( Y,0, 3, 3);
        pGridInfo.initialize( Z,0, 3, 3);

        Position_t position (   0.0, 0.0, 0.0 );
        Position_t direction(   1.0, 1.0, 1.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 7, distances.size() );
        checkDistances( {0,1,4,13,14,17,26}, {s3,0,0,s3,0,0,s3}, distances );
    }
    TEST( rayTrace_2D_start_outside_an_external_corner_posX_posY ) {
        //        std::cout << "Debug: ---------------------------------------" << std::endl;
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,0, 3, 3);
        pGridInfo.initialize( Y,0, 3, 3);
        pGridInfo.initialize( Z,0, 3, 3);

        Position_t position (  -1.0, -1.0, 0.5 );
        Position_t direction(   1.0, 1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 5, distances.size() );
        checkDistances( {0,1,4,5,8}, {s2,0,s2,0,s2}, distances );
    }

    TEST( rayTrace_2D_start_outside_an_external_corner_posX_negY ) {
        //        std::cout << "Debug: ---------------------------------------" << std::endl;
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,0, 3, 3);
        pGridInfo.initialize( Y,0, 3, 3);
        pGridInfo.initialize( Z,0, 3, 3);

        Position_t position (  -1.0, 4.0, 0.5 );
        Position_t direction(   1.0,-1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 5, distances.size() );
        checkDistances( {6,7,4,5,2}, {s2,0,s2,0,s2}, distances );
    }
    TEST( rayTrace_2D_start_outside_an_external_corner_negX_posY ) {
        //        std::cout << "Debug: ---------------------------------------" << std::endl;
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,0, 3, 3);
        pGridInfo.initialize( Y,0, 3, 3);
        pGridInfo.initialize( Z,0, 3, 3);

        Position_t position (   4.0,-1.0, 0.5 );
        Position_t direction(  -1.0, 1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 5, distances.size() );
        checkDistances( {2,1,4,3,6}, {s2,0,s2,0,s2}, distances );
    }

    TEST( rayTrace_2D_start_outside_an_external_corner_negX_negY ) {
        //        std::cout << "Debug: ---------------------------------------" << std::endl;
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,0, 3, 3);
        pGridInfo.initialize( Y,0, 3, 3);
        pGridInfo.initialize( Z,0, 3, 3);

        Position_t position (   4.0,  4.0, 0.5 );
        Position_t direction(  -1.0, -1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 5, distances.size() );
        checkDistances( {8,7,4,3,0}, {s2,0,s2,0,s2}, distances );
    }

    TEST( rayTrace_3D_start_outside_an_external_corner_posX_posY_posZ ) {
        // std::cout << "Debug: ---------------------------------------" << std::endl;
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,0, 3, 3);
        pGridInfo.initialize( Y,0, 3, 3);
        pGridInfo.initialize( Z,0, 3, 3);

        Position_t position (  -1.0, -1.0, -1.0 );
        Position_t direction(   1.0, 1.0, 1.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 7, distances.size() );
        checkDistances( {0,1,4,13,14,17,26}, {s3,0,0,s3,0,0,s3}, distances );
    }

    TEST( rayTrace_3D_start_outside_an_external_corner_negX_negY_negZ ) {
        // std::cout << "Debug: ---------------------------------------" << std::endl;
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,0, 3, 3);
        pGridInfo.initialize( Y,0, 3, 3);
        pGridInfo.initialize( Z,0, 3, 3);

        Position_t position (    4.0,  4.0,  4.0 );
        Position_t direction(   -1.0, -1.0, -1.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 7, distances.size() );
        checkDistances( {26,25,22,13,12,9,0}, {s3,0,0,s3,0,0,s3}, distances );
    }

    TEST( rayTrace_3D_start_outside_an_external_corner_negX_negY_negZ_wOutsideDistance ) {
        // std::cout << "Debug: ---------------------------------------" << std::endl;
        gridInfoTestData pGridInfo;
        pGridInfo.initialize( X,0, 3, 3);
        pGridInfo.initialize( Y,0, 3, 3);
        pGridInfo.initialize( Z,0, 3, 3);

        Position_t position (    4.0,  4.0,  4.0 );
        Position_t direction(   -1.0, -1.0, -1.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        CartesianGrid cart(3,pGridInfo.data);

        RayWorkInfo rayInfo(1,true);
        cart.rayTrace(0, rayInfo, position, direction, distance, true);
        rayTraceList_t distances( rayInfo, 0 );

        CHECK_EQUAL( 10, distances.size() );
        unsigned maxuint = 4294967295;
        checkDistances( {maxuint,maxuint,maxuint,26,25,22,13,12,9,0}, {s3,0,0,s3,0,0,s3,0,0,s3}, distances );
    }

}

}

