#include <UnitTest++.h>

#include <iostream>
#include <cmath>
#include <iomanip>

#include "MonteRayDefinitions.hh"
#include "GPUUtilityFunctions.hh"

#include "gpuDistanceCalculator_test_helper.hh"
#include "RayWorkInfo.hh"
#include "MonteRay_SpatialGrid.hh"


namespace MonteRay_DistanceCalculatorTester {

using namespace MonteRay;
using Position_t = MonteRay_SpatialGrid::Position_t;
using Direction_t = MonteRay_SpatialGrid::Direction_t;

SUITE( DistanceCalculatorGPU_Tester ) {

    TEST( setup ) {
        //gpuCheck();
    }

    class DistanceCalculatorGPUTest	{
    public:

        std::unique_ptr<MonteRay_SpatialGrid> pGrid;

        DistanceCalculatorGPUTest(){

            pGrid = std::make_unique<MonteRay_SpatialGrid>(TransportMeshType::Cartesian, 
              std::array<MonteRay_GridBins, 3>{
              MonteRay_GridBins{0.0, 10.0, 10},
              MonteRay_GridBins{0.0, 10.0, 10},
              MonteRay_GridBins{0.0, 10.0, 10} }
            );
        }

    };

    TEST_FIXTURE(DistanceCalculatorGPUTest, rayTrace_outside_to_inside_posDir_one_crossing ) {
        gpuDistanceCalculatorTestHelper tester;
        Position_t pos( -0.5, 0.5, 0.5 );
        Direction_t dir( 1.0, 0.0, 0.0);
        gpuRayFloat_t distance = 2.0;

        tester.setupTimers();
        tester.launchRayTrace(pos,dir,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 2, tester.pRayInfo->getRayCastSize(0));
        CHECK_EQUAL( 2, numCrossings);
        CHECK_EQUAL( 0U, cells[0]);
        CHECK_CLOSE( 1.0f, distances[0], 1e-11 );
        CHECK_EQUAL( 1U, cells[1]);
        CHECK_CLOSE( 0.5f, distances[1], 1e-11 );
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest, rayTrace_outside_to_inside_posDir_outsideDistances ) {
        gpuDistanceCalculatorTestHelper tester;
        Position_t pos( -0.5, 0.5, 0.5 );
        Direction_t dir( 1.0, 0.0, 0.0);
        gpuRayFloat_t distance = 2.0;

        tester.setupTimers();
        tester.launchRayTrace(pos,dir,distance,true,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 3, numCrossings);
        CHECK_EQUAL( -1, cells[0]);
        CHECK_CLOSE( 0.5f, distances[0], 1e-11 );
        CHECK_EQUAL(  0, cells[1]);
        CHECK_CLOSE( 1.0f, distances[1], 1e-11 );
        CHECK_EQUAL(  1, cells[2]);
        CHECK_CLOSE( 0.5f, distances[2], 1e-11 );
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest, rayTrace_inside_to_outside_posDir_one_crossing ) {
        gpuDistanceCalculatorTestHelper tester;
        Position_t pos(  8.5, 0.5, 0.5 );
        Direction_t dir( 1.0, 0.0, 0.0);
        gpuRayFloat_t distance = 2.0;

        tester.setupTimers();
        tester.launchRayTrace(pos,dir,distance,true,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 2, numCrossings);
        CHECK_EQUAL( 8U, cells[0]);
        CHECK_CLOSE( 0.5f, distances[0], 1e-11 );
        CHECK_EQUAL( 9U, cells[1]);
        CHECK_CLOSE( 1.0f, distances[1], 1e-11 );
    }

    class DistanceCalculatorGPUTest2	{
    public:

        std::unique_ptr<MonteRay_SpatialGrid> pGrid;

        DistanceCalculatorGPUTest2(){
            pGrid = std::make_unique<MonteRay_SpatialGrid>(TransportMeshType::Cartesian, 
                std::array<MonteRay_GridBins, 3>{
                MonteRay_GridBins{-10.0, 10.0, 20},
                MonteRay_GridBins{-10.0, 10.0, 20},
                MonteRay_GridBins{-10.0, 10.0, 20} }
            );
        }
    };

    TEST_FIXTURE(DistanceCalculatorGPUTest2, rayTrace_in_1D_PosXDir ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position ( -9.5, -9.5,  -9.5 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 1.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,true,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 2, numCrossings);
        CHECK_EQUAL( 0, cells[0]);
        CHECK_CLOSE( 0.5f, distances[0], 1e-11 );
        CHECK_EQUAL( 1, cells[1]);
        CHECK_CLOSE( 0.5f, distances[1], 1e-11 );
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest2, rayTrace_in_1D_NegXDir ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position ( -8.5, -9.5,  -9.5 );
        Position_t direction(    -1,   0,    0 );
        gpuRayFloat_t distance = 1.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,true,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 2, numCrossings);
        CHECK_EQUAL( 1, cells[0]);
        CHECK_CLOSE( 0.5f, distances[0], 1e-11 );
        CHECK_EQUAL( 0, cells[1]);
        CHECK_CLOSE( 0.5f, distances[1], 1e-11 );
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest2, rayTrace_Outside_negSide_negDir ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position ( -10.5, 0.5,  0.5 );
        Position_t direction(    -1,   0,    0 );
        gpuRayFloat_t distance = 2.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,true,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 0, numCrossings);
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest2, rayTrace_Outside_posSide_posDir ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (  10.5, 0.5,  0.5 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 2.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,true,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 0, numCrossings);
    }
    TEST_FIXTURE(DistanceCalculatorGPUTest2, rayTrace_Outside_negSide_posDir ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position ( -10.5, -9.5,  -9.5 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 2.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 2, numCrossings);
        CHECK_EQUAL( 0, cells[0]);
        CHECK_CLOSE( 1.0f, distances[0], 1e-11 );
        CHECK_EQUAL( 1, cells[1]);
        CHECK_CLOSE( 0.5f, distances[1], 1e-11 );
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest2, rayTrace_Outside_posSide_negDir ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (  10.5, -9.5,  -9.5 );
        Position_t direction(    -1,   0,    0 );
        gpuRayFloat_t distance = 2.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 2, numCrossings);
        CHECK_EQUAL( 19, cells[0]);
        CHECK_CLOSE( 1.0f, distances[0], 1e-11 );
        CHECK_EQUAL( 18, cells[1]);
        CHECK_CLOSE( 0.5f, distances[1], 1e-11 );
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest2, rayTrace_Crossing_entire_grid_starting_outside_finish_outside_pos_dir ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (  -10.5, -9.5,  -9.5 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 21.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 20, numCrossings);
        CHECK_EQUAL( 0, cells[0]);
        CHECK_CLOSE( 1.0f, distances[0], 1e-11 );
        CHECK_EQUAL( 1, cells[1]);
        CHECK_CLOSE( 1.0f, distances[1], 1e-11 );
        CHECK_EQUAL( 17, cells[17]);
        CHECK_CLOSE( 1.0f, distances[1], 1e-11 );
        CHECK_EQUAL( 18, cells[18]);
        CHECK_CLOSE( 1.0f, distances[1], 1e-11 );
        CHECK_EQUAL( 19, cells[19]);
        CHECK_CLOSE( 1.0f, distances[1], 1e-11 );
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest2, rayTrace_Inside_cross_out_negDir ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (  -8.5, -9.5,  -9.5 );
        Position_t direction(    -1,   0,    0 );
        gpuRayFloat_t distance = 2.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 2, numCrossings);
        CHECK_EQUAL( 1, cells[0]);
        CHECK_CLOSE( 0.5f, distances[0], 1e-11 );
        CHECK_EQUAL( 0, cells[1]);
        CHECK_CLOSE( 1.0f, distances[1], 1e-11 );
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest2, rayTrace_cross_out_posDir ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (  8.5, -9.5, -9.5 );
        Position_t direction(    1,   0,    0 );
        gpuRayFloat_t distance = 2.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 2, numCrossings);
        CHECK_EQUAL( 18, cells[0]);
        CHECK_CLOSE( 0.5f, distances[0], 1e-11 );
        CHECK_EQUAL( 19, cells[1]);
        CHECK_CLOSE( 1.0f, distances[1], 1e-11 );
    }

    class DistanceCalculatorGPUTest3	{
    public:

        std::unique_ptr<MonteRay_SpatialGrid> pGrid;

        DistanceCalculatorGPUTest3(){
            pGrid = std::make_unique<MonteRay_SpatialGrid>(TransportMeshType::Cartesian, 
                std::array<MonteRay_GridBins, 3>{
                MonteRay_GridBins{-1.0, 1.0, 2},
                MonteRay_GridBins{-1.0, 1.0, 2},
                MonteRay_GridBins{-1.0, 1.0, 2} }
            );
        }
    };

    TEST( reset2) {
        gpuReset();
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest3, rayTrace_2D_internal_to_external_posX_posY ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (  -0.5, -.25, -0.5 );
        Position_t direction(    1,   1,    0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        tester.pRayInfo->clear();
        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 3, numCrossings);
        CHECK_EQUAL( 0, cells[0]);
        CHECK_CLOSE( 0.25*std::sqrt(2.0), distances[0], 1e-6 );
        CHECK_EQUAL( 2, cells[1]);
        CHECK_CLOSE( 0.25*std::sqrt(2.0), distances[1], 1e-6 );
        CHECK_EQUAL( 3, cells[2]);
        CHECK_CLOSE( 0.75*std::sqrt(2.0), distances[2], 1e-6 );
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest3, rayTrace_2D_internal_to_external_negX_negY ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (  0.25, 0.5, -0.5 );
        Position_t direction(  -1.0, -1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 3, numCrossings);
        CHECK_EQUAL( 3, cells[0]);
        CHECK_CLOSE( 0.25*std::sqrt(2.0), distances[0], 1e-6 );
        CHECK_EQUAL( 2, cells[1]);
        CHECK_CLOSE( 0.25*std::sqrt(2.0), distances[1], 1e-6 );
        CHECK_EQUAL( 0, cells[2]);
        CHECK_CLOSE( 0.75*std::sqrt(2.0), distances[2], 1e-6 );
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest3, rayTrace_2D_internal_to_internal_posX_posY ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (  -0.5, -.25, -0.5 );
        Position_t direction(    1,   1,    0 );
        direction.normalize();
        gpuRayFloat_t distance = (0.5+0.25+0.25)*std::sqrt(2.0);

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 3, numCrossings);
        CHECK_EQUAL( 0, cells[0]);
        CHECK_CLOSE( 0.25*std::sqrt(2.0), distances[0], 1e-6 );
        CHECK_EQUAL( 2, cells[1]);
        CHECK_CLOSE( 0.25*std::sqrt(2.0), distances[1], 1e-6 );
        CHECK_EQUAL( 3, cells[2]);
        CHECK_CLOSE( 0.50*std::sqrt(2.0), distances[2], 1e-6 );
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest3, rayTrace_2D_internal_to_internal_negX_negY ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (  0.25, 0.5, -0.5 );
        Position_t direction(  -1.0, -1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = (0.5+0.25+0.25)*std::sqrt(2.0);

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 3, numCrossings);
        CHECK_EQUAL( 3, cells[0]);
        CHECK_CLOSE( 0.25*std::sqrt(2.0), distances[0], 1e-6 );
        CHECK_EQUAL( 2, cells[1]);
        CHECK_CLOSE( 0.25*std::sqrt(2.0), distances[1], 1e-6 );
        CHECK_EQUAL( 0, cells[2]);
        CHECK_CLOSE( 0.50*std::sqrt(2.0), distances[2], 1e-6 );
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest3, rayTrace_2D_external_to_external_posX_posY ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (  -1.5, -1.25, -0.5 );
        Position_t direction(  1.0, 1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 3, numCrossings);
        CHECK_EQUAL( 0, cells[0]);
        CHECK_CLOSE( 0.75f*std::sqrt(2.0f), distances[0], 1e-7 );
        CHECK_EQUAL( 2, cells[1]);
        CHECK_CLOSE( 0.25f*std::sqrt(2.0f), distances[1], 1e-7 );
        CHECK_EQUAL( 3, cells[2]);
        CHECK_CLOSE( 0.75f*std::sqrt(2.0f), distances[2], 1e-7 );
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest3, rayTrace_2D_external_to_external_negX_negY ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (  1.25, 1.50, -0.5 );
        Position_t direction(  -1.0, -1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 3, numCrossings);
        CHECK_EQUAL( 3, cells[0]);
        CHECK_CLOSE( 0.75f*std::sqrt(2.0f), distances[0], 1e-7 );
        CHECK_EQUAL( 2, cells[1]);
        CHECK_CLOSE( 0.25f*std::sqrt(2.0f), distances[1], 1e-7 );
        CHECK_EQUAL( 0, cells[2]);
        CHECK_CLOSE( 0.75f*std::sqrt(2.0f), distances[2], 1e-7 );
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest3, rayTrace_2D_external_miss_posXDir ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (  -1.5, -.5, -1.5 );
        Position_t direction(  1.0,  0.0,  0.0 );
        gpuRayFloat_t distance = 10.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 0, numCrossings);
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest3, rayTrace_2D_external_miss_negXDir ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (  1.5, -.5, -1.5 );
        Position_t direction(  1.0,  0.0,  0.0 );
        gpuRayFloat_t distance = 10.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 0, numCrossings);
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest3, rayTrace_2D_internal_hit_corner_posXDir_posYDir ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (  -.5, -.5, -.5 );
        Position_t direction(  1.0,  1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 1.0*std::sqrt(2.0);

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 3, numCrossings);
        CHECK_EQUAL( 0, cells[0]);
        CHECK_CLOSE( 0.50f*std::sqrt(2.0f), distances[0], 1e-7 );
        CHECK_EQUAL( 1, cells[1]);
        CHECK_CLOSE( 0, distances[1], 1e-7 );
        CHECK_EQUAL( 3, cells[2]);
        CHECK_CLOSE( 0.50f*std::sqrt(2.0f), distances[2], 1e-7 );
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest3, rayTrace_2D_internal_hit_corner_negXDir_negYDir ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (  .5, .5, -.5 );
        Position_t direction(  -1.0,  -1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 1.0*std::sqrt(2.0);

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 3, numCrossings);
        CHECK_EQUAL( 3, cells[0]);
        CHECK_CLOSE( 0.50f*std::sqrt(2.0f), distances[0], 1e-7 );
        CHECK_EQUAL( 2, cells[1]);
        CHECK_CLOSE( 0, distances[1], 1e-7 );
        CHECK_EQUAL( 0, cells[2]);
        CHECK_CLOSE( 0.50f*std::sqrt(2.0f), distances[2], 1e-7 );
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest3, rayTrace_2D_posX_start_on_internal_gridline ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (   0.0, -.5, -.5 );
        Position_t direction(   1.0,  0.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 1.5;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 1, numCrossings);
        CHECK_EQUAL( 1, cells[0]);
        CHECK_CLOSE( 1.0f, distances[0], 1e-7 );
    }
    TEST_FIXTURE(DistanceCalculatorGPUTest3, rayTrace_2D_negX_start_on_internal_gridline ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (   0.0, -.5, -.5 );
        Position_t direction(  -1.0,  0.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 1.5;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 2, numCrossings);
        CHECK_EQUAL( 1, cells[0]);
        CHECK_CLOSE( 0, distances[0], 1e-7 );
        CHECK_EQUAL( 0, cells[1]);
        CHECK_CLOSE( 1.0f, distances[1], 1e-7 );
    }
    TEST_FIXTURE(DistanceCalculatorGPUTest3, rayTrace_2D_posX_start_on_external_boundary_gridline ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position ( -1.0, -.5, -.5 );
        Position_t direction(  1.0,  0.0,  0.0 );
        direction.normalize();
        double distance = 1.5;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 2, numCrossings);
        CHECK_EQUAL( 0, cells[0]);
        CHECK_CLOSE( 1.0f, distances[0], 1e-7 );
        CHECK_EQUAL( 1, cells[1]);
        CHECK_CLOSE( 0.5f, distances[1], 1e-7 );
    }
    TEST_FIXTURE(DistanceCalculatorGPUTest3, rayTrace_2D_negX_start_on_external_boundary_gridline ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (  1.0, -.5, -.5 );
        Position_t direction( -1.0,  0.0,  0.0 );
        direction.normalize();
        double distance = 1.5;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 2, numCrossings);
        CHECK_EQUAL( 1, cells[0]);
        CHECK_CLOSE( 1.0f, distances[0], 1e-7 );
        CHECK_EQUAL( 0, cells[1]);
        CHECK_CLOSE( 0.5f, distances[1], 1e-7 );
    }

    class DistanceCalculatorGPUTest4	{
    public:

        std::unique_ptr<MonteRay_SpatialGrid> pGrid;

        DistanceCalculatorGPUTest4(){
            pGrid = std::make_unique<MonteRay_SpatialGrid>(TransportMeshType::Cartesian, 
                std::array<MonteRay_GridBins, 3>{
                MonteRay_GridBins{0.0, 3.0, 3},
                MonteRay_GridBins{0.0, 3.0, 3},
                MonteRay_GridBins{0.0, 3.0, 3} }
            );
        }
    };

    TEST_FIXTURE(DistanceCalculatorGPUTest4, rayTrace_2D_start_on_an_internal_corner_posX_posY ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (  1.0, 1.0, 0.5 );
        Position_t direction(  1.0, 1.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 10.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 3, numCrossings);
        CHECK_EQUAL( 4, cells[0]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[0], 1e-7 );
        CHECK_EQUAL( 5, cells[1]);
        CHECK_CLOSE( 0.0f, distances[1], 1e-7 );
        CHECK_EQUAL( 8, cells[2]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[2], 1e-7 );
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest4, rayTrace_2D_start_on_an_internal_corner_negX_negY ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (  2.0,  2.0,  0.5 );
        Position_t direction( -1.0, -1.0,  0.0 );
        direction.normalize();
        double distance = 10.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 5, numCrossings);
        CHECK_EQUAL( 8, cells[0]);
        CHECK_CLOSE( 0.0f, distances[0], 1e-7 );

        CHECK_EQUAL( 7, cells[1]);
        CHECK_CLOSE( 0.0f, distances[1], 1e-7 );

        CHECK_EQUAL( 4, cells[2]);
        CHECK_CLOSE( std::sqrt(2.0), distances[2], 1e-7 );

        CHECK_EQUAL( 3, cells[3]);
        CHECK_CLOSE( 0.0f, distances[3], 1e-7 );

        CHECK_EQUAL( 0, cells[4]);
        CHECK_CLOSE( std::sqrt(2.0), distances[4], 1e-7 );

    }

    TEST_FIXTURE(DistanceCalculatorGPUTest4, rayTrace_2D_start_on_an_internal_corner_posX_negY ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (   1.0, 2.0, 0.5 );
        Position_t direction(   1.0, -1.0,  0.0 );
        direction.normalize();
        double distance = 10.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 4, numCrossings);
        CHECK_EQUAL( 7, cells[0]);
        CHECK_CLOSE( 0.0f, distances[0], 1e-7 );
        CHECK_EQUAL( 4, cells[1]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[1], 1e-7 );
        CHECK_EQUAL( 5, cells[2]);
        CHECK_CLOSE( 0.0f, distances[2], 1e-7 );
        CHECK_EQUAL( 2, cells[3]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[3], 1e-7 );
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest4, rayTrace_2D_start_on_an_internal_corner_negX_posY ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (   2.0, 1.0, 0.5 );
        Position_t direction(  -1.0, 1.0,  0.0 );
        direction.normalize();
        double distance = 10.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 4, numCrossings);
        CHECK_EQUAL( 5, cells[0]);
        CHECK_CLOSE( 0.0, distances[0], 1e-7 );
        CHECK_EQUAL( 4, cells[1]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[1], 1e-7 );
        CHECK_EQUAL( 3, cells[2]);
        CHECK_CLOSE( 0.0, distances[2], 1e-7 );
        CHECK_EQUAL( 6, cells[3]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[3], 1e-7 );
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest4, rayTrace_2D_start_on_an_external_corner_posX_posY ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (   0.0, 0.0, 0.5 );
        Position_t direction(   1.0, 1.0,  0.0 );
        direction.normalize();
        double distance = 10.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 5, numCrossings);
        CHECK_EQUAL( 0, cells[0]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[0], 1e-6 );
        CHECK_EQUAL( 1, cells[1]);
        CHECK_CLOSE( 0.0, distances[1], 1e-6 );
        CHECK_EQUAL( 4, cells[2]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[2], 1e-6 );
        CHECK_EQUAL( 5, cells[3]);
        CHECK_CLOSE( 0.0, distances[3], 1e-6 );
        CHECK_EQUAL( 8, cells[4]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[4], 1e-6 );
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest4, rayTrace_2D_start_on_an_external_corner_negX_negY ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (   3.0,  3.0, 0.5 );
        Position_t direction(  -1.0, -1.0,  0.0 );
        direction.normalize();
        double distance = 10.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 5, numCrossings);
        CHECK_EQUAL( 8, cells[0]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[0], 1e-6 );
        CHECK_EQUAL( 7, cells[1]);
        CHECK_CLOSE( 0.0, distances[1], 1e-6 );
        CHECK_EQUAL( 4, cells[2]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[2], 1e-6 );
        CHECK_EQUAL( 3, cells[3]);
        CHECK_CLOSE( 0.0, distances[3], 1e-6 );
        CHECK_EQUAL( 0, cells[4]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[4], 1e-6 );
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest4, rayTrace_2D_start_on_an_external_corner_negX_posY ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (   3.0,  0.0, 0.5 );
        Position_t direction(  -1.0,  1.0,  0.0 );
        direction.normalize();
        double distance = 10.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 5, numCrossings);
        CHECK_EQUAL( 2, cells[0]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[0], 1e-6 );
        CHECK_EQUAL( 1, cells[1]);
        CHECK_CLOSE( 0.0, distances[1], 1e-6 );
        CHECK_EQUAL( 4, cells[2]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[2], 1e-6 );
        CHECK_EQUAL( 3, cells[3]);
        CHECK_CLOSE( 0.0, distances[3], 1e-6 );
        CHECK_EQUAL( 6, cells[4]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[4], 1e-6 );
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest4, rayTrace_2D_start_on_an_external_corner_posX_negY ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (   0.0,  3.0, 0.5 );
        Position_t direction(   1.0, -1.0,  0.0 );
        direction.normalize();
        double distance = 10.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 5, numCrossings);
        CHECK_EQUAL( 6, cells[0]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[0], 1e-6 );
        CHECK_EQUAL( 7, cells[1]);
        CHECK_CLOSE( 0.0, distances[1], 1e-6 );
        CHECK_EQUAL( 4, cells[2]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[2], 1e-6 );
        CHECK_EQUAL( 5, cells[3]);
        CHECK_CLOSE( 0.0, distances[3], 1e-6 );
        CHECK_EQUAL( 2, cells[4]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[4], 1e-6 );
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest4, rayTrace_3D_start_on_an_external_corner_posX_posY_posZ ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (   0.0, 0.0, 0.0 );
        Position_t direction(   1.0, 1.0, 1.0 );
        direction.normalize();
        double distance = 10.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 7, numCrossings);
        CHECK_EQUAL( 0, cells[0]);
        CHECK_CLOSE( 1.0f*std::sqrt(1.0+2.0), distances[0], 1e-6 );
        CHECK_EQUAL( 1, cells[1]);
        CHECK_CLOSE( 0.0, distances[1], 1e-6 );
        CHECK_EQUAL( 4, cells[2]);
        CHECK_CLOSE( 0.0, distances[2], 1e-6 );
        CHECK_EQUAL( 13, cells[3]);
        CHECK_CLOSE( 1.0f*std::sqrt(1.0+2.0), distances[3], 1e-6 );
        CHECK_EQUAL( 14, cells[4]);
        CHECK_CLOSE( 0.0, distances[4], 1e-6 );
        CHECK_EQUAL( 17, cells[5]);
        CHECK_CLOSE( 0.0, distances[5], 1e-6 );
        CHECK_EQUAL( 26, cells[6]);
        CHECK_CLOSE( 1.0f*std::sqrt(1.0+2.0), distances[6], 1e-6 );
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest4, rayTrace_2D_start_outside_an_external_corner_posX_posY ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (  -1.0, -1.0, 0.5 );
        Position_t direction(   1.0, 1.0,  0.0 );
        direction.normalize();
        double distance = 10.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 5, numCrossings);
        CHECK_EQUAL( 0, cells[0]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[0], 1e-6 );
        CHECK_EQUAL( 1, cells[1]);
        CHECK_CLOSE( 0.0, distances[1], 1e-6 );
        CHECK_EQUAL( 4, cells[2]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[2], 1e-6 );
        CHECK_EQUAL( 5, cells[3]);
        CHECK_CLOSE( 0.0, distances[3], 1e-6 );
        CHECK_EQUAL( 8, cells[4]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[4], 1e-6 );
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest4, rayTrace_2D_start_outside_an_external_corner_posX_negY ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (  -1.0, 4.0, 0.5 );
        Position_t direction(   1.0,-1.0,  0.0 );
        direction.normalize();
        double distance = 10.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();


        CHECK_EQUAL( 5, numCrossings);
        CHECK_EQUAL( 6, cells[0]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[0], 1e-6 );
        CHECK_EQUAL( 7, cells[1]);
        CHECK_CLOSE( 0.0, distances[1], 1e-6 );
        CHECK_EQUAL( 4, cells[2]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[2], 1e-6 );
        CHECK_EQUAL( 5, cells[3]);
        CHECK_CLOSE( 0.0, distances[3], 1e-6 );
        CHECK_EQUAL( 2, cells[4]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[4], 1e-6 );

    }

    TEST_FIXTURE(DistanceCalculatorGPUTest4, rayTrace_2D_start_outside_an_external_corner_negX_posY ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (   4.0,-1.0, 0.5 );
        Position_t direction(  -1.0, 1.0,  0.0 );
        direction.normalize();
        double distance = 10.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 5, numCrossings);
        CHECK_EQUAL( 2, cells[0]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[0], 1e-6 );
        CHECK_EQUAL( 1, cells[1]);
        CHECK_CLOSE( 0.0, distances[1], 1e-6 );
        CHECK_EQUAL( 4, cells[2]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[2], 1e-6 );
        CHECK_EQUAL( 3, cells[3]);
        CHECK_CLOSE( 0.0, distances[3], 1e-6 );
        CHECK_EQUAL( 6, cells[4]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[4], 1e-6 );

    }

    TEST_FIXTURE(DistanceCalculatorGPUTest4, rayTrace_2D_start_outside_an_external_corner_negX_negY ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (   4.0,  4.0, 0.5 );
        Position_t direction(  -1.0, -1.0,  0.0 );
        direction.normalize();
        double distance = 10.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();


        CHECK_EQUAL( 5, numCrossings);
        CHECK_EQUAL( 8, cells[0]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[0], 1e-6 );
        CHECK_EQUAL( 7, cells[1]);
        CHECK_CLOSE( 0.0, distances[1], 1e-6 );
        CHECK_EQUAL( 4, cells[2]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[2], 1e-6 );
        CHECK_EQUAL( 3, cells[3]);
        CHECK_CLOSE( 0.0, distances[3], 1e-6 );
        CHECK_EQUAL( 0, cells[4]);
        CHECK_CLOSE( 1.0f*std::sqrt(2.0), distances[4], 1e-6 );

    }

    TEST_FIXTURE(DistanceCalculatorGPUTest4, rayTrace_3D_start_outside_an_external_corner_posX_posY_posZ ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (  -1.0, -1.0, -1.0 );
        Position_t direction(   1.0, 1.0, 1.0 );
        direction.normalize();
        double distance = 10.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 7, numCrossings);
        CHECK_EQUAL( 0, cells[0]);
        CHECK_CLOSE( 1.0f*std::sqrt(1.0+2.0), distances[0], 1e-6 );
        CHECK_EQUAL( 1, cells[1]);
        CHECK_CLOSE( 0.0, distances[1], 1e-6 );
        CHECK_EQUAL( 4, cells[2]);
        CHECK_CLOSE( 0.0, distances[2], 1e-6 );
        CHECK_EQUAL( 13, cells[3]);
        CHECK_CLOSE( 1.0f*std::sqrt(1.0+2.0), distances[3], 1e-6 );
        CHECK_EQUAL( 14, cells[4]);
        CHECK_CLOSE( 0.0, distances[4], 1e-6 );
        CHECK_EQUAL( 17, cells[5]);
        CHECK_CLOSE( 0.0, distances[5], 1e-6 );
        CHECK_EQUAL( 26, cells[6]);
        CHECK_CLOSE( 1.0f*std::sqrt(1.0+2.0), distances[6], 1e-6 );
    }

    TEST_FIXTURE(DistanceCalculatorGPUTest4, rayTrace_3D_start_outside_an_external_corner_negX_negY_negZ ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (    4.0,  4.0,  4.0 );
        Position_t direction(   -1.0, -1.0, -1.0 );
        direction.normalize();
        double distance = 10.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 7, numCrossings);
        CHECK_EQUAL( 26, cells[0]);
        CHECK_CLOSE( 1.0f*std::sqrt(1.0+2.0), distances[0], 1e-6 );
        CHECK_EQUAL( 25, cells[1]);
        CHECK_CLOSE( 0.0, distances[1], 1e-6 );
        CHECK_EQUAL( 22, cells[2]);
        CHECK_CLOSE( 0.0, distances[2], 1e-6 );
        CHECK_EQUAL( 13, cells[3]);
        CHECK_CLOSE( 1.0f*std::sqrt(1.0+2.0), distances[3], 1e-6 );
        CHECK_EQUAL( 12, cells[4]);
        CHECK_CLOSE( 0.0, distances[4], 1e-6 );
        CHECK_EQUAL( 9, cells[5]);
        CHECK_CLOSE( 0.0, distances[5], 1e-6 );
        CHECK_EQUAL( 0, cells[6]);
        CHECK_CLOSE( 1.0f*std::sqrt(1.0+2.0), distances[6], 1e-6 );

    }

    class DistanceCalculatorGPUTest5	{
    public:

        std::unique_ptr<MonteRay_SpatialGrid> pGrid;

        DistanceCalculatorGPUTest5(){
            pGrid = std::make_unique<MonteRay_SpatialGrid>(TransportMeshType::Cartesian, 
                std::array<MonteRay_GridBins, 3>{
                MonteRay_GridBins{-5.0, 5.0, 10},
                MonteRay_GridBins{-5.0, 5.0, 10},
                MonteRay_GridBins{-5.0, 5.0, 10} }
            );
        }
    };

    TEST_FIXTURE(DistanceCalculatorGPUTest5, rayTrace_3D_start_in_middle ) {
        gpuDistanceCalculatorTestHelper tester;

        Position_t position (    0.5,  0.5,  0.5 );
        Position_t direction(    1.0,  0.0,  0.0 );
        direction.normalize();
        gpuRayFloat_t distance = 1000.0;

        tester.setupTimers();
        tester.launchRayTrace(position,direction,distance,false,pGrid.get());
        tester.stopTimers();
        auto distances = tester.getDistancesFromGPU();
        auto cells = tester.getCellsFromCPU();
        unsigned numCrossings = tester.getNumCrossingsFromGPU();

        CHECK_EQUAL( 5, numCrossings);
        CHECK_EQUAL( 555, cells[0]);
        CHECK_CLOSE( 0.5, distances[0], 1e-6 );
        CHECK_EQUAL( 556, cells[1]);
        CHECK_CLOSE( 1.0, distances[1], 1e-6 );
        CHECK_EQUAL( 557, cells[2]);
        CHECK_CLOSE( 1.0, distances[2], 1e-6 );
        CHECK_EQUAL( 558, cells[3]);
        CHECK_CLOSE( 1.0, distances[3], 1e-6 );
        CHECK_EQUAL( 559, cells[4]);
        CHECK_CLOSE( 1.0, distances[4], 1e-6 );
    }
}

} // end namespace
