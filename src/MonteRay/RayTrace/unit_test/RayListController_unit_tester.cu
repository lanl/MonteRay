#include <UnitTest++.h>

#include <iostream>
#include <functional>
#include <cmath>

#include "GPUUtilityFunctions.hh"
#include "BasicTally.hh"
#include "ExpectedPathLength.hh"
#include "MonteRay_timer.hh"
#include "RayListInterface.hh"
#include "RayListController.hh"
#include "GPUErrorCheck.hh"
#include "MaterialProperties.hh"

#include "UnitControllerBase.hh"
#include "MonteRay_SpatialGrid.hh"

namespace RayListController_unit_tester{

using namespace MonteRay;

SUITE( RayListController_unit_tester_basic_tests ) {

    class UnitControllerSetup: public UnitControllerBase {
    public:
        UnitControllerSetup(){

            pGrid = std::make_unique<MonteRay_SpatialGrid>(TransportMeshType::Cartesian, 
              std::array<MonteRay_GridBins, 3>{
              MonteRay_GridBins{-5, 5, 10},
              MonteRay_GridBins{-5, 5, 10},
              MonteRay_GridBins{-5, 5, 10} }
            );

            pTally = std::make_unique<BasicTally>(pGrid->getNumCells());

            // Density of 1.0 for mat number 0
            MaterialProperties::Builder matPropBuilder{};
            matPropBuilder.disableMemoryReduction();
            matPropBuilder.initializeMaterialDescription( std::vector<int>( pGrid->getNumCells(), 0), std::vector<float>( pGrid->getNumCells(), 1.0), pGrid->getNumCells());
            pMatProps = std::make_unique<MaterialProperties>(matPropBuilder.build());

        }

        void setup(){
#ifdef __CUDACC__
            gpuErrchk( cudaPeekAtLastError() );
#endif

#ifdef __CUDACC__
            gpuErrchk( cudaPeekAtLastError() );
#endif
        }

        std::unique_ptr<MaterialProperties> pMatProps;
        std::unique_ptr<MonteRay_SpatialGrid> pGrid;
        std::unique_ptr<BasicTally> pTally;
    };

    TEST_FIXTURE(UnitControllerSetup, ctor_set_capacity ){
        std::cout << "Debug: CollisionPointController_unit_tester -- ctor\n";
        CollisionPointController<MonteRay_SpatialGrid> controller( 1,
                1024,
                pGrid.get(),
                pMatList.get(),
                pMatProps.get(),
                pTally.get() );

        CHECK_EQUAL(100000, controller.capacity());
        CHECK_EQUAL(0, controller.size());
        controller.setCapacity(10);
        CHECK_EQUAL(10, controller.capacity());
    }

    TEST_FIXTURE(UnitControllerSetup, add_a_particle ){
        std::cout << "Debug: CollisionPointController_unit_tester -- add_a_particle\n";
        CollisionPointController<MonteRay_SpatialGrid> controller( 1,
                32,
                pGrid.get(),
                pMatList.get(),
                pMatProps.get(),
                pTally.get() );

        unsigned i = pGrid->getIndex( Position_t( 0.0, 0.0, 0.0 ) );
        ParticleRay_t particle;

        particle.pos[0] = 0.0;
        particle.pos[1] = 0.0;
        particle.pos[2] = 0.0;

        particle.dir[0] = 1.0;
        particle.dir[1] = 0.0;
        particle.dir[2] = 0.0;

        particle.energy[0] = 1.0;
        particle.weight[0] = 1.0;
        particle.index = i;
        particle.detectorIndex = 1;
        particle.particleType = 0;

        controller.add( particle );
        CHECK_EQUAL(1, controller.size());
    }

    TEST_FIXTURE(UnitControllerSetup, add_a_particle_via_ptr ){
        std::cout << "Debug: CollisionPointController_unit_tester -- add_a_particle_via_ptr1\n";
        CollisionPointController<MonteRay_SpatialGrid> controller( 1,
                32,
                pGrid.get(),
                pMatList.get(),
                pMatProps.get(),
                pTally.get() );

        unsigned i = pGrid->getIndex( Position_t(0.0, 0.0, 0.0) );

        ParticleRay_t particle;
        particle.pos[0] = 0.0;
        particle.pos[1] = 0.0;
        particle.pos[2] = 0.0;
        particle.dir[0] = 1.0;
        particle.dir[1] = 0.0;
        particle.dir[2] = 0.0;
        particle.energy[0] = 1.0;
        particle.weight[0] = 1.0;
        particle.index = i;
        particle.detectorIndex = 99;
        particle.particleType = 0;

        controller.add( &particle );
        CHECK_EQUAL(1, controller.size());
    }

    TEST_FIXTURE(UnitControllerSetup, add_two_particles_via_ptr ){
        std::cout << "Debug: CollisionPointController_unit_tester -- add_a_particle_via_ptr2\n";
        CollisionPointController<MonteRay_SpatialGrid> controller( 1,
                1024,
                pGrid.get(),
                pMatList.get(),
                pMatProps.get(),
                pTally.get() );

        unsigned i = pGrid->getIndex( Position_t(0.0, 0.0, 0.0) );

        ParticleRay_t particle[2];
        particle[0].pos[0] = 1.0;
        particle[0].pos[1] = 2.0;
        particle[0].pos[2] = 3.0;
        particle[0].dir[0] = 4.0;
        particle[0].dir[1] = 5.0;
        particle[0].dir[2] = 6.0;
        particle[0].energy[0] = 7.0;
        particle[0].weight[0] = 8.0;
        particle[0].index = 9;
        particle[0].detectorIndex = 99;
        particle[0].particleType = 0;

        particle[1].pos[0] = 11.0;
        particle[1].pos[1] = 12.0;
        particle[1].pos[2] = 13.0;
        particle[1].dir[0] = 14.0;
        particle[1].dir[1] = 15.0;
        particle[1].dir[2] = 16.0;
        particle[1].energy[0] = 17.0;
        particle[1].weight[0] = 18.0;
        particle[1].index = 19;
        particle[1].detectorIndex = 99;
        particle[1].particleType = 0;

        controller.add( particle, 2 );
        CHECK_EQUAL(2, controller.size());
    }

    TEST_FIXTURE(UnitControllerSetup, add_ten_particles_via_ptr ){
        std::cout << "Debug: CollisionPointController_unit_tester -- add_a_particle_via_ptr3\n";
        setup();
        CollisionPointController<MonteRay_SpatialGrid> controller( 1,
                32,
                pGrid.get(),
                pMatList.get(),
                pMatProps.get(),
                pTally.get() );

        ParticleRay_t particle[10];
        for( auto i = 0; i < 10; ++i ){
            particle[i].pos[0] = 1.0;
            particle[i].pos[1] = 2.0;
            particle[i].pos[2] = 3.0;
            particle[i].dir[0] = 4.0;
            particle[i].dir[1] = 5.0;
            particle[i].dir[2] = 6.0;
            particle[i].energy[0] = 7.0;
            particle[i].weight[0] = 8.0;
            particle[i].index = i;
            particle[i].detectorIndex = 99;
            particle[i].particleType = 0;
        }
        controller.setCapacity(3);
        controller.add( particle, 10 );
        CHECK_EQUAL(1, controller.size());
        CHECK_EQUAL(3, controller.getNFlushes());
    }

    TEST_FIXTURE(UnitControllerSetup, single_ray ){
        std::cout << "Debug: CollisionPointController_unit_tester -- single_ray\n";
        CollisionPointController<MonteRay_SpatialGrid> controller( 1,
                1,
                pGrid.get(),
                pMatList.get(),
                pMatProps.get(),
                pTally.get() );

        setup();

        unsigned int matID=0;
        gpuFloatType_t energy = 1.0;
        gpuFloatType_t density = 1.0;
        double testXS = pMatList->material(matID).getTotalXS(energy, density);
        CHECK_CLOSE(.602214179f/1.00866491597f, testXS, 1e-6);

        gpuFloatType_t x = 0.5;
        gpuFloatType_t y = 0.5;
        gpuFloatType_t z = 0.5;

        unsigned i = pGrid->getIndex( Position_t(x, y, z) );
        CHECK_EQUAL( 555, i);

        ParticleRay_t particle;

        particle.pos[0] = x;
        particle.pos[1] = y;
        particle.pos[2] = z;

        particle.dir[0] = 1.0;
        particle.dir[1] = 0.0;
        particle.dir[2] = 0.0;

        particle.energy[0] = 1.0;
        particle.weight[0] = 1.0;
        particle.index = i;
        particle.detectorIndex = 1;
        particle.particleType = 0;

        controller.add(  particle );

        std::cout << "Debug: CollisionPointController_unit_tester -- single_ray - flushing controller \n";
        controller.flush(true);

        std::cout << "Debug: CollisionPointController_unit_tester -- single_ray - copyToCPU \n";

        float distance = 0.5f;
        CHECK_CLOSE( (1.0f-std::exp(-testXS*distance))/testXS, pTally->getTally(i), 1e-5 );
        std::cout << "Debug: CollisionPointController_unit_tester -- finished- single_ray\n";
    }

    TEST_FIXTURE(UnitControllerSetup, write_single_ray_to_file ){
        std::cout << "Debug: CollisionPointController_unit_tester -- write_single_ray_to_file\n";
        CollisionPointController<MonteRay_SpatialGrid> controller( 1,
                1,
                pGrid.get(),
                pMatList.get(),
                pMatProps.get(),
                pTally.get() );

        setup();

        controller.setOutputFileName( "single_ray_collision.bin" );
        CHECK_EQUAL( true, controller.isSendingToFile() );

        unsigned int matID=0;
        gpuFloatType_t energy = 1.0;
        gpuFloatType_t density = 1.0;
        double testXS = pMatList->material(matID).getTotalXS(energy, density);
        CHECK_CLOSE(.602214179f/1.00866491597f, testXS, 1e-6);

        gpuFloatType_t x = 0.5;
        gpuFloatType_t y = 0.5;
        gpuFloatType_t z = 0.5;

        unsigned i = pGrid->getIndex( Position_t(x, y, z) );
        CHECK_EQUAL( 555, i);

        ParticleRay_t particle;

        particle.pos[0] = x;
        particle.pos[1] = y;
        particle.pos[2] = z;

        particle.dir[0] = 1.0;
        particle.dir[1] = 0.0;
        particle.dir[2] = 0.0;

        particle.energy[0] = 1.0;
        particle.weight[0] = 1.0;
        particle.index = i;
        particle.detectorIndex = 1;
        particle.particleType = 0;

        controller.add(  particle );

        controller.flush(true);
    }


    TEST_FIXTURE(UnitControllerSetup, read_single_ray_to_file ){
        std::cout << "Debug: CollisionPointController_unit_tester -- read_single_ray_to_file\n";
        CollisionPointController<MonteRay_SpatialGrid> controller( 1,
                1,
                pGrid.get(),
                pMatList.get(),
                pMatProps.get(),
                pTally.get() );

        setup();


        unsigned int matID=0;
        gpuFloatType_t energy = 1.0;
        gpuFloatType_t density = 1.0;
        double testXS = pMatList->material(matID).getTotalXS(energy, density);
        CHECK_CLOSE(.602214179f/1.00866491597f, testXS, 1e-6);

        gpuFloatType_t x = 0.5;
        gpuFloatType_t y = 0.5;
        gpuFloatType_t z = 0.5;

        unsigned i = pGrid->getIndex( Position_t(x, y, z) );
        CHECK_EQUAL( 555, i);

        controller.readCollisionsFromFile( "single_ray_collision.bin" );

        float distance = 0.5f;
        CHECK_CLOSE( (1.0f-std::exp(-testXS*distance))/testXS, pTally->getTally(i), 1e-5 );
    }

    TEST_FIXTURE(UnitControllerSetup, set_write_to_file_only_via_ctor ){
        std::cout << "Debug: CollisionPointController_unit_tester -- add_a_particle\n";
        CollisionPointController<MonteRay_SpatialGrid> controller( 2, std::string("collisionPoints_via_ctor_test_file.bin") );
        CHECK_EQUAL( true, controller.isSendingToFile() );

        gpuFloatType_t x = 0.5;
        gpuFloatType_t y = 0.5;
        gpuFloatType_t z = 0.5;

        unsigned i = pGrid->getIndex( Position_t(x, y, z) );
        CHECK_EQUAL( 555, i);

        ParticleRay_t particle;

        particle.pos[0] = x;
        particle.pos[1] = y;
        particle.pos[2] = z;

        particle.dir[0] = 1.0;
        particle.dir[1] = 0.0;
        particle.dir[2] = 0.0;

        particle.energy[0] = 1.0;
        particle.weight[0] = 1.0;
        particle.index = i;
        particle.detectorIndex = 1;
        particle.particleType = 0;

        controller.add(  particle );

        controller.flush(true);
    }

    TEST_FIXTURE(UnitControllerSetup, read_single_ray_to_file_from_writeonly_ctor ){
        std::cout << "Debug: CollisionPointController_unit_tester -- read_single_ray_to_file_from_writeonly_ctor\n";
        CollisionPointController<MonteRay_SpatialGrid> controller( 1,
                1,
                pGrid.get(),
                pMatList.get(),
                pMatProps.get(),
                pTally.get() );

        setup();


        unsigned int matID=0;
        gpuFloatType_t energy = 1.0;
        gpuFloatType_t density = 1.0;
        double testXS = pMatList->material(matID).getTotalXS(energy, density);
        CHECK_CLOSE(.602214179f/1.00866491597f, testXS, 1e-6);

        gpuFloatType_t x = 0.5;
        gpuFloatType_t y = 0.5;
        gpuFloatType_t z = 0.5;

        unsigned i = pGrid->getIndex( Position_t(x, y, z) );
        CHECK_EQUAL( 555, i);

        controller.readCollisionsFromFile( "collisionPoints_via_ctor_test_file.bin" );

        float distance = 0.5f;
        CHECK_CLOSE( (1.0f-std::exp(-testXS*distance))/testXS, pTally->getTally(i), 1e-5 );
    }
}

}
