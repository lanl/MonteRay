#include <UnitTest++.h>

#include <iostream>
#include <functional>
#include <cmath>

#include "GPUUtilityFunctions.hh"
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
        CollisionPointController::Builder controllerBuilder;
        UnitControllerSetup(){

            pGrid = std::make_unique<MonteRay_SpatialGrid>(TransportMeshType::Cartesian, 
              std::array<MonteRay_GridBins, 3>{
              MonteRay_GridBins{-5, 5, 10},
              MonteRay_GridBins{-5, 5, 10},
              MonteRay_GridBins{-5, 5, 10} }
            );

            MonteRay::ExpectedPathLengthTally::Builder tallyBuilder;
            tallyBuilder.spatialBins(pGrid->getNumCells());

            // Density of 1.0 for mat number 0
            MaterialProperties::Builder matPropBuilder{};
            matPropBuilder.disableMemoryReduction();
            matPropBuilder.initializeMaterialDescription( std::vector<int>( pGrid->getNumCells(), 0), std::vector<float>( pGrid->getNumCells(), 1.0), pGrid->getNumCells());
            pMatProps = std::make_unique<MaterialProperties>(matPropBuilder.build());

            controllerBuilder.nBlocks(1)
                      .nThreads(1)
                      .geometry(pGrid.get())
                      .materialProperties(pMatProps.get())
                      .materialList(pMatList.get())
                      .expectedPathLengthTally(tallyBuilder.build());
        }

        std::unique_ptr<MaterialProperties> pMatProps;
        std::unique_ptr<MonteRay_SpatialGrid> pGrid;
    };

    TEST_FIXTURE(UnitControllerSetup, ctor_default_capacity){
      auto controller = controllerBuilder.build();
      CHECK_EQUAL(100000, controller.capacity());
      CHECK_EQUAL(0, controller.size());
    }

    TEST_FIXTURE(UnitControllerSetup, ctor_set_capacity){
      controllerBuilder.capacity(10);
      auto controller = controllerBuilder.build();
      CHECK_EQUAL(10, controller.capacity());
    }

    TEST_FIXTURE(UnitControllerSetup, add_a_particle ){
      auto controller = controllerBuilder.build();
      ParticleRay_t particle;
      controller.add( particle );
      CHECK_EQUAL(1, controller.size());
    }

    TEST_FIXTURE(UnitControllerSetup, add_a_particle_via_ptr ){
      auto controller = controllerBuilder.build();
      ParticleRay_t particle;
      controller.add( &particle );
      CHECK_EQUAL(1, controller.size());
    }

    TEST_FIXTURE(UnitControllerSetup, add_two_particles_via_ptr ){
      auto controller = controllerBuilder.build();
      ParticleRay_t particle[2];
      controller.add( particle, 2 );
      CHECK_EQUAL(2, controller.size());
    }

    TEST_FIXTURE(UnitControllerSetup, add_ten_particles_via_ptr ){
      controllerBuilder.capacity(3);
      auto controller = controllerBuilder.build();
      ParticleRay_t particle[10];
      controller.add( particle, 10 );
      CHECK_EQUAL(1, controller.size());
      CHECK_EQUAL(3, controller.getNFlushes());
    }

    TEST_FIXTURE(UnitControllerSetup, single_ray_expected_path_length ){
      auto controller = controllerBuilder.capacity(1).build();
      printf("TESTING TEST TEST 1\n");

      unsigned int matID=0;
      gpuFloatType_t energy = 1.0;
      gpuFloatType_t density = 1.0;
      double testXS = pMatList->material(matID).getTotalXS(energy, density);
      CHECK_CLOSE(.602214179f/1.00866491597f, testXS, 1e-6);
      printf("TESTING TEST TEST 1\n");

      ParticleRay_t particle;

      particle.pos[0] = 0.5;
      particle.pos[1] = 0.5;
      particle.pos[2] = 0.5;

      particle.dir[0] = 1.0;
      particle.dir[1] = 0.0;
      particle.dir[2] = 0.0;

      particle.energy[0] = 1.0;
      particle.weight[0] = 1.0;

      particle.index = pGrid->getIndex(particle.pos);
      CHECK_EQUAL(555, particle.index);
      printf("TESTING TEST TEST 2\n");

      particle.particleType = 0;

      controller.add(particle);
      printf("TESTING TEST TEST 3\n");

      controller.flush(true);
      printf("TESTING TEST TEST 4\n");

      float distance = 0.5f;
      CHECK_CLOSE( (1.0f-std::exp(-testXS*distance))/testXS, controller.contribution(particle.index), 1e-5 );
      printf("TESTING TEST TEST 5\n");
    }

    TEST_FIXTURE(UnitControllerSetup, write_single_ray_to_file ){
      std::cout << "Debug: CollisionPointController_unit_tester -- write_single_ray_to_file\n";
      controllerBuilder.outputFileName( "single_ray_collision.bin" );
      auto controller = controllerBuilder.build();
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

      controller.add(particle);
      controller.flush(true);
    }

    TEST_FIXTURE(UnitControllerSetup, read_single_ray_from_file){
      std::cout << "Debug: CollisionPointController_unit_tester -- read_single_ray_from_file\n";
      controllerBuilder.capacity(1);
      auto controller = controllerBuilder.build();
      controller.readCollisionsFromFile("single_ray_collision.bin");
      unsigned int matID=0;
      gpuFloatType_t energy = 1.0;
      gpuFloatType_t density = 1.0;
      double testXS = pMatList->material(matID).getTotalXS(energy, density);
      float distance = 0.5f;
      CHECK_CLOSE((1.0f-std::exp(-testXS*distance))/testXS, controller.contribution(555), 1e-5);
    }

}

}
