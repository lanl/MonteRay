#include <UnitTest++.h>

#include <iostream>
#include <functional>
#include <cmath>

#include "GPUUtilityFunctions.hh"
#include "gpuTally.h"
#include "ExpectedPathLength.h"
#include "cpuTimer.h"
#include "CollisionPointController.h"
#include "GridBins.h"
#include "SimpleMaterialList.h"
#include "SimpleMaterialProperties.h"
#include "gpuTally.h"
#include "CollisionPoints.h"

namespace {

using namespace MonteRay;

SUITE( Collision_unit_bank_controller_tester ) {

	class UnitControllerSetup {
	public:
		UnitControllerSetup(){

	    	cudaReset();
	    	gpuCheck();
			pGrid = new GridBinsHost(
					  -5.0, 5.0, 10,
					  -5.0, 5.0, 10,
					  -5.0, 5.0, 10);


	    	pTally = new gpuTallyHost( pGrid->getNumCells() );

	    	pMatProps = new SimpleMaterialPropertiesHost(pGrid->getNumCells());

	    	// xs from 0.0 to 100.0 mev with total cross-section of 1.0
	    	xs = new SimpleCrossSectionHost(2);

	        metal = new SimpleMaterialHost(1);

	        pMatList = new SimpleMaterialListHost(1);

		}

		void setup(){

	    	pGrid->copyToGPU();

	    	pTally->copyToGPU();
	    	pTally->clear();

	    	// Density of 1.0 for mat number 0
	    	for( unsigned i = 0; i < pGrid->getNumCells(); ++i ) {
	    		pMatProps->addDensityAndID( i, 1.0, 0 );
	    	}

	    	pMatProps->copyToGPU();

	    	xs->setTotalXS(0, 0.00001, 1.0 );
	    	xs->setTotalXS(1, 100.0, 1.0 );
	    	xs->setAWR( 1.0 );


	        metal->add(0, *xs, 1.0);
	        metal->copyToGPU();

	        // add metal as mat number 0
	        pMatList->add( 0, *metal, 0 );
	        pMatList->copyToGPU();
	        xs->copyToGPU();
		}

		~UnitControllerSetup(){
			delete pGrid;
			delete pMatList;
			delete pMatProps;
			delete pTally;
			delete xs;
			delete metal;
		}

		GridBinsHost* pGrid;
		SimpleMaterialListHost* pMatList;
		SimpleMaterialPropertiesHost* pMatProps;
		gpuTallyHost* pTally;

    	SimpleCrossSectionHost* xs;

        SimpleMaterialHost* metal;
	};

	TEST( setup ) {
		gpuCheck();
	}

    TEST_FIXTURE(UnitControllerSetup, ctor ){
    	std::cout << "Debug: CollisionPointController_unit_tester -- ctor\n";
        CollisionPointController controller( 1024,
 				                             1024,
 				                             pGrid,
 				                             pMatList,
 				                             pMatProps,
 				                             pTally );

        CHECK_EQUAL(1000000, controller.capacity());
        CHECK_EQUAL(0, controller.size());
    }

    TEST_FIXTURE(UnitControllerSetup, setCapacity ){
    	std::cout << "Debug: CollisionPointController_unit_tester -- setCapacity\n";
        CollisionPointController controller( 1024,
 				                             1024,
 				                             pGrid,
 				                             pMatList,
 				                             pMatProps,
 				                             pTally );

        CHECK_EQUAL(1000000, controller.capacity());
        controller.setCapacity(10);
        CHECK_EQUAL(10, controller.capacity());
    }

    TEST_FIXTURE(UnitControllerSetup, add_a_particle ){
    	std::cout << "Debug: CollisionPointController_unit_tester -- add_a_particle\n";
        CollisionPointController controller( 1024,
 				                             1024,
 				                             pGrid,
 				                             pMatList,
 				                             pMatProps,
 				                             pTally );

        unsigned i = pGrid->getIndex( 0.0, 0.0, 0.0 );
        controller.add( 0.0, 0.0, 0.0,
        		        1.0, 0.0, 0.0,
        		        1.0, 1.0, i);
        //     	s
        CHECK_EQUAL(1, controller.size());
    }

    TEST_FIXTURE(UnitControllerSetup, add_a_particle_via_ptr ){
    	std::cout << "Debug: CollisionPointController_unit_tester -- add_a_particle_via_ptr1\n";
        CollisionPointController controller( 1024,
 				                             1024,
 				                             pGrid,
 				                             pMatList,
 				                             pMatProps,
 				                             pTally );

        unsigned i = pGrid->getIndex( 0.0, 0.0, 0.0 );

        gpuParticle_t particle;
        particle.pos[0] = 0.0;
        particle.pos[1] = 0.0;
        particle.pos[2] = 0.0;
        particle.dir[0] = 1.0;
        particle.dir[1] = 0.0;
        particle.dir[2] = 0.0;
        particle.energy = 1.0;
        particle.weight = 1.0;
        particle.index = i;

        controller.add( &particle );
        CHECK_EQUAL(1, controller.size());
    }

    TEST_FIXTURE(UnitControllerSetup, add_two_particles_via_ptr ){
    	std::cout << "Debug: CollisionPointController_unit_tester -- add_a_particle_via_ptr2\n";
        CollisionPointController controller( 1024,
 				                             1024,
 				                             pGrid,
 				                             pMatList,
 				                             pMatProps,
 				                             pTally );

        unsigned i = pGrid->getIndex( 0.0, 0.0, 0.0 );

        gpuParticle_t particle[2];
        particle[0].pos[0] = 1.0;
        particle[0].pos[1] = 2.0;
        particle[0].pos[2] = 3.0;
        particle[0].dir[0] = 4.0;
        particle[0].dir[1] = 5.0;
        particle[0].dir[2] = 6.0;
        particle[0].energy = 7.0;
        particle[0].weight = 8.0;
        particle[0].index = 9;

        particle[1].pos[0] = 11.0;
        particle[1].pos[1] = 12.0;
        particle[1].pos[2] = 13.0;
        particle[1].dir[0] = 14.0;
        particle[1].dir[1] = 15.0;
        particle[1].dir[2] = 16.0;
        particle[1].energy = 17.0;
        particle[1].weight = 18.0;
        particle[1].index = 19;

        controller.add( particle, 2 );
        CHECK_EQUAL(2, controller.size());
    }

    TEST_FIXTURE(UnitControllerSetup, add_ten_particles_via_ptr ){
    	std::cout << "Debug: CollisionPointController_unit_tester -- add_a_particle_via_ptr3\n";
        CollisionPointController controller( 1024,
 				                             1024,
 				                             pGrid,
 				                             pMatList,
 				                             pMatProps,
 				                             pTally );
        setup();

        gpuParticle_t particle[10];
        for( auto i = 0; i < 10; ++i ){
        	particle[i].pos[0] = 1.0;
        	particle[i].pos[1] = 2.0;
        	particle[i].pos[2] = 3.0;
        	particle[i].dir[0] = 4.0;
        	particle[i].dir[1] = 5.0;
        	particle[i].dir[2] = 6.0;
        	particle[i].energy = 7.0;
        	particle[i].weight = 8.0;
        	particle[i].index = i;
        }
        controller.setCapacity(3);
        controller.add( particle, 10 );
        CHECK_EQUAL(1, controller.size());
        CHECK_EQUAL(3, controller.getNFlushes());
    }

    TEST_FIXTURE(UnitControllerSetup, single_ray ){
    	std::cout << "Debug: CollisionPointController_unit_tester -- single_ray\n";
    	CollisionPointController controller( 1,
    			1,
    			pGrid,
    			pMatList,
    			pMatProps,
    			pTally );

     	setup();

     	unsigned int matID=0;
     	gpuFloatType_t energy = 1.0;
     	gpuFloatType_t density = 1.0;
     	unsigned HashBin = getHashBin( pMatList->getHashPtr()->getPtr(), energy);
     	double testXS = MonteRay::getTotalXS( pMatList->getPtr(), matID, pMatList->getHashPtr()->getPtr(), HashBin, energy, density);
     	CHECK_CLOSE(.602214179f/1.00866491597f, testXS, 1e-6);

    	gpuFloatType_t x = 0.5;
    	gpuFloatType_t y = 0.5;
    	gpuFloatType_t z = 0.5;

    	unsigned i = pGrid->getIndex( x, y, z );
    	CHECK_EQUAL( 555, i);

        controller.add(   x,   y,   z,
        		        1.0, 0.0, 0.0,
        		        1.0, 1.0, i);

        std::cout << "Debug: CollisionPointController_unit_tester -- single_ray - flushing controller \n";
        controller.flush(true);

        std::cout << "Debug: CollisionPointController_unit_tester -- single_ray - copyToCPU \n";
    	pTally->copyToCPU();

    	float distance = 0.5f;
    	CHECK_CLOSE( (1.0f-std::exp(-testXS*distance))/testXS, pTally->getTally(i), 1e-5 );
    	std::cout << "Debug: CollisionPointController_unit_tester -- finished- single_ray\n";
    }

//    TEST_FIXTURE(ControllerSetup, launch_with_collisions_From_file ){
//    	CollisionPointController controller( 1024,
//    			1024,
//    			pGrid,
//    			pMatList,
//    			pMatProps,
//    			pTally );
//
//    	setup();
//
//    	CollisionPointsHost bank1(500000);
//    	bool end = false;
//    	unsigned offset = 0;
//
//    	while( ! end ) {
//    		end = bank1.readToBank( "/usr/projects/mcatk/user/jsweezy/link_files/collisionsGodivaCyl100x100x100InWater.bin", offset );
//    		offset += bank1.size();
//
//    		for( unsigned i=0; i<bank1.size(); ++i ) {
//
//    			controller.add(
//    				bank1.getPosition(i).x, bank1.getPosition(i).y, bank1.getPosition(i).z,
//    				bank1.getDirection(i).u, bank1.getDirection(i).v, bank1.getDirection(i).w,
//    				bank1.getEnergy(i), bank1.getWeight(i), bank1.getIndex(i)
//    			);
//    		}
//
//    		if( end ) {
//    			controller.flush(true);
//    		}
//
//    	}
//
//    	pTally->copyToCPU();
//
//    	CHECK_CLOSE( 9.43997, pTally->getTally(0), 1e-5 );
//    	CHECK_CLOSE( 16.5143, pTally->getTally(50+100*100), 1e-4 );
//
//    }


}

}
