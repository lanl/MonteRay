#include <UnitTest++.h>

#include <iostream>
#include <functional>

#include "GPUUtilityFunctions.hh"

#include "gpuTally.h"
#include "ExpectedPathLength.h"
#include "MonteRay_timer.hh"
#include "CollisionPointController.h"
#include "GridBins.h"
#include "SimpleMaterialList.h"
#include "MonteRay_CellProperties.hh"
#include "gpuTally.h"
#include "CollisionPoints.h"

namespace {

using namespace MonteRay;

SUITE( Collision_fi_bank_controller_tester ) {

	class ControllerSetup {
	public:
		ControllerSetup(){

	    	cudaReset();
	    	gpuCheck();
			pGrid = new GridBinsHost(-33.5, 33.5, 100,
			          -33.5, 33.5, 100,
			          -33.5, 33.5, 100);


	    	pTally = new gpuTallyHost( pGrid->getNumCells() );

	    	pMatProps = new CellPropertiesHost(2);

	    	u234s = new MonteRayCrossSectionHost(1);
	    	u235s = new MonteRayCrossSectionHost(1);
	    	u238s = new MonteRayCrossSectionHost(1);
	    	h1s = new MonteRayCrossSectionHost(1);
	    	o16s = new MonteRayCrossSectionHost(1);

	        metal = new SimpleMaterialHost(3);
	        water = new SimpleMaterialHost(2);

	        pMatList = new SimpleMaterialListHost(2,5);

		}

		void setup(){


	    	pGrid->copyToGPU();

	    	pTally->copyToGPU();

	    	pMatProps->read( "/usr/projects/mcatk/user/jsweezy/link_files/godivaR_geometry_100x100x100.bin" );
	    	pMatProps->copyToGPU();

	        u234s->read( "/usr/projects/mcatk/user/jsweezy/link_files/u234_simpleCrossSection.bin" );
	        u235s->read( "/usr/projects/mcatk/user/jsweezy/link_files/u235_simpleCrossSection.bin" );
	        u238s->read( "/usr/projects/mcatk/user/jsweezy/link_files/u238_simpleCrossSection.bin" );
	        h1s->read( "/usr/projects/mcatk/user/jsweezy/link_files/h1_simpleCrossSection.bin" );
	        o16s->read( "/usr/projects/mcatk/user/jsweezy/link_files/o16_simpleCrossSection.bin" );

	        metal->add(0, *u234s, 0.01);
	        metal->add(1, *u235s, 0.98);
	        metal->add(2, *u238s, 0.01);
	        metal->copyToGPU();

	        water->add(0, *h1s, 2.0f/3.0f );
	        water->add(1, *o16s, 1.0f/3.0f );
	        water->copyToGPU();

	        pMatList->add( 0, *metal, 0 );
	        pMatList->add( 1, *water, 1 );
	        pMatList->copyToGPU();

	        u234s->copyToGPU();
	        u235s->copyToGPU();
	        u238s->copyToGPU();
	        h1s->copyToGPU();
	        o16s->copyToGPU();
		}

		~ControllerSetup(){
			delete pGrid;
			delete pMatList;
			delete pMatProps;
			delete pTally;
			delete u234s;
			delete u235s;
			delete u238s;
			delete h1s;
			delete o16s;
			delete metal;
			delete water;
		}

		GridBinsHost* pGrid;
		SimpleMaterialListHost* pMatList;
		CellPropertiesHost* pMatProps;
		gpuTallyHost* pTally;

    	MonteRayCrossSectionHost* u234s;
        MonteRayCrossSectionHost* u235s;
        MonteRayCrossSectionHost* u238s;
        MonteRayCrossSectionHost* h1s;
        MonteRayCrossSectionHost* o16s;

        SimpleMaterialHost* metal;
        SimpleMaterialHost* water;

	};

#if false
	TEST( setup ) {
		gpuCheck();
	}

    TEST_FIXTURE(ControllerSetup, ctor ){
        CollisionPointController controller( 1024,
 				                             1024,
 				                             pGrid,
 				                             pMatList,
 				                             pMatProps,
 				                             pTally );

        CHECK_EQUAL(1000000, controller.capacity());
        CHECK_EQUAL(0, controller.size());
    }

    TEST_FIXTURE(ControllerSetup, setCapacity ){
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

    TEST_FIXTURE(ControllerSetup, add_a_particle ){
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

        CHECK_EQUAL(1, controller.size());
    }
#endif

    TEST_FIXTURE(ControllerSetup, compare_with_mcatk ){
    	// exact numbers from expected path length tally in mcatk

    	CollisionPointController controller( 1024,
    			1024,
    			pGrid,
    			pMatList,
    			pMatProps,
    			pTally );

    	setup();

    	CollisionPointsHost bank1(500000);
    	bool end = false;
    	unsigned offset = 0;

    	double x = 0.0001;
    	double y = 0.0001;
    	double z = 0.0001;
    	double u = 1.0;
    	double v = 0.0;
    	double w = 0.0;
    	double energy = 1.0;
    	double weight = 1.0;
    	double index = 505050;

    	unsigned nI = 2;
    	unsigned nJ = 1;
    	for( unsigned i = 0; i < nI; ++i ) {
    	    for( unsigned j = 0; j < nJ; ++j ) {
    	        controller.add( x, y, z, u, v, w, energy, weight, index );
    	    }
    	    CHECK_EQUAL( nJ, controller.size() );
    	    controller.flush(false);
    	}
    	CHECK_EQUAL( 0, controller.size() );
    	controller.flush(true);

    	pTally->copyToCPU();

    	CHECK_CLOSE( 0.601248*nI*nJ, pTally->getTally(index), 1e-6*nI*nJ );
    	CHECK_CLOSE( 0.482442*nI*nJ, pTally->getTally(index+1), 1e-6*nI*nJ );

    }

#if( false )
    TEST_FIXTURE(ControllerSetup, launch_with_collisions_From_file ){
    	CollisionPointController controller( 1024,
    			1024,
    			pGrid,
    			pMatList,
    			pMatProps,
    			pTally );

    	setup();

    	CollisionPointsHost bank1(500000);
    	bool end = false;
    	unsigned offset = 0;

    	while( ! end ) {
    		end = bank1.readToBank( "/usr/projects/mcatk/user/jsweezy/link_files/collisionsGodivaCyl100x100x100InWater.bin", offset );
    		offset += bank1.size();

    		for( unsigned i=0; i<bank1.size(); ++i ) {

    			controller.add(
    				bank1.getPosition(i).x, bank1.getPosition(i).y, bank1.getPosition(i).z,
    				bank1.getDirection(i).u, bank1.getDirection(i).v, bank1.getDirection(i).w,
    				bank1.getEnergy(i), bank1.getWeight(i), bank1.getIndex(i)
    			);
    		}

    		if( end ) {
    			controller.flush(true);
    		}

    	}

    	pTally->copyToCPU();

    	CHECK_CLOSE( 9.43997, pTally->getTally(0), 1e-5 );
    	CHECK_CLOSE( 16.5143, pTally->getTally(50+100*100), 1e-4 );

    }
#endif

}

}
