#include <UnitTest++.h>

#include <iostream>
#include <functional>
#include <memory>

#include "GPUUtilityFunctions.hh"

#include "gpuTally.h"

#include "MonteRay_timer.hh"
#include "RayListController.hh"
#include "GridBins.h"
#include "MonteRayMaterialList.hh"
#include "MonteRay_MaterialProperties.hh"
#include "MonteRay_ReadLnk3dnt.hh"
#include "gpuTally.h"
#include "RayListInterface.hh"
#include "MonteRayConstants.hh"
#include "MonteRayNextEventEstimator.hh"

namespace RayListController_wNextEventEstimator_fi_tester {

using namespace MonteRay;

SUITE( RayListController_wNextEventEstimator_fi_tester_suite ) {

	class ControllerSetup {
	public:
		ControllerSetup(){

	    	cudaReset();
	    	gpuCheck();
			pGrid = new GridBinsHost();
			pGrid->setVertices( 0, 0.0, 2.0, 2);
			pGrid->setVertices( 1, -10.0, 10.0, 1);
			pGrid->setVertices( 2, -10.0, 10.0, 1);
			pGrid->finalize();

	    	pMatProps = new MonteRay_MaterialProperties;

	    	MonteRay_CellProperties cell1, cell2;
	    	cell1.add( 0, 0.0); // vacuum
	    	pMatProps->add( cell1 );

	    	cell2.add( 0, 1.0); // density = 1.0
	    	pMatProps->add( cell2 );

	    	pXS = new MonteRayCrossSectionHost(4);

			pMat = new MonteRayMaterialHost(1);

	        pMatList = new MonteRayMaterialListHost(1,1,3);

		}

		void setup(){

	    	pGrid->copyToGPU();

	    	pXS->setParticleType( photon );
			pXS->setTotalXS(0,  std::log( 1e-11 ), std::log( 1.0 ) );
			pXS->setTotalXS(1,  std::log( 0.75 ),  std::log( 1.0 ) );
			pXS->setTotalXS(2,  std::log( 1.00 ),  std::log( 2.0 ) );
			pXS->setTotalXS(3,  std::log( 3.00 ),  std::log( 4.0 ) );
			pXS->setAWR( gpu_AvogadroBarn / gpu_neutron_molar_mass );

	        pMat->add(0, *pXS, 1.0);
	        pMat->copyToGPU();

	        pMatList->add( 0, *pMat, 0 );
	        pMatList->copyToGPU();

	        pMatProps->renumberMaterialIDs(*pMatList);
	        pMatProps->copyToGPU();

	        pXS->copyToGPU();

		}

		~ControllerSetup(){
			delete pGrid;
			delete pMatList;
			delete pMatProps;
			delete pXS;
			delete pMat;
		}

		GridBinsHost* pGrid;
		MonteRayMaterialListHost* pMatList;
		MonteRay_MaterialProperties* pMatProps;
    	MonteRayCrossSectionHost* pXS;
        MonteRayMaterialHost* pMat;

	};

#if true
	TEST( setup ) {
		gpuCheck();
	}

    TEST_FIXTURE(ControllerSetup, ctorForNEE ){
    	unsigned numPointDets = 1;
    	NextEventEstimatorController controller( 1,
 				                             1,
 				                             pGrid,
 				                             pMatList,
 				                             pMatProps,
 				                             numPointDets );

    	CHECK_EQUAL( true, controller.isUsingNextEventEstimator() );
        CHECK_EQUAL(1000000, controller.capacity());
        CHECK_EQUAL(0, controller.size());
    }
#endif

    TEST_FIXTURE(ControllerSetup, testOnGPU ){
    	unsigned numPointDets = 1;

    	setup();
    	NextEventEstimatorController controller( 1,
 				                             1,
 				                             pGrid,
 				                             pMatList,
 				                             pMatProps,
 				                             numPointDets );
    	controller.setCapacity(10);

    	CHECK_EQUAL( true, controller.isUsingNextEventEstimator() );
    	unsigned id = controller.addPointDet( 2.0, 0.0, 0.0 );
    	CHECK_EQUAL( 0, id);

    	controller.copyPointDetToGPU();

		gpuFloatType_t x = 0.0;
		gpuFloatType_t y = 0.0;
		gpuFloatType_t z = 0.0;
		gpuFloatType_t u = 1.0;
		gpuFloatType_t v = 0.0;
		gpuFloatType_t w = 0.0;

		gpuFloatType_t energy[3];
		energy[0]= 0.5;
		energy[1]= 1.0;
		energy[2]= 3.0;

		gpuFloatType_t weight[3];
		weight[0] = 0.3;  // isotropic
		weight[1] = 1.0;
		weight[2] = 2.0;

		Ray_t<3> ray;
		ray.pos[0] = x;
		ray.pos[1] = y;
		ray.pos[2] = z;
		ray.dir[0] = u;
		ray.dir[1] = v;
		ray.dir[2] = w;

		for( unsigned i=0;i<3;++i) {
			ray.energy[i] = energy[i];
			ray.weight[i] = weight[i];
		}
		ray.index = 0;
		ray.detectorIndex = 0;
		ray.particleType = photon;

		controller.add( ray );
		controller.add( ray );
		CHECK_EQUAL(2, controller.size());
        CHECK_EQUAL(10, controller.capacity());

//        std::cout << "Debug: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n";
//        std::cout << "Debug: RayListController_wNexteVentEstimator_fi_tester.cu:: testOnGPU - flushing \n";
        controller.sync();
        controller.flush(true);
        controller.sync();
        controller.copyPointDetTallyToCPU();
//        std::cout << "Debug: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n";

        gpuFloatType_t expected1 = ( 0.3f / (2.0f * MonteRay::pi * 4.0f ) ) * exp( -1.0*1.0 );
        gpuFloatType_t expected2 = ( 1.0f / (2.0f * MonteRay::pi * 4.0f ) ) * exp( -1.0*2.0 );
        gpuFloatType_t expected3 = ( 2.0f / (2.0f * MonteRay::pi * 4.0f ) ) * exp( -1.0*4.0 );

        CHECK_CLOSE( 2*(expected1+expected2+expected3), controller.getPointDetTally(0), 1e-7);
    }

#if false
    TEST_FIXTURE(ControllerSetup, compare_with_mcatk ){
    	// exact numbers from expected path length tally in mcatk

    	CollisionPointController controller( 256,
    			256,
    			pGrid,
    			pMatList,
    			pMatProps,
    			pTally );

    	setup();

    	RayListInterface<1> bank1(500000);
//    	bool end = false;
//    	unsigned offset = 0;

    	double x = 0.0001;
    	double y = 0.0001;
    	double z = 0.0001;
    	double u = 1.0;
    	double v = 0.0;
    	double w = 0.0;
    	double energy = 1.0;
    	double weight = 1.0;
    	unsigned index = 505050;
    	unsigned detectorIndex = 101;
    	short int particleType = 0;

    	unsigned nI = 2;
    	unsigned nJ = 1;
    	for( unsigned i = 0; i < nI; ++i ) {
    	    for( unsigned j = 0; j < nJ; ++j ) {
    	    	ParticleRay_t ray;
    	    	ray.pos[0] = x;
    	    	ray.pos[1] = y;
    	    	ray.pos[2] = z;
    	    	ray.dir[0] = u;
    	    	ray.dir[1] = v;
    	    	ray.dir[2] = w;
    	    	ray.energy[0] = energy;
    	    	ray.weight[0] = weight;
    	    	ray.index = index;
    	    	ray.detectorIndex = detectorIndex;
    	    	ray.particleType = particleType;
    	        controller.add( ray );
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
#endif

#if( false )
    TEST_FIXTURE(ControllerSetup, launch_with_collisions_From_file ){
    	std::cout << "Debug: ********************************************* \n";
    	std::cout << "Debug: Starting rayTrace tester with single looping bank \n";
    	CollisionPointController controller( 256,
    			256,
    			pGrid,
    			pMatList,
    			pMatProps,
    			pTally );
    	controller.setCapacity( 1000000 );
    	setup();

    	RayListInterface<1> bank1(50000);
    	bool end = false;
    	unsigned offset = 0;

    	while( ! end ) {
//    		std::cout << "Debug: reading to bank\n";
    		end = bank1.readToBank( "MonteRayTestFiles/collisionsGodivaRCart100x100x100InWater_2568016Rays.bin", offset );
    		offset += bank1.size();

    		for( unsigned i=0; i<bank1.size(); ++i ) {
    			controller.add( bank1.getParticle(i) );
    		}

    		if( end ) {
    			controller.flush(true);
    		}

    	}

    	pTally->copyToCPU();

    	// TODO - find the discrepancy
    	CHECK_CLOSE( 0.0201738, pTally->getTally(24), 1e-5 );  // 0.0201584 is benchmark value - not sure why the slight difference
    	CHECK_CLOSE( 0.0504394, pTally->getTally(500182), 1e-4 );

    }
#endif

}

}
