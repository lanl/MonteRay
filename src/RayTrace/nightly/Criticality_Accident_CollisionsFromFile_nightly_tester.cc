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
#include "MonteRay_MaterialProperties.hh"
#include "MonteRay_ReadLnk3dnt.hh"
#include "gpuTally.h"
#include "CollisionPoints.h"

namespace Criticality_Accident_wCollisionFile_nightly_tester{

using namespace MonteRay;

SUITE( Criticality_Accident_wCollisionFile_tester ) {

	class ControllerSetup {
	public:
		ControllerSetup(){

	    	cudaReset();
	    	gpuCheck();

	    	iso1001  = new MonteRayCrossSectionHost(1);
	    	iso6000  = new MonteRayCrossSectionHost(1);
	    	iso7014  = new MonteRayCrossSectionHost(1);
	    	iso8016  = new MonteRayCrossSectionHost(1);
	    	iso12000 = new MonteRayCrossSectionHost(1);
	    	iso13027 = new MonteRayCrossSectionHost(1);
	    	iso14000 = new MonteRayCrossSectionHost(1);
	    	iso18040 = new MonteRayCrossSectionHost(1);
	    	iso20000 = new MonteRayCrossSectionHost(1);
	    	iso26000 = new MonteRayCrossSectionHost(1);
	    	iso92234 = new MonteRayCrossSectionHost(1);
	    	iso92235 = new MonteRayCrossSectionHost(1);
	    	iso92238 = new MonteRayCrossSectionHost(1);

	    	metal     = new SimpleMaterialHost(3);
	    	air       = new SimpleMaterialHost(4);
	    	concrete  = new SimpleMaterialHost(8);

	        pMatList = new SimpleMaterialListHost(3,13);
	        pMatProps = new MonteRay_MaterialProperties;
		}

		void setup(){

			MonteRay_ReadLnk3dnt readerObject( "lnk3dnt/Room_cartesian_200x200x50.lnk3dnt" );
			readerObject.ReadMatData();

			pMatProps->disableReduction();
			pMatProps->setMaterialDescription( readerObject );

			pGrid = new GridBinsHost(readerObject);
			CHECK_EQUAL( 2000000, pGrid->getNumCells() );

			CHECK_CLOSE( -600.0, MonteRay::min(pGrid->getPtr(),0), 1e-2 );
			CHECK_CLOSE( -600.0, MonteRay::min(pGrid->getPtr(),1), 1e-2 );
			CHECK_CLOSE( -250.0, MonteRay::min(pGrid->getPtr(),2), 1e-2 );
			CHECK_CLOSE(  600.0, MonteRay::max(pGrid->getPtr(),0), 1e-2 );
			CHECK_CLOSE(  600.0, MonteRay::max(pGrid->getPtr(),1), 1e-2 );
			CHECK_CLOSE(  250.0, MonteRay::max(pGrid->getPtr(),2), 1e-2 );

	    	pTally = new gpuTallyHost( pGrid->getNumCells() );

	    	pGrid->copyToGPU();

	    	pTally->copyToGPU();

	        iso1001->read( "/usr/projects/mcatk/user/jsweezy/link_files/1001_MonteRayCrossSection.bin" );
	        iso6000->read( "/usr/projects/mcatk/user/jsweezy/link_files/6000_MonteRayCrossSection.bin" );
	        iso7014->read( "/usr/projects/mcatk/user/jsweezy/link_files/7014_MonteRayCrossSection.bin" );
	        iso8016->read( "/usr/projects/mcatk/user/jsweezy/link_files/8016_MonteRayCrossSection.bin" );
	        iso12000->read( "/usr/projects/mcatk/user/jsweezy/link_files/12000_MonteRayCrossSection.bin" );
	        iso13027->read( "/usr/projects/mcatk/user/jsweezy/link_files/13027_MonteRayCrossSection.bin" );
	        iso14000->read( "/usr/projects/mcatk/user/jsweezy/link_files/14000_MonteRayCrossSection.bin" );
	        iso18040->read( "/usr/projects/mcatk/user/jsweezy/link_files/18040_MonteRayCrossSection.bin" );
	        iso20000->read( "/usr/projects/mcatk/user/jsweezy/link_files/20000_MonteRayCrossSection.bin" );
	        iso26000->read( "/usr/projects/mcatk/user/jsweezy/link_files/26000_MonteRayCrossSection.bin" );
	        iso92234->read( "/usr/projects/mcatk/user/jsweezy/link_files/92234_MonteRayCrossSection.bin" );
	        iso92235->read( "/usr/projects/mcatk/user/jsweezy/link_files/92235_MonteRayCrossSection.bin" );
	        iso92238->read( "/usr/projects/mcatk/user/jsweezy/link_files/92238_MonteRayCrossSection.bin" );

	        metal->add(0, *iso92234, 1.025e-2 );
	        metal->add(1, *iso92235, 9.37683e-1 );
	        metal->add(2, *iso92238, 5.20671e-2 );
	        metal->normalizeFractions();
	        metal->copyToGPU();

	        air->add(0, *iso1001,  1.3851e-2 );
	        air->add(1, *iso7014,  7.66749e-1 );
	        air->add(2, *iso8016,  2.13141e-1 );
	        air->add(3, *iso18040, 6.24881e-3 );
	        air->normalizeFractions();
	        air->copyToGPU();

	        concrete->add(0, *iso1001,  1.06692e-1 );
	        concrete->add(1, *iso6000,  2.53507e-1 );
	        concrete->add(2, *iso8016,  4.45708e-1 );
	        concrete->add(3, *iso12000, 2.23318e-2 );
	        concrete->add(4, *iso13027, 2.97588e-3 );
	        concrete->add(5, *iso14000, 2.13364e-2 );
	        concrete->add(6, *iso20000, 1.39329e-1 );
	        concrete->add(7, *iso26000, 2.42237e-3 );
	        concrete->normalizeFractions();
	        concrete->copyToGPU();

	        const unsigned METAL_ID=2;
	        const unsigned AIR_ID=3;
	        const unsigned CONCRETE_ID=4;

	    	pMatList->add( 0, *metal, METAL_ID );
	    	pMatList->add( 1, *air, AIR_ID );
	    	pMatList->add( 2, *concrete, CONCRETE_ID );
	    	pMatList->copyToGPU();

	        pMatProps->renumberMaterialIDs(*pMatList);
	        pMatProps->copyToGPU();

	        iso1001->copyToGPU();
	        iso6000->copyToGPU();
	        iso7014->copyToGPU();
	        iso8016->copyToGPU();
	        iso12000->copyToGPU();
	        iso13027->copyToGPU();
	        iso14000->copyToGPU();
	        iso18040->copyToGPU();
	        iso20000->copyToGPU();
	        iso26000->copyToGPU();
	        iso92234->copyToGPU();
	        iso92235->copyToGPU();
	        iso92238->copyToGPU();

		}

		~ControllerSetup(){
			delete pGrid;
			delete pMatList;
			delete pMatProps;
			delete pTally;

			delete iso1001;  //1
			delete iso6000;  //2
			delete iso7014;  //3
			delete iso8016;  //4
			delete iso12000; //5
			delete iso13027; //6
			delete iso14000; //7
			delete iso18040; //8
			delete iso20000; //9
			delete iso26000; //10
			delete iso92234; //11
			delete iso92235; //12
			delete iso92238; //13

			delete metal;
			delete air;
			delete concrete;
		}

		GridBinsHost* pGrid;
		SimpleMaterialListHost* pMatList;
		MonteRay_MaterialProperties* pMatProps;
		gpuTallyHost* pTally;

    	MonteRayCrossSectionHost* iso1001;  //1
    	MonteRayCrossSectionHost* iso6000;  //2
    	MonteRayCrossSectionHost* iso7014;  //3
    	MonteRayCrossSectionHost* iso8016;  //4
    	MonteRayCrossSectionHost* iso12000; //5
    	MonteRayCrossSectionHost* iso13027; //6
    	MonteRayCrossSectionHost* iso14000; //7
    	MonteRayCrossSectionHost* iso18040; //8
    	MonteRayCrossSectionHost* iso20000; //9
    	MonteRayCrossSectionHost* iso26000; //10
    	MonteRayCrossSectionHost* iso92234; //11
    	MonteRayCrossSectionHost* iso92235; //12
    	MonteRayCrossSectionHost* iso92238; //13

        SimpleMaterialHost* metal;
        SimpleMaterialHost* air;
        SimpleMaterialHost* concrete;

	};


    TEST_FIXTURE(ControllerSetup, compare_with_mcatk ){
    	// compare with mcatk calling MonteRay -- validated against MCATK itself

    	setup();

    	// 256
    	//  64
    	// 192
    	CollisionPointController controller( 256,
    			256,
    			pGrid,
    			pMatList,
    			pMatProps,
    			pTally );
    	controller.setCapacity(40000*8U*20*10U);

    	size_t numCollisions = controller.readCollisionsFromFile( "/usr/projects/mcatk/user/jsweezy/link_files/Criticality_accident_collisions.bin" );

    	controller.sync();
    	CHECK_EQUAL( 56592340 , numCollisions );

    	pTally->copyToCPU();

    	gpuTallyHost benchmarkTally(1);
    	benchmarkTally.read( "/usr/projects/mcatk/user/jsweezy/link_files/Criticality_Accident_gpuTally_n20_particles40000_cycles1.bin" );

    	gpuTallyType_t maxdiff = 0.0;
    	unsigned numBenchmarkZeroNonMatching = 0;
    	unsigned numGPUZeroNonMatching = 0;
    	unsigned numZeroZero = 0;
    	for( unsigned i=0; i<benchmarkTally.size(); ++i ) {
    		if( pTally->getTally(i) > 0.0 &&  benchmarkTally.getTally(i) > 0.0 ){
    			gpuTallyType_t relDiff = 100.0*( benchmarkTally.getTally(i) - pTally->getTally(i) ) / benchmarkTally.getTally(i);
    			CHECK_CLOSE( 0.0, relDiff, 0.20 );
    			if( std::abs(relDiff) > maxdiff ){
    				maxdiff = std::abs(relDiff);
    			}
    		} else if( pTally->getTally(i) > 0.0) {
    			++numBenchmarkZeroNonMatching;
    		} else if( benchmarkTally.getTally(i) > 0.0) {
    			++numGPUZeroNonMatching;
    		} else {
    			++numZeroZero;
    		}
    	}

    	std::cout << "Debug:  maxdiff=" << maxdiff << "\n";
    	std::cout << "Debug:  tally size=" << benchmarkTally.size() << "\n";
//    	std::cout << "Debug:  tally from file size=" << pTally->size() << "\n";
//    	std::cout << "Debug:  numBenchmarkZeroNonMatching=" << numBenchmarkZeroNonMatching << "\n";
//    	std::cout << "Debug:        numGPUZeroNonMatching=" << numGPUZeroNonMatching << "\n";
//    	std::cout << "Debug:                num both zero=" << numZeroZero << "\n";

    	// timings on GTX TitanX GPU 256x256
    	//Debug: total gpuTime = 10.4895
    	//Debug: total cpuTime = 0.326181
    	//Debug: total wallTime = 10.4896


    	// timings on GTX TitanX GPU 1024x1024
    	// total gpuTime = 10.3725
    	// total cpuTime = 0.32771
    	// total wallTime = 10.3726


    }

}

}