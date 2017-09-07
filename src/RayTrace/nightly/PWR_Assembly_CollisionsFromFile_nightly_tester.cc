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

namespace PWR_Assembly_wCollisionFile_fi_tester{

using namespace MonteRay;

SUITE( PWR_Assembly_wCollisionFile_tester ) {

	class ControllerSetup {
	public:
		ControllerSetup(){

	    	cudaReset();
	    	gpuCheck();

	    	iso1001 = new MonteRayCrossSectionHost(1);
	    	iso5010 = new MonteRayCrossSectionHost(1);
	    	iso5011 = new MonteRayCrossSectionHost(1);
	    	iso6000 = new MonteRayCrossSectionHost(1);
	    	iso7014 = new MonteRayCrossSectionHost(1);
	    	iso8016 = new MonteRayCrossSectionHost(1);
	    	iso26000 = new MonteRayCrossSectionHost(1);
	    	iso40000 = new MonteRayCrossSectionHost(1);
	    	iso50000 = new MonteRayCrossSectionHost(1);
	    	iso92235 = new MonteRayCrossSectionHost(1);
	    	iso92238 = new MonteRayCrossSectionHost(1);

	        fuel      = new SimpleMaterialHost(3);
	        stainless = new SimpleMaterialHost(3);
	        b4c       = new SimpleMaterialHost(3);
	        water     = new SimpleMaterialHost(2);
	        graphite  = new SimpleMaterialHost(1);
	        soln      = new SimpleMaterialHost(5);

	        pMatList = new SimpleMaterialListHost(6,11);
	        pMatProps = new MonteRay_MaterialProperties;
		}

		void setup(){

			MonteRay_ReadLnk3dnt readerObject( "lnk3dnt/pwr16x16_assembly_fine.lnk3dnt" );
			readerObject.ReadMatData();

			pMatProps->disableReduction();
			pMatProps->setMaterialDescription( readerObject );

			pGrid = new GridBinsHost(readerObject);
			CHECK_EQUAL( 584820, pGrid->getNumCells() );

			CHECK_CLOSE( -40.26, MonteRay::min(pGrid->getPtr(),0), 1e-2 );
			CHECK_CLOSE( -40.26, MonteRay::min(pGrid->getPtr(),1), 1e-2 );
			CHECK_CLOSE( -80.00, MonteRay::min(pGrid->getPtr(),2), 1e-2 );
			CHECK_CLOSE(  40.26, MonteRay::max(pGrid->getPtr(),0), 1e-2 );
			CHECK_CLOSE(  40.26, MonteRay::max(pGrid->getPtr(),1), 1e-2 );
			CHECK_CLOSE(  80.00, MonteRay::max(pGrid->getPtr(),2), 1e-2 );

	    	pTally = new gpuTallyHost( pGrid->getNumCells() );

	    	pGrid->copyToGPU();

	    	pTally->copyToGPU();

	        iso1001->read( "MonteRayTestFiles/1001_MonteRayCrossSection.bin" );;
	        iso5010->read( "MonteRayTestFiles/5010_MonteRayCrossSection.bin" );;
	        iso5011->read( "MonteRayTestFiles/5011_MonteRayCrossSection.bin" );;
	        iso6000->read( "MonteRayTestFiles/6000_MonteRayCrossSection.bin" );;
	        iso7014->read( "MonteRayTestFiles/7014_MonteRayCrossSection.bin" );;
	        iso8016->read( "MonteRayTestFiles/8016_MonteRayCrossSection.bin" );;
	        iso26000->read( "MonteRayTestFiles/26000_MonteRayCrossSection.bin" );;
	        iso40000->read( "MonteRayTestFiles/40000_MonteRayCrossSection.bin" );;
	        iso50000->read( "MonteRayTestFiles/50000_MonteRayCrossSection.bin" );;
	        iso92235->read( "MonteRayTestFiles/92235_MonteRayCrossSection.bin" );;
	        iso92238->read( "MonteRayTestFiles/92238_MonteRayCrossSection.bin" );;

	        fuel->add(0, *iso8016,  2.0 );
	        fuel->add(1, *iso92235, 0.05 );
	        fuel->add(2, *iso92238, 0.95 );
	        fuel->normalizeFractions();
	        fuel->copyToGPU();

	        stainless->add(0, *iso26000, 8.17151e-3 );
	        stainless->add(1, *iso40000, 0.979604 );
	        stainless->add(2, *iso50000, 0.0122247 );
	        stainless->normalizeFractions();
	        stainless->copyToGPU();

	        b4c->add(0, *iso5010, 0.16 );
	        b4c->add(1, *iso5011, 0.64 );
	        b4c->add(2, *iso6000, 0.204 );
	        b4c->normalizeFractions();
	        b4c->copyToGPU();

	        water->add(0, *iso1001, 2.0 );
	        water->add(1, *iso8016, 1.0 );
	        water->normalizeFractions();
	        water->copyToGPU();

	        graphite->add(0, *iso6000, 1.0 );
	        graphite->copyToGPU();

	        soln->add(0, *iso1001,  5.7745e-1 );
	        soln->add(1, *iso7014,  2.9900e-2 );
	        soln->add(2, *iso8016,  3.8536e-1 );
	        soln->add(3, *iso92235, 7.1826e-4 );
	        soln->add(4, *iso92238, 6.5700e-3 );
	        soln->normalizeFractions();
	        soln->copyToGPU();

	        const unsigned FUEL_ID=2;
	        const unsigned STAINLESS_ID=3;
	        const unsigned B4C_ID=4;
	        const unsigned WATER_ID=5;
	        const unsigned GRAPHITE_ID=6;
	        const unsigned SOLN_ID=7;

	    	pMatList->add( 0, *fuel, FUEL_ID );
	    	pMatList->add( 1, *stainless, STAINLESS_ID );
	    	pMatList->add( 2, *b4c, B4C_ID );
	    	pMatList->add( 3, *water, WATER_ID );
	    	pMatList->add( 4, *graphite, GRAPHITE_ID );
	    	pMatList->add( 5, *soln, SOLN_ID );
	    	pMatList->copyToGPU();

	        pMatProps->renumberMaterialIDs(*pMatList);
	        pMatProps->copyToGPU();

	        iso1001->copyToGPU();
	        iso5010->copyToGPU();
	        iso5011->copyToGPU();
	        iso6000->copyToGPU();
	        iso7014->copyToGPU();
	        iso8016->copyToGPU();
	        iso26000->copyToGPU();
	        iso40000->copyToGPU();
	        iso50000->copyToGPU();
	        iso92235->copyToGPU();
	        iso92238->copyToGPU();
		}

		~ControllerSetup(){
			delete pGrid;
			delete pMatList;
			delete pMatProps;
			delete pTally;

			delete iso1001;
			delete iso5010;
			delete iso5011;
			delete iso6000;
			delete iso7014;
			delete iso8016;
			delete iso26000;
			delete iso40000;
			delete iso50000;
			delete iso92235;
			delete iso92238;

			delete fuel;
			delete stainless;
			delete b4c;
			delete water;
			delete graphite;
			delete soln;
		}

		GridBinsHost* pGrid;
		SimpleMaterialListHost* pMatList;
		MonteRay_MaterialProperties* pMatProps;
		gpuTallyHost* pTally;

    	MonteRayCrossSectionHost* iso1001;
    	MonteRayCrossSectionHost* iso5010;
    	MonteRayCrossSectionHost* iso5011;
    	MonteRayCrossSectionHost* iso6000;
    	MonteRayCrossSectionHost* iso7014;
    	MonteRayCrossSectionHost* iso8016;
    	MonteRayCrossSectionHost* iso26000;
    	MonteRayCrossSectionHost* iso40000;
    	MonteRayCrossSectionHost* iso50000;
    	MonteRayCrossSectionHost* iso92235;
    	MonteRayCrossSectionHost* iso92238;

        SimpleMaterialHost* fuel;
        SimpleMaterialHost* stainless;
        SimpleMaterialHost* b4c;
        SimpleMaterialHost* water;
        SimpleMaterialHost* graphite;
        SimpleMaterialHost* soln;

	};


    TEST_FIXTURE(ControllerSetup, compare_with_mcatk ){
    	// compare with mcatk calling MonteRay -- validated against MCATK itself

    	setup();

    	// 256
    	//  64
    	// 192
    	unsigned nBlocks = 256;
    	unsigned nThreads = 256;
    	unsigned capacity = std::min( 64000000U, 40000*8U*8*10U );
    	capacity = 16698849;
#ifdef TITANX_MAXWELL_GPU
        nThreads = 416;
        nBlocks  = 16384;
//        nThreads  = std::min( 1024U, (capacity-1) / ( 2* ( nBlocks -1 ) ));
//        nThreads = (( nThreads + 32 -1 ) / 32 ) *32;

        nBlocks = std::min(( capacity + nThreads*4 -1 ) / (nThreads*4), 65535U);

#endif
#ifdef TESLA_K40_GPU
//        nBlocks = 16384;
//        nThreads  = (capacity-1) / ( nBlocks -1 );
//        nThreads = (( nThreads + 32 -1 ) / 32 ) *32;
//        nThreads = std::min( 1024U, nThreads );

        unsigned N = 4;
        nThreads = 416;
        nBlocks = std::min(( capacity + N*nThreads -1 ) / (N*nThreads), 65535U);
#endif
#ifdef K420_GPU
    	nBlocks = 128;
    	nThreads = 128;
    	capacity = 1000000;
#endif
    	std::cout << "Running PWR_Assembly from collision file with nBlocks=" << nBlocks <<
    			     " nThreads=" << nThreads << " collision buffer capacity=" << capacity << "\n";

    	CollisionPointController controller( nBlocks,
    			nThreads,
    			pGrid,
    			pMatList,
    			pMatProps,
    			pTally );

    	controller.setCapacity(capacity);

    	size_t numCollisions = controller.readCollisionsFromFile( "MonteRayTestFiles/PWR_assembly_collisions.bin" );
    	CHECK_EQUAL( 16698848 , numCollisions );

    	controller.sync();
    	pTally->copyToCPU();

    	gpuTallyHost benchmarkTally(1);
    	benchmarkTally.read( "MonteRayTestFiles/PWR_Assembly_gpuTally_n8_particles40000_cycles1.bin" );

    	for( unsigned i=0; i<benchmarkTally.size(); ++i ) {
    		if( pTally->getTally(i) > 0.0 &&  benchmarkTally.getTally(i) > 0.0 ){
    			gpuTallyType_t relDiff = 100.0*( benchmarkTally.getTally(i) - pTally->getTally(i) ) / benchmarkTally.getTally(i);
    			CHECK_CLOSE( 0.0, relDiff, 0.34 );
    		} else {
    			CHECK_CLOSE( 0.0, pTally->getTally(i), 1e-4);
    			CHECK_CLOSE( 0.0, benchmarkTally.getTally(i), 1e-4);
    		}
    	}

    	gpuTallyType_t maxdiff = 0.0;
    	unsigned numBenchmarkZeroNonMatching = 0;
    	unsigned numGPUZeroNonMatching = 0;
    	unsigned numZeroZero = 0;
    	for( unsigned i=0; i<benchmarkTally.size(); ++i ) {
    		if( pTally->getTally(i) > 0.0 &&  benchmarkTally.getTally(i) > 0.0 ){
    			gpuTallyType_t relDiff = 100.0*( benchmarkTally.getTally(i) - pTally->getTally(i) ) / benchmarkTally.getTally(i);
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
      	std::cout << "Debug:  tally from file size=" << pTally->size() << "\n";
      	std::cout << "Debug:  numBenchmarkZeroNonMatching=" << numBenchmarkZeroNonMatching << "\n";
       	std::cout << "Debug:        numGPUZeroNonMatching=" << numGPUZeroNonMatching << "\n";
       	std::cout << "Debug:                num both zero=" << numZeroZero << "\n";

    	// timings on GTX TitanX GPU 256x256
       	// Debug: total gpuTime = 6.4024
       	// Debug: total cpuTime = 0.0860779
       	// Debug: total wallTime = 6.40247

       	// timings on GTX TitanX GPU 1024x1024
       	// Debug: total gpuTime = 6.37461
       	// Debug: total cpuTime = 0.084251
       	// Debug: total wallTime = 6.37465

       	// timings on GTX TitanX GPU 16384x1024
       	// Debug: total gpuTime = 6.1284
       	// Debug: total cpuTime = 0.0829004
       	// Debug: total wallTime = 6.12846

       	// timings on GTX TitanX GPU 16384x416
       	// Debug: total gpuTime = 5.68947
       	// Debug: total cpuTime = 0.0825951
       	// Debug: total wallTime = 5.68952

       	// timings on Tesla K40c GPU 10036x416
       	// Debug: total gpuTime = 12.709
       	// Debug: total cpuTime = 0.118516
       	// Debug: total wallTime = 12.7091

       	// timing on Nvidia Quandro K420 128x128'
       	// total gpuTime = 101.266
       	// total cpuTime = 0.904188
       	// ntotal wallTime = 101.267


    }

}

}
