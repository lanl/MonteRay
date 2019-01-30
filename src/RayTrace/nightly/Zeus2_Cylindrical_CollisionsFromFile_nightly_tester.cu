#include <UnitTest++.h>

#include <iostream>
#include <functional>

#include "GPUUtilityFunctions.hh"

#include "gpuTally.hh"
#include "ExpectedPathLength.hh"
#include "MonteRay_timer.hh"
#include "RayListController.hh"
#include "MonteRayMaterial.hh"
#include "MonteRayMaterialList.hh"
#include "MonteRay_MaterialProperties.hh"
#include "MonteRay_ReadLnk3dnt.hh"
#include "RayListInterface.hh"
#include "MonteRayCrossSection.hh"
#include "MonteRay_SpatialGrid.hh"

namespace Zeus2_Cylindrical_wCollisionFile_fi_tester{

using namespace MonteRay;

SUITE(  Zeus2_Cylindrical_wCollisionFile_tester ) {
    typedef MonteRay_SpatialGrid Grid_t;

    class ControllerSetup {
    public:
        ControllerSetup(){

            cudaReset();
            gpuCheck();
            setCudaStackSize( 2*1024 );
            iso6000 = new MonteRayCrossSectionHost(1);
            iso13027 = new MonteRayCrossSectionHost(1);
            iso29063 = new MonteRayCrossSectionHost(1);
            iso29065 = new MonteRayCrossSectionHost(1);
            iso92234 = new MonteRayCrossSectionHost(1);
            iso92235 = new MonteRayCrossSectionHost(1);
            iso92236 = new MonteRayCrossSectionHost(1);
            iso92238 = new MonteRayCrossSectionHost(1);

            metal    = new MonteRayMaterialHost(4);
            aluminum = new MonteRayMaterialHost(1);
            graphite = new MonteRayMaterialHost(1);
            copper   = new MonteRayMaterialHost(2);

            pMatList = new MonteRayMaterialListHost(4,8);
            pMatProps = new MonteRay_MaterialProperties;
        }

        void setup(){

            MonteRay_ReadLnk3dnt readerObject( "lnk3dnt/zeus2.lnk3dnt" );
            readerObject.ReadMatData();

            pMatProps->disableMemoryReduction();
            pMatProps->setMaterialDescription( readerObject );

            pGrid = new Grid_t(readerObject);
            CHECK_EQUAL( 952, pGrid->getNumCells() );

            pTally = new gpuTallyHost( pGrid->getNumCells() );

            pGrid->copyToGPU();

            pTally->copyToGPU();

            iso6000->read( "MonteRayTestFiles/6000-70c_MonteRayCrossSection.bin" );
            iso13027->read( "MonteRayTestFiles/13027-70c_MonteRayCrossSection.bin" );
            iso29063->read( "MonteRayTestFiles/29063-70c_MonteRayCrossSection.bin" );
            iso29065->read( "MonteRayTestFiles/29065-70c_MonteRayCrossSection.bin" );
            iso92234->read( "MonteRayTestFiles/92234-70c_MonteRayCrossSection.bin" );
            iso92235->read( "MonteRayTestFiles/92235-70c_MonteRayCrossSection.bin" );
            iso92236->read( "MonteRayTestFiles/92236-70c_MonteRayCrossSection.bin" );
            iso92238->read( "MonteRayTestFiles/92238-70c_MonteRayCrossSection.bin" );

            metal->add(0, *iso92234, 4.9576e-4 );
            metal->add(1, *iso92235, 4.4941e-2 );
            metal->add(2, *iso92236, 1.5931e-4 );
            metal->add(3, *iso92238, 2.5799e-3 );
            metal->normalizeFractions();
            metal->copyToGPU();

            aluminum->add(0, *iso13027, 1.0 );
            aluminum->normalizeFractions();
            aluminum->copyToGPU();

            graphite->add(0, *iso6000, 1.0 );
            graphite->copyToGPU();

            copper->add(0, *iso29063,  5.7325e-2 );
            copper->add(1, *iso29065,  2.5550e-2 );
            copper->normalizeFractions();
            copper->copyToGPU();

            const unsigned METAL_ID=2;
            const unsigned ALUMINUM_ID=3;
            const unsigned GRAPHITE_ID=4;
            const unsigned COPPER_ID=5;

            pMatList->add( 0, *metal, METAL_ID );
            pMatList->add( 1, *aluminum, ALUMINUM_ID );
            pMatList->add( 2, *graphite, GRAPHITE_ID );
            pMatList->add( 3, *copper, COPPER_ID );
            pMatList->copyToGPU();

            pMatProps->renumberMaterialIDs(*pMatList);
            pMatProps->copyToGPU();

            iso6000->copyToGPU();
            iso13027->copyToGPU();
            iso29063->copyToGPU();
            iso29065->copyToGPU();
            iso92234->copyToGPU();
            iso92235->copyToGPU();
            iso92236->copyToGPU();
            iso92238->copyToGPU();
        }

        ~ControllerSetup(){
            delete pGrid;
            delete pMatList;
            delete pMatProps;
            delete pTally;

            delete iso6000;
            delete iso13027;
            delete iso29063;
            delete iso29065;
            delete iso92234;
            delete iso92235;
            delete iso92236;
            delete iso92238;

            delete metal;
            delete aluminum;
            delete graphite;
            delete copper;
        }

        Grid_t* pGrid;
        MonteRayMaterialListHost* pMatList;
        MonteRay_MaterialProperties* pMatProps;
        gpuTallyHost* pTally;

        MonteRayCrossSectionHost* iso6000;
        MonteRayCrossSectionHost* iso13027;
        MonteRayCrossSectionHost* iso29063;
        MonteRayCrossSectionHost* iso29065;
        MonteRayCrossSectionHost* iso92234;
        MonteRayCrossSectionHost* iso92235;
        MonteRayCrossSectionHost* iso92236;
        MonteRayCrossSectionHost* iso92238;

        MonteRayMaterialHost* metal;
        MonteRayMaterialHost* aluminum;
        MonteRayMaterialHost* graphite;
        MonteRayMaterialHost* copper;
    };


    TEST_FIXTURE(ControllerSetup, compare_with_mcatk ){
        // compare with mcatk calling MonteRay -- validated against MCATK itself

        setup();

        // 256
        //  64
        // 192
        unsigned nThreadsPerBlock = 1;
        unsigned nThreads = 256;
        unsigned capacity = std::min( 64000000U, 40000*8U*8*10U );
        capacity = 12118351;

        auto launchBounds = setLaunchBounds( nThreads, nThreadsPerBlock, capacity);

        std::cout << "Running ZEUS2 from collision file with nBlocks=" << launchBounds.first <<
                " nThreads=" << launchBounds.second << " collision buffer capacity=" << capacity << "\n";

        CollisionPointController<Grid_t> controller(
                nThreadsPerBlock,
                nThreads,
                pGrid,
                pMatList,
                pMatProps,
                pTally );

        controller.setCapacity(capacity);

        size_t numCollisions = controller.readCollisionsFromFile( "MonteRayTestFiles/Zeus2_Cylindrical_VCRE_rays.bin" );
        CHECK_EQUAL( 12118350 , numCollisions );

        controller.sync();
        pTally->copyToCPU();

        gpuTallyHost benchmarkTally(952);
        benchmarkTally.read( "MonteRayTestFiles/Zeus2_Cylindrical_cpuTally_n5_particles1000_cycles50.bin" );

        for( unsigned i=0; i<benchmarkTally.size(); ++i ) {

            if( pTally->getTally(i) > 0.0 &&  benchmarkTally.getTally(i) > 0.0 ){
                gpuTallyType_t relDiff = 100.0*( benchmarkTally.getTally(i) - pTally->getTally(i) ) / benchmarkTally.getTally(i);
                //printf( "Tally %u, MCATK=%e, MonteRay=%e, diff=%f\n", i, benchmarkTally.getTally(i), pTally->getTally(i), relDiff  );
                CHECK_CLOSE( 0.0, relDiff, 0.034 );
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
        std::cout << "Debug:  tally size=" << pTally->size()  << "\n";
        std::cout << "Debug:  tally from file size=" << benchmarkTally.size() << "\n";
        std::cout << "Debug:  numBenchmarkZeroNonMatching=" << numBenchmarkZeroNonMatching << "\n";
        std::cout << "Debug:        numGPUZeroNonMatching=" << numGPUZeroNonMatching << "\n";
        std::cout << "Debug:                num both zero=" << numZeroZero << "\n";

        // timings on GTX TitanX Pascal GPU 390x256
        // Debug: total gpuTime = 5.248





    }

}

}
