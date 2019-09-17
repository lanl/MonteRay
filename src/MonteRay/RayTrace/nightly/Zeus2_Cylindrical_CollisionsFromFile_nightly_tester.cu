#include <UnitTest++.h>

#include <iostream>
#include <functional>

#include "GPUUtilityFunctions.hh"
#include "ReadAndWriteFiles.hh"

#include "gpuTally.hh"
#include "ExpectedPathLength.hh"
#include "MonteRay_timer.hh"
#include "RayListController.hh"
#include "MonteRayMaterial.hh"
#include "MonteRayMaterialList.hh"
#include "MaterialProperties.hh"
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

            const unsigned METAL_ID=2;
            const unsigned ALUMINUM_ID=3;
            const unsigned GRAPHITE_ID=4;
            const unsigned COPPER_ID=5;

            cudaReset();
            gpuCheck();
            setCudaStackSize( 2*1024 );


            CrossSectionList::Builder xsListBuilder;

            auto xsBuilder = CrossSectionBuilder();
            readInPlaceFromFile( "MonteRayTestFiles/6000-70c_MonteRayCrossSection.bin", xsBuilder);
            xsBuilder.setZAID(6000);
            xsListBuilder.add(xsBuilder.build());
            readInPlaceFromFile( "MonteRayTestFiles/13027-70c_MonteRayCrossSection.bin", xsBuilder);
            xsBuilder.setZAID(13027);
            xsListBuilder.add(xsBuilder.build());
            readInPlaceFromFile( "MonteRayTestFiles/29063-70c_MonteRayCrossSection.bin", xsBuilder );
            xsBuilder.setZAID(29063);
            xsListBuilder.add(xsBuilder.build());
            readInPlaceFromFile( "MonteRayTestFiles/29065-70c_MonteRayCrossSection.bin", xsBuilder );
            xsBuilder.setZAID(29065);
            xsListBuilder.add(xsBuilder.build());
            readInPlaceFromFile( "MonteRayTestFiles/92234-70c_MonteRayCrossSection.bin", xsBuilder);
            xsBuilder.setZAID(92234);
            xsListBuilder.add(xsBuilder.build());
            readInPlaceFromFile( "MonteRayTestFiles/92235-70c_MonteRayCrossSection.bin", xsBuilder);
            xsBuilder.setZAID(92235);
            xsListBuilder.add(xsBuilder.build());
            readInPlaceFromFile( "MonteRayTestFiles/92236-70c_MonteRayCrossSection.bin", xsBuilder);
            xsBuilder.setZAID(92236);
            xsListBuilder.add(xsBuilder.build());
            readInPlaceFromFile( "MonteRayTestFiles/92238-70c_MonteRayCrossSection.bin", xsBuilder);
            xsBuilder.setZAID(92238);
            xsListBuilder.add(xsBuilder.build());

            pXsList = std::make_unique<CrossSectionList>(xsListBuilder.build());

            MaterialList::Builder matListBuilder{};

            auto mb = Material::make_builder(*pXsList);

            mb.addIsotope(4.9576e-4, 92234);
            mb.addIsotope(4.4941e-2, 92235);
            mb.addIsotope(1.5931e-4, 92236);
            mb.addIsotope(2.5799e-3, 92238);
            matListBuilder.addMaterial( METAL_ID, mb.build() );

            mb.addIsotope(1.0, 13027);
            matListBuilder.addMaterial( ALUMINUM_ID, mb.build() );

            mb.addIsotope(1.0, 6000);
            matListBuilder.addMaterial( GRAPHITE_ID, mb.build() );

            mb.addIsotope(5.7325e-2, 29063);
            mb.addIsotope(2.5550e-2, 29065);
            matListBuilder.addMaterial( COPPER_ID, mb.build() );

            pMatList = std::make_unique<MaterialList>(matListBuilder.build());
        }

        void setup(){

            MonteRay_ReadLnk3dnt readerObject( "lnk3dnt/zeus2.lnk3dnt" );
            readerObject.ReadMatData();

            MaterialProperties::Builder matPropBuilder{};
            matPropBuilder.disableMemoryReduction();
            matPropBuilder.setMaterialDescription( readerObject );

            pGrid = new Grid_t(readerObject);
            CHECK_EQUAL( 952, pGrid->getNumCells() );

            pTally = new gpuTallyHost( pGrid->getNumCells() );

            pGrid->copyToGPU();

            pTally->copyToGPU();

            matPropBuilder.renumberMaterialIDs(*pMatList);
            pMatProps = std::make_unique<MaterialProperties>(matPropBuilder.build());

        }

        ~ControllerSetup(){
            delete pGrid;
            delete pTally;
        }

        Grid_t* pGrid;
        std::unique_ptr<CrossSectionList> pXsList;
        std::unique_ptr<MaterialList> pMatList;
        std::unique_ptr<MaterialProperties> pMatProps;
        gpuTallyHost* pTally;

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
                pMatList.get(),
                pMatProps.get(),
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

        // timings on GTX 1080 Ti GP102 rev a1 Pascal GPU 390x256
        // Debug: total gpuTime = 0.922235


    }

}

}
