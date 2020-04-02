#include <UnitTest++.h>

#include <iostream>
#include <functional>

#include "GPUUtilityFunctions.hh"
#include "ReadAndWriteFiles.hh"

#include "ExpectedPathLength.hh"
#include "MonteRay_timer.hh"
#include "RayListController.hh"
#include "Material.hh"
#include "MaterialList.hh"
#include "MaterialProperties.hh"
#include "MonteRay_ReadLnk3dnt.hh"
#include "RayListInterface.hh"
#include "CrossSection.hh"
#include "MonteRay_SpatialGrid.hh"

#include "BenchmarkTally.hh"

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

            pGrid = std::make_unique<MonteRay_SpatialGrid>(readerObject);
            CHECK_EQUAL( 952, pGrid->getNumCells() );

            matPropBuilder.renumberMaterialIDs(*pMatList);
            pMatProps = std::make_unique<MaterialProperties>(matPropBuilder.build());

        }

        void setupWithMovingMaterials(){

            MonteRay_ReadLnk3dnt readerObject( "lnk3dnt/zeus2.lnk3dnt" );
            readerObject.ReadMatData();

            MaterialProperties::Builder matPropBuilder{};
            matPropBuilder.disableMemoryReduction();
            matPropBuilder.setMaterialDescription( readerObject );

            pGrid = std::make_unique<MonteRay_SpatialGrid>(readerObject);
            CHECK_EQUAL( 952, pGrid->getNumCells() );

            matPropBuilder.renumberMaterialIDs(*pMatList);
            matPropBuilder.setVelocities( SimpleVector<gpuRayFloat_t>(pGrid->getNumCells(), 0.0) );
            pMatProps = std::make_unique<MaterialProperties>(matPropBuilder.build());

        }

        std::unique_ptr<MonteRay_SpatialGrid> pGrid;
        std::unique_ptr<CrossSectionList> pXsList;
        std::unique_ptr<MaterialList> pMatList;
        std::unique_ptr<MaterialProperties> pMatProps;
    };


    TEST_FIXTURE(ControllerSetup, compare_with_mcatk ){
    /*     // compare with mcatk calling MonteRay -- validated against MCATK itself */

        setup();

    /*     // 256 */
    /*     //  64 */
    /*     // 192 */
        unsigned nThreadsPerBlock = 1;
        unsigned nThreads = 256;
        unsigned capacity = 12118351;

        auto launchBounds = setLaunchBounds( nThreads, nThreadsPerBlock, capacity);

        std::cout << "Running ZEUS2 from collision file with nBlocks=" << launchBounds.first <<
                " nThreads=" << launchBounds.second << " collision buffer capacity=" << capacity << "\n";

        ExpectedPathLengthTally::Builder tallyBuilder;
        tallyBuilder.spatialBins(pGrid->size());

        auto controller = CollisionPointController::Builder()
                .nThreads(launchBounds.second)
                .nBlocks(launchBounds.first)
                .geometry(pGrid.get())
                .materialList(pMatList.get())
                .materialProperties(pMatProps.get())
                .expectedPathLengthTally(tallyBuilder.build())
                .capacity(capacity)
                .build();

        size_t numCollisions = controller.readCollisionsFromFile( "MonteRayTestFiles/Zeus2_Cylindrical_VCRE_rays.bin" );
        CHECK_EQUAL( 12118350 , numCollisions );

        controller.sync();

        auto benchmarkTally = readFromFile<BenchmarkTally>( "MonteRayTestFiles/Zeus2_Cylindrical_cpuTally_n5_particles1000_cycles50.bin");

        for( unsigned i=0; i<benchmarkTally.size(); ++i ) {

            if( controller.contribution(i) > 0.0 &&  benchmarkTally[i] > 0.0 ){
                gpuTallyType_t relDiff = 100.0*( benchmarkTally[i] - controller.contribution(i) ) / benchmarkTally[i];
                //printf( "Tally %u, MCATK=%e, MonteRay=%e, diff=%f\n", i, benchmarkTally[i], controller.contribution(i), relDiff  );
                CHECK_CLOSE( 0.0, relDiff, 0.034 );
            } else {
                CHECK_CLOSE( 0.0, controller.contribution(i), 1e-4);
                CHECK_CLOSE( 0.0, benchmarkTally[i], 1e-4);
            }
        }

        gpuTallyType_t maxdiff = 0.0;
        unsigned numBenchmarkZeroNonMatching = 0;
        unsigned numGPUZeroNonMatching = 0;
        unsigned numZeroZero = 0;
        for( unsigned i=0; i<benchmarkTally.size(); ++i ) {
            if( controller.contribution(i) > 0.0 &&  benchmarkTally[i] > 0.0 ){
                gpuTallyType_t relDiff = 100.0*( benchmarkTally[i] - controller.contribution(i) ) / benchmarkTally[i];
                if( std::abs(relDiff) > maxdiff ){
                    maxdiff = std::abs(relDiff);
                }
            } else if( controller.contribution(i) > 0.0) {
                ++numBenchmarkZeroNonMatching;
            } else if( benchmarkTally[i] > 0.0) {
                ++numGPUZeroNonMatching;
            } else {
                ++numZeroZero;
            }
        }

        std::cout << "Debug:  maxdiff=" << maxdiff << "\n";
        std::cout << "Debug:  tally from file size=" << benchmarkTally.size() << "\n";
        std::cout << "Debug:  numBenchmarkZeroNonMatching=" << numBenchmarkZeroNonMatching << "\n";
        std::cout << "Debug:        numGPUZeroNonMatching=" << numGPUZeroNonMatching << "\n";
        std::cout << "Debug:                num both zero=" << numZeroZero << "\n";

    /*     // timings on GTX 1080 Ti GP102 rev a1 Pascal GPU 390x256 */
    /*     // Debug: total gpuTime = 0.922235 */


    }

    TEST_FIXTURE(ControllerSetup, RayTraceWithMovingMaterials ){
        // compare with mcatk calling MonteRay -- validated against MCATK itself
        setupWithMovingMaterials();

        // 256
        //  64
        // 192
        unsigned nThreadsPerBlock = 1;
        unsigned nThreads = 256;
        unsigned capacity = 12118351;

        auto launchBounds = setLaunchBounds( nThreads, nThreadsPerBlock, capacity);

        std::cout << "Running ZEUS2 from collision file with nBlocks=" << launchBounds.first <<
                " nThreads=" << launchBounds.second << " collision buffer capacity=" << capacity << "\n";

        ExpectedPathLengthTally::Builder tallyBuilder;
        tallyBuilder.spatialBins(pGrid->size());

        auto controller = CollisionPointController::Builder()
                .nThreads(launchBounds.second)
                .nBlocks(launchBounds.first)
                .geometry(pGrid.get())
                .materialList(pMatList.get())
                .materialProperties(pMatProps.get())
                .expectedPathLengthTally(tallyBuilder.build())
                .capacity(capacity)
                .build();

        size_t numCollisions = controller.readCollisionsFromFile( "MonteRayTestFiles/Zeus2_Cylindrical_VCRE_rays.bin" );
        CHECK_EQUAL( 12118350 , numCollisions );

        controller.sync();

        auto benchmarkTally = readFromFile<BenchmarkTally>( "MonteRayTestFiles/Zeus2_Cylindrical_cpuTally_n5_particles1000_cycles50.bin");

        for( unsigned i=0; i<benchmarkTally.size(); ++i ) {

            if( controller.contribution(i) > 0.0 &&  benchmarkTally[i] > 0.0 ){
                gpuTallyType_t relDiff = 100.0*( benchmarkTally[i] - controller.contribution(i) ) / benchmarkTally[i];
                //printf( "Tally %u, MCATK=%e, MonteRay=%e, diff=%f\n", i, benchmarkTally[i], controller.contribution(i), relDiff  );
                CHECK_CLOSE( 0.0, relDiff, 0.034 );
            } else {
                CHECK_CLOSE( 0.0, controller.contribution(i), 1e-4);
                CHECK_CLOSE( 0.0, benchmarkTally[i], 1e-4);
            }
        }

        gpuTallyType_t maxdiff = 0.0;
        unsigned numBenchmarkZeroNonMatching = 0;
        unsigned numGPUZeroNonMatching = 0;
        unsigned numZeroZero = 0;
        for( unsigned i=0; i<benchmarkTally.size(); ++i ) {
            if( controller.contribution(i) > 0.0 &&  benchmarkTally[i] > 0.0 ){
                gpuTallyType_t relDiff = 100.0*( benchmarkTally[i] - controller.contribution(i) ) / benchmarkTally[i];
                if( std::abs(relDiff) > maxdiff ){
                    maxdiff = std::abs(relDiff);
                }
            } else if( controller.contribution(i) > 0.0) {
                ++numBenchmarkZeroNonMatching;
            } else if( benchmarkTally[i] > 0.0) {
                ++numGPUZeroNonMatching;
            } else {
                ++numZeroZero;
            }
        }

        std::cout << "Debug:  maxdiff=" << maxdiff << "\n";
        std::cout << "Debug:  tally from file size=" << benchmarkTally.size() << "\n";
        std::cout << "Debug:  numBenchmarkZeroNonMatching=" << numBenchmarkZeroNonMatching << "\n";
        std::cout << "Debug:        numGPUZeroNonMatching=" << numGPUZeroNonMatching << "\n";
        std::cout << "Debug:                num both zero=" << numZeroZero << "\n";

        // timings on GTX 1080 Ti GP102 rev a1 Pascal GPU 390x256
        // Debug: total gpuTime = 0.922235


    }

}

}
