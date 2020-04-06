#include <UnitTest++.h>

#include <iostream>
#include <functional>
#include <memory>

#include "GPUUtilityFunctions.hh"
#include "ReadAndWriteFiles.hh"

#include "MonteRay_SpatialGrid.hh"
#include "ExpectedPathLength.hh"
#include "MonteRay_timer.hh"
#include "RayListController.hh"
#include "Material.hh"
#include "MaterialList.hh"
#include "MaterialProperties.hh"
#include "MonteRay_ReadLnk3dnt.hh"
#include "RayListInterface.hh"
#include "CrossSection.hh"
#include "BenchmarkTally.hh"

namespace PWR_Assembly_wCollisionFile_fi_tester{

using namespace MonteRay;

SUITE( PWR_Assembly_wCollisionFile_tester ) {

    class ControllerSetup {
    public:
        ControllerSetup():
          benchmarkTally(readFromFile<BenchmarkTally>( "MonteRayTestFiles/PWR_Assembly_gpuTally_n8_particles40000_cycles1.bin")) 
        {

            const unsigned FUEL_ID=2;
            const unsigned STAINLESS_ID=3;
            const unsigned B4C_ID=4;
            const unsigned WATER_ID=5;
            const unsigned GRAPHITE_ID=6;
            const unsigned SOLN_ID=7;
            cudaReset();
            gpuCheck();

            CrossSectionList::Builder xsListBuilder;

            auto xsBuilder = CrossSectionBuilder();
            readInPlaceFromFile( "MonteRayTestFiles/1001-70c_MonteRayCrossSection.bin", xsBuilder);
            xsBuilder.setZAID(1001);
            xsListBuilder.add(xsBuilder.build());
            readInPlaceFromFile( "MonteRayTestFiles/6000-70c_MonteRayCrossSection.bin", xsBuilder);
            xsBuilder.setZAID(6000);
            xsListBuilder.add(xsBuilder.build());
            readInPlaceFromFile( "MonteRayTestFiles/5010-70c_MonteRayCrossSection.bin", xsBuilder );;
            xsBuilder.setZAID(5010);
            xsListBuilder.add(xsBuilder.build());
            readInPlaceFromFile( "MonteRayTestFiles/5011-70c_MonteRayCrossSection.bin", xsBuilder );;
            xsBuilder.setZAID(5011);
            xsListBuilder.add(xsBuilder.build());
            readInPlaceFromFile( "MonteRayTestFiles/7014-70c_MonteRayCrossSection.bin", xsBuilder);
            xsBuilder.setZAID(7014);
            xsListBuilder.add(xsBuilder.build());
            readInPlaceFromFile( "MonteRayTestFiles/8016-70c_MonteRayCrossSection.bin", xsBuilder);
            xsBuilder.setZAID(8016);
            xsListBuilder.add(xsBuilder.build());
            readInPlaceFromFile( "MonteRayTestFiles/26000-55c_MonteRayCrossSection.bin", xsBuilder);
            xsBuilder.setZAID(26000);
            xsListBuilder.add(xsBuilder.build());
            readInPlaceFromFile( "MonteRayTestFiles/40000-66c_MonteRayCrossSection.bin", xsBuilder );;
            xsBuilder.setZAID(40000);
            xsListBuilder.add(xsBuilder.build());
            readInPlaceFromFile( "MonteRayTestFiles/50000-42c_MonteRayCrossSection.bin", xsBuilder );;
            xsBuilder.setZAID(50000);
            xsListBuilder.add(xsBuilder.build());
            readInPlaceFromFile( "MonteRayTestFiles/92235-70c_MonteRayCrossSection.bin", xsBuilder);
            xsBuilder.setZAID(92235);
            xsListBuilder.add(xsBuilder.build());
            readInPlaceFromFile( "MonteRayTestFiles/92238-70c_MonteRayCrossSection.bin", xsBuilder);
            xsBuilder.setZAID(92238);
            xsListBuilder.add(xsBuilder.build());

            pXsList = std::make_unique<CrossSectionList>(xsListBuilder.build());

            MaterialList::Builder matListBuilder{};

            auto mb = Material::make_builder(*pXsList);

            mb.addIsotope(2.0, 8016);
            mb.addIsotope(0.05, 92235);
            mb.addIsotope(0.95, 92238);
            matListBuilder.addMaterial( FUEL_ID, mb.build() );

            mb.addIsotope(8.17151e-3, 26000);
            mb.addIsotope(0.979604, 40000);
            mb.addIsotope(0.0122247, 50000);
            matListBuilder.addMaterial( STAINLESS_ID, mb.build() );

            mb.addIsotope(0.16, 5010);
            mb.addIsotope(0.64, 5011);
            mb.addIsotope(0.204, 6000);
            matListBuilder.addMaterial( B4C_ID, mb.build() );

            mb.addIsotope(2.0, 1001);
            mb.addIsotope(1.0, 8016);
            matListBuilder.addMaterial( WATER_ID, mb.build() );

            mb.addIsotope(1.0, 6000);
            matListBuilder.addMaterial( GRAPHITE_ID, mb.build() );

            mb.addIsotope(5.7745e-1, 1001);
            mb.addIsotope(2.9900e-2, 7014);
            mb.addIsotope(3.8536e-1, 8016);
            mb.addIsotope(7.1826e-4, 92235);
            mb.addIsotope(6.5700e-3, 92238);
            matListBuilder.addMaterial( SOLN_ID, mb.build() );

            pMatList = std::make_unique<MaterialList>(matListBuilder.build());


        }

        void setup(){
          MonteRay_ReadLnk3dnt readerObject( "lnk3dnt/pwr16x16_assembly_fine.lnk3dnt" );
          readerObject.ReadMatData();


          pGrid = std::make_unique<MonteRay_SpatialGrid>(readerObject);
          CHECK_EQUAL( 584820, pGrid->getNumCells() );

          CHECK_CLOSE( -40.26, pGrid->getMinVertex(0), 1e-2 );
          CHECK_CLOSE( -40.26, pGrid->getMinVertex(1), 1e-2 );
          CHECK_CLOSE( -80.00, pGrid->getMinVertex(2), 1e-2 );
          CHECK_CLOSE(  40.26, pGrid->getMaxVertex(0), 1e-2 );
          CHECK_CLOSE(  40.26, pGrid->getMaxVertex(1), 1e-2 );
          CHECK_CLOSE(  80.00, pGrid->getMaxVertex(2), 1e-2 );

          MaterialProperties::Builder matPropBuilder{};
          matPropBuilder.disableMemoryReduction();
          matPropBuilder.setMaterialDescription( readerObject );

          matPropBuilder.renumberMaterialIDs(*pMatList);
          pMatProps = std::make_unique<MaterialProperties>(matPropBuilder.build());
        }

        void setupWithMovingMaterials(){
          MonteRay_ReadLnk3dnt readerObject( "lnk3dnt/pwr16x16_assembly_fine.lnk3dnt" );
          readerObject.ReadMatData();


          pGrid = std::make_unique<MonteRay_SpatialGrid>(readerObject);
          CHECK_EQUAL( 584820, pGrid->getNumCells() );

          CHECK_CLOSE( -40.26, pGrid->getMinVertex(0), 1e-2 );
          CHECK_CLOSE( -40.26, pGrid->getMinVertex(1), 1e-2 );
          CHECK_CLOSE( -80.00, pGrid->getMinVertex(2), 1e-2 );
          CHECK_CLOSE(  40.26, pGrid->getMaxVertex(0), 1e-2 );
          CHECK_CLOSE(  40.26, pGrid->getMaxVertex(1), 1e-2 );
          CHECK_CLOSE(  80.00, pGrid->getMaxVertex(2), 1e-2 );

          MaterialProperties::Builder matPropBuilder{};
          matPropBuilder.disableMemoryReduction();
          matPropBuilder.setMaterialDescription( readerObject );

          matPropBuilder.renumberMaterialIDs(*pMatList);
          matPropBuilder.setVelocities( SimpleVector<gpuRayFloat_t>(pGrid->getNumCells(), 0.0) );
          pMatProps = std::make_unique<MaterialProperties>(matPropBuilder.build());
        }

        BenchmarkTally benchmarkTally;
        std::unique_ptr<MonteRay_SpatialGrid> pGrid;
        std::unique_ptr<CrossSectionList> pXsList;
        std::unique_ptr<MaterialList> pMatList;
        std::unique_ptr<MaterialProperties> pMatProps;
    };

    TEST_FIXTURE(ControllerSetup, compare_with_mcatk ){
        // compare with mcatk calling MonteRay -- validated against MCATK itself

        setup();

        // 256
        //  64
        // 192
        unsigned nThreadsPerBlock = 1;
        unsigned nThreads= 256;
        unsigned capacity = std::min( 64000000U, 40000*8U*8*10U );
        /* unsigned capacity = 1000000; */

        auto launchBounds = setLaunchBounds( nThreads, nThreadsPerBlock, capacity);

        std::cout << "Running PWR_Assembly from collision file with nBlocks = " << launchBounds.first <<
                " nThreads = " << launchBounds.second << " collision buffer capacity = " << capacity << "\n";

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


        size_t numCollisions = controller.readCollisionsFromFile( "MonteRayTestFiles/PWR_assembly_collisions.bin" );
        CHECK_EQUAL( 16698848 , numCollisions );

        controller.sync();

        for( unsigned i=0; i<benchmarkTally.size(); ++i ) {
            if( controller.contribution(i) > 0.0 &&  benchmarkTally[i] > 0.0 ){
                gpuTallyType_t relDiff = 100.0*( benchmarkTally[i] - controller.contribution(i) ) / benchmarkTally[i];
                CHECK_CLOSE( 0.0, relDiff, 0.34 );
            } else {
                CHECK_CLOSE( 0.0, controller.contribution(i), 1e-4);
                CHECK_CLOSE( 0.0, benchmarkTally[i], 1e-4);
            }
        }

        gpuTallyType_t maxdiff = 0.0;
        int numBenchmarkZeroNonMatching = 0;
        int numGPUZeroNonMatching = 0;
        int numZeroZero = 0;
        int numNonMatching = 0;
        for( size_t i=0; i<benchmarkTally.size(); ++i ) {
            if( controller.contribution(i) > 0.0 &&  benchmarkTally[i] > 0.0 ){
                gpuTallyType_t relDiff = 100.0*( benchmarkTally[i] - controller.contribution(i) ) / benchmarkTally[i];
                if (std::abs(relDiff) > 0.34){
                  numNonMatching++;
                }
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

        std::cout << "Debug:  maxPercentDiff = " << maxdiff << "\n";
        std::cout << "Debug:  tally size = " << benchmarkTally.size() << "\n";
        std::cout << "Debug:  numNonMatching = " << numNonMatching << "\n";
        std::cout << "Debug:  numBenchmarkZeroNonMatching = " << numBenchmarkZeroNonMatching << "\n";
        std::cout << "Debug:        numGPUZeroNonMatching = " << numGPUZeroNonMatching << "\n";
        std::cout << "Debug:                num both zero = " << numZeroZero << "\n";

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

    TEST_FIXTURE(ControllerSetup, RayTraceWithMovingMaterials){
        // compare with mcatk calling MonteRay -- validated against MCATK itself

        setupWithMovingMaterials();

        // 256
        //  64
        // 192
        unsigned nThreadsPerBlock = 1;
        unsigned nThreads = 256;
        /* unsigned capacity = std::min( 64000000U, 40000*8U*8*10U ); */
        unsigned capacity = 16698849;

        auto launchBounds = setLaunchBounds( nThreads, nThreadsPerBlock, capacity);

        std::cout << "Running PWR_Assembly from collision file with nBlocks = " << launchBounds.first <<
                " nThreads = " << launchBounds.second << " collision buffer capacity = " << capacity << "\n";

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

        size_t numCollisions = controller.readCollisionsFromFile( "MonteRayTestFiles/PWR_assembly_collisions.bin" );
        CHECK_EQUAL(16698848, numCollisions);

        controller.sync();

        for( unsigned i=0; i<benchmarkTally.size(); ++i ) {
            if( controller.contribution(i) > 0.0 &&  benchmarkTally[i] > 0.0 ){
                gpuTallyType_t relDiff = 100.0*( benchmarkTally[i] - controller.contribution(i) ) / benchmarkTally[i];
                if (relDiff > 0.44){
                  CHECK_CLOSE( 0.0, relDiff, 0.44 );
                  break;
                }
            } else {
                /* CHECK_CLOSE( 0.0, controller.contribution(i), 1e-4); */
                /* CHECK_CLOSE( 0.0, benchmarkTally[i], 1e-4); */
            }
        }

        gpuTallyType_t maxdiff = 0.0;
        int numBenchmarkZeroNonMatching = 0;
        int numGPUZeroNonMatching = 0;
        int numZeroZero = 0;
        int numNonMatching = 0;
        for( unsigned i=0; i<benchmarkTally.size(); ++i ) {
            if( controller.contribution(i) > 0.0 &&  benchmarkTally[i] > 0.0 ){
                gpuTallyType_t relDiff = 100.0*( benchmarkTally[i] - controller.contribution(i) ) / benchmarkTally[i];
                if (std::abs(relDiff) > 0.34){
                  numNonMatching++;
                }
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

        std::cout << "Debug:  maxPercentDiff = " << maxdiff << "\n";
        std::cout << "Debug:  tally size = " << benchmarkTally.size() << "\n";
        std::cout << "Debug:  numNonMatching = " << numNonMatching << "\n";
        std::cout << "Debug:  numBenchmarkZeroNonMatching = " << numBenchmarkZeroNonMatching << "\n";
        std::cout << "Debug:        numGPUZeroNonMatching = " << numGPUZeroNonMatching << "\n";
        std::cout << "Debug:                num both zero = " << numZeroZero << "\n";

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
