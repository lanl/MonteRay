#include <UnitTest++.h>

#include <iostream>
#include <functional>

#include "fi_genericGPU_test_helper.hh"

#include "ReadAndWriteFiles.hh"
#include "GPUUtilityFunctions.hh"
#include "BasicTally.hh"
#include "ExpectedPathLength.t.hh"
#include "Material.hh"
#include "MaterialProperties.hh"
#include "MonteRay_ReadLnk3dnt.hh"
#include "MonteRay_timer.hh"
#include "HashLookup.hh"

#if( true )
SUITE( RayListInterface_fi_tester ) {

    TEST( setup ) {
        std::cout << "Debug: starting - RayListInterface_fi_tester\n";
        //CHECK(false);
        //gpuCheck();
    }

#if true
    // these tests are commented out as now only 100000 rays per batch can be processed and they are
    // duplicated in RayTrace fi tests
    TEST(get_total_xs_from_gpu ) {
        RayListInterface<1>* points = new RayListInterface<1>(2);
        points->readToMemory( "MonteRayTestFiles/collisionsGodivaRCart100x100x100InWater_2568016Rays.bin"  );
        FIGenericGPUTestHelper<1> helper(points->size());
        points->copyToGPU();

        CrossSectionList::Builder xsListBuilder;
        auto xsBuilder = CrossSectionBuilder();
        readInPlaceFromFile("MonteRayTestFiles/92235-65c_MonteRayCrossSection.bin", xsBuilder);
        xsBuilder.setZAID(92235);
        auto xsPtr = std::make_unique<CrossSection>(xsBuilder.build());

        gpuFloatType_t energy = points->getEnergy(0);
        gpuFloatType_t expected = xsPtr->getTotalXS(energy);
        CHECK_CLOSE(  7.85419f, expected, 1e-5);

        helper.setupTimers();
        helper.launchTallyCrossSection(1, 1024, points, xsPtr.get());
        helper.stopTimers();

        CHECK_CLOSE( expected, helper.getTally(0), 1e-7 );

        delete points;
    }

    TEST(load_godiva_metal_from_file_small_file ) {
        RayListInterface<1> points(2);
        points.readToMemory( "MonteRayTestFiles/collisionsGodivaRCart100x100x100InWater_2568016Rays.bin"  );
        FIGenericGPUTestHelper<1> helper(points.size());
        points.copyToGPU();

        CrossSectionList::Builder xsListBuilder;
        auto xsBuilder = CrossSectionBuilder();
        readInPlaceFromFile( "MonteRayTestFiles/92234-69c_MonteRayCrossSection.bin", xsBuilder );
        xsBuilder.setZAID(92234);
        xsListBuilder.add(xsBuilder.build());
        readInPlaceFromFile( "MonteRayTestFiles/92235-65c_MonteRayCrossSection.bin", xsBuilder );
        xsBuilder.setZAID(92235);
        xsListBuilder.add(xsBuilder.build());
        readInPlaceFromFile( "MonteRayTestFiles/92238-69c_MonteRayCrossSection.bin", xsBuilder );
        xsBuilder.setZAID(92238);
        xsListBuilder.add(xsBuilder.build());
        readInPlaceFromFile( "MonteRayTestFiles/1001-66c_MonteRayCrossSection.bin", xsBuilder );
        xsBuilder.setZAID(1001);
        xsListBuilder.add(xsBuilder.build());
        readInPlaceFromFile( "MonteRayTestFiles/8016-70c_MonteRayCrossSection.bin", xsBuilder );
        xsBuilder.setZAID(8016);
        xsListBuilder.add(xsBuilder.build());

        auto pXsList = std::make_unique<CrossSectionList>(xsListBuilder.build());

        MaterialList::Builder matListBuilder{};
        auto mb = Material::make_builder(*pXsList);
        mb.addIsotope(0.01, 92234);
        mb.addIsotope(0.98, 92235);
        mb.addIsotope(0.01, 92238);
        matListBuilder.addMaterial(0, mb.build() );

        mb.addIsotope(0.667, 1001);
        mb.addIsotope(0.333, 8016);
        matListBuilder.addMaterial(1, mb.build() );
        auto pMatList = std::make_unique<MaterialList>(matListBuilder.build());

        gpuFloatType_t energy = points.getEnergy(0);
        gpuFloatType_t expected = pMatList->material(0).getTotalXS( energy, 18.0 );
        CHECK_CLOSE(  0.36215, expected, 1e-5);

        helper.setupTimers();
        helper.launchTallyCrossSection(1, 1024, &points, pMatList.get(), 0, 18.0);
        helper.stopTimers();

        CHECK_CLOSE( expected, helper.getTally(0), 1e-7 );
    }

    TEST( load_godivaR_materials_godivaR_geom_and_collisions_tally_collision ) {
        RayListInterface<1> points(2);
        points.readToMemory( "MonteRayTestFiles/collisionsGodivaRCart100x100x100InWater_2568016Rays.bin"  );
        FIGenericGPUTestHelper<1> helper(points.size());
        points.copyToGPU();

        MonteRay_ReadLnk3dnt readerObject( "lnk3dnt/godivaR_lnk3dnt_cartesian_100x100x100.lnk3dnt" );
        readerObject.ReadMatData();

        MaterialProperties::Builder mpb;
        mpb.disableMemoryReduction();
        mpb.setMaterialDescription( readerObject );

        CrossSectionList::Builder xsListBuilder;
        auto xsBuilder = CrossSectionBuilder();
        readInPlaceFromFile( "MonteRayTestFiles/92234-69c_MonteRayCrossSection.bin", xsBuilder );
        xsBuilder.setZAID(92234);
        xsListBuilder.add(xsBuilder.build());
        readInPlaceFromFile( "MonteRayTestFiles/92235-65c_MonteRayCrossSection.bin", xsBuilder );
        xsBuilder.setZAID(92235);
        xsListBuilder.add(xsBuilder.build());
        readInPlaceFromFile( "MonteRayTestFiles/92238-69c_MonteRayCrossSection.bin", xsBuilder );
        xsBuilder.setZAID(92238);
        xsListBuilder.add(xsBuilder.build());
        readInPlaceFromFile( "MonteRayTestFiles/1001-66c_MonteRayCrossSection.bin", xsBuilder );
        xsBuilder.setZAID(1001);
        xsListBuilder.add(xsBuilder.build());
        readInPlaceFromFile( "MonteRayTestFiles/8016-70c_MonteRayCrossSection.bin", xsBuilder );
        xsBuilder.setZAID(8016);
        xsListBuilder.add(xsBuilder.build());

        auto pXsList = std::make_unique<CrossSectionList>(xsListBuilder.build());

        MaterialList::Builder matListBuilder{};
        auto mb = Material::make_builder(*pXsList);
        mb.addIsotope(0.01, 92234);
        mb.addIsotope(0.98, 92235);
        mb.addIsotope(0.01, 92238);
        matListBuilder.addMaterial(2, mb.build() );

        mb.addIsotope(0.667, 1001);
        mb.addIsotope(0.333, 8016);
        matListBuilder.addMaterial(3, mb.build() );
        auto pMatList = std::make_unique<MaterialList>(matListBuilder.build());

        mpb.renumberMaterialIDs( *pMatList );
        auto mp = std::make_unique<MaterialProperties>(mpb.build());

        gpuFloatType_t energy = points.getEnergy(0);
        unsigned cell = points.getIndex(0);
        gpuFloatType_t expected1 = helper.getTotalXSByMatProp(mp.get(), pMatList.get(), cell, energy );
        CHECK_CLOSE( 4.44875, energy, 1e-5);
        CHECK_EQUAL( 485557, cell);
        CHECK_CLOSE( 0.353442, expected1, 1e-6);

        helper.setupTimers();
        helper.launchTallyCrossSectionAtCollision(1, 1024, &points, pMatList.get(), mp.get() );
        helper.stopTimers();

        CHECK_CLOSE( expected1, helper.getTally(0), 1e-7 );
    }

    TEST( sum_crossSection_by_startingCell )  {
        MonteRay_ReadLnk3dnt readerObject( "lnk3dnt/godivaR_lnk3dnt_cartesian_100x100x100.lnk3dnt" );
        readerObject.ReadMatData();

        MaterialProperties::Builder mpb;
        mpb.disableMemoryReduction();
        mpb.setMaterialDescription( readerObject );


        RayListInterface<1> points(2);
        points.readToMemory( "MonteRayTestFiles/collisionsGodivaRCart100x100x100InWater_2568016Rays.bin"  );
        points.copyToGPU();

        CrossSectionList::Builder xsListBuilder;
        auto xsBuilder = CrossSectionBuilder();
        readInPlaceFromFile( "MonteRayTestFiles/92234-69c_MonteRayCrossSection.bin", xsBuilder );
        xsBuilder.setZAID(92234);
        xsListBuilder.add(xsBuilder.build());
        readInPlaceFromFile( "MonteRayTestFiles/92235-65c_MonteRayCrossSection.bin", xsBuilder );
        xsBuilder.setZAID(92235);
        xsListBuilder.add(xsBuilder.build());
        readInPlaceFromFile( "MonteRayTestFiles/92238-69c_MonteRayCrossSection.bin", xsBuilder );
        xsBuilder.setZAID(92238);
        xsListBuilder.add(xsBuilder.build());
        readInPlaceFromFile( "MonteRayTestFiles/1001-66c_MonteRayCrossSection.bin", xsBuilder );
        xsBuilder.setZAID(1001);
        xsListBuilder.add(xsBuilder.build());
        readInPlaceFromFile( "MonteRayTestFiles/8016-70c_MonteRayCrossSection.bin", xsBuilder );
        xsBuilder.setZAID(8016);
        xsListBuilder.add(xsBuilder.build());

        auto pXsList = std::make_unique<CrossSectionList>(xsListBuilder.build());

        MaterialList::Builder matListBuilder{};
        auto mb = Material::make_builder(*pXsList);
        mb.addIsotope(0.01, 92234);
        mb.addIsotope(0.98, 92235);
        mb.addIsotope(0.01, 92238);
        matListBuilder.addMaterial(2, mb.build() );

        mb.addIsotope(0.667, 1001);
        mb.addIsotope(0.333, 8016);
        matListBuilder.addMaterial(3, mb.build() );
        auto pMatList = std::make_unique<MaterialList>(matListBuilder.build());


        mpb.renumberMaterialIDs( *pMatList );
        auto mp = std::make_unique<MaterialProperties>(mpb.build());

        FIGenericGPUTestHelper<1> helper( mp->numCells() );

        gpuFloatType_t energy = points.getEnergy(0);
        unsigned cell = points.getIndex(0);
        gpuFloatType_t expected = helper.getTotalXSByMatProp(mp.get(), pMatList.get(), cell, energy );
        CHECK_CLOSE( 4.44875, energy, 1e-5);
        CHECK_EQUAL( 485557, cell);
        CHECK_CLOSE( 0.353442, expected, 1e-6);

        expected=0.0;
        for( unsigned i=0; i<points.size(); ++i){
            if( points.getIndex(i) == cell ) {
                energy = points.getEnergy(i);
                expected += helper.getTotalXSByMatProp(mp.get(), pMatList.get(), cell, energy );
            }
        }
        CHECK_CLOSE( 16.6541, expected, 1e-3);

        helper.setupTimers();
        helper.launchSumCrossSectionAtCollisionLocation(1, 1024, &points, pMatList.get(), mp.get() );
        helper.stopTimers();

        CHECK_CLOSE( expected, helper.getTally(cell), 1e-3 );
    }

    TEST( rayTraceTally_GodivaR )
    {
        auto pGrid = std::make_unique<MonteRay_SpatialGrid>(TransportMeshType::Cartesian, 
          std::array<MonteRay_GridBins, 3>{
          MonteRay_GridBins{-33.5, 33.5, 100},
          MonteRay_GridBins{-33.5, 33.5, 100},
          MonteRay_GridBins{-33.5, 33.5, 100} }
          );

        FIGenericGPUTestHelper<1> helper( pGrid->getNumCells() );

        MonteRay_ReadLnk3dnt readerObject( "lnk3dnt/godivaR_lnk3dnt_cartesian_100x100x100.lnk3dnt" );
        readerObject.ReadMatData();

        MaterialProperties::Builder mpb;
        mpb.disableMemoryReduction();
        mpb.setMaterialDescription( readerObject );

        CrossSectionList::Builder xsListBuilder;
        auto xsBuilder = CrossSectionBuilder();
        readInPlaceFromFile( "MonteRayTestFiles/92234-69c_MonteRayCrossSection.bin", xsBuilder );
        xsBuilder.setZAID(92234);
        xsListBuilder.add(xsBuilder.build());
        readInPlaceFromFile( "MonteRayTestFiles/92235-65c_MonteRayCrossSection.bin", xsBuilder );
        xsBuilder.setZAID(92235);
        xsListBuilder.add(xsBuilder.build());
        readInPlaceFromFile( "MonteRayTestFiles/92238-69c_MonteRayCrossSection.bin", xsBuilder );
        xsBuilder.setZAID(92238);
        xsListBuilder.add(xsBuilder.build());
        readInPlaceFromFile( "MonteRayTestFiles/1001-66c_MonteRayCrossSection.bin", xsBuilder );
        xsBuilder.setZAID(1001);
        xsListBuilder.add(xsBuilder.build());
        readInPlaceFromFile( "MonteRayTestFiles/8016-70c_MonteRayCrossSection.bin", xsBuilder );
        xsBuilder.setZAID(8016);
        xsListBuilder.add(xsBuilder.build());

        auto pXsList = std::make_unique<CrossSectionList>(xsListBuilder.build());

        MaterialList::Builder matListBuilder{};
        auto mb = Material::make_builder(*pXsList);
        mb.addIsotope(0.01, 92234);
        mb.addIsotope(0.98, 92235);
        mb.addIsotope(0.01, 92238);
        matListBuilder.addMaterial(2, mb.build() );

        mb.addIsotope(0.667, 1001);
        mb.addIsotope(0.333, 8016);
        matListBuilder.addMaterial(3, mb.build() );
        auto pMatList = std::make_unique<MaterialList>(matListBuilder.build());

        mpb.renumberMaterialIDs( *pMatList );
        auto mp = std::make_unique<MaterialProperties>(mpb.build());

        RayListInterface<1> points(2);
        points.readToMemory( "MonteRayTestFiles/collisionsGodivaRCart100x100x100InWater_2568016Rays.bin"  );

        points.copyToGPU();
        CHECK_EQUAL(2568016, points.size());

        gpuFloatType_t energy = points.getEnergy(0);
        unsigned cell = points.getIndex(0);
        gpuFloatType_t expected = helper.getTotalXSByMatProp(mp.get(), pMatList.get(), cell, energy );
        CHECK_CLOSE( 4.44875, energy, 1e-6);
        CHECK_EQUAL( 485557, cell);
        CHECK_CLOSE( 0.353442, expected, 1e-6);

        helper.setupTimers();
        helper.launchRayTraceTally(1, 256, &points, pMatList.get(), mp.get(), pGrid.get() );
        helper.stopTimers();

        //    	CHECK_CLOSE( 0.0803215, helper.getTally(0), 1e-5 );
        //    	CHECK_CLOSE( 0.186005, helper.getTally(50+100*100), 1e-4 );

        CHECK_CLOSE( 0.0201584, helper.getTally(24), 1e-5 );
        CHECK_CLOSE( 0.0504394, helper.getTally(500182), 1e-4 );
    }
#endif

}

SUITE( Collision_fi_looping_tester ) {

    TEST( rayTraceTally_GodivaR_wGlobalLauncher )
    {
        std::cout << "Debug: ********************************************* \n";
        std::cout << "Debug: Starting rayTrace tester with Global Launcher \n";
        FIGenericGPUTestHelper<1> helper(  1 );

        auto pGrid = std::make_unique<MonteRay_SpatialGrid>(TransportMeshType::Cartesian, 
          std::array<MonteRay_GridBins, 3>{
          MonteRay_GridBins{-33.5, 33.5, 100},
          MonteRay_GridBins{-33.5, 33.5, 100},
          MonteRay_GridBins{-33.5, 33.5, 100} }
        );

        auto pTally = std::make_unique<BasicTally>( pGrid->getNumCells() );

        MonteRay_ReadLnk3dnt readerObject( "lnk3dnt/godivaR_lnk3dnt_cartesian_100x100x100.lnk3dnt" );
        readerObject.ReadMatData();

        MaterialProperties::Builder mpb;
        mpb.disableMemoryReduction();
        mpb.setMaterialDescription( readerObject );

        CrossSectionList::Builder xsListBuilder;
        auto xsBuilder = CrossSectionBuilder();
        readInPlaceFromFile( "MonteRayTestFiles/92234-69c_MonteRayCrossSection.bin", xsBuilder );
        xsBuilder.setZAID(92234);
        xsListBuilder.add(xsBuilder.build());
        readInPlaceFromFile( "MonteRayTestFiles/92235-65c_MonteRayCrossSection.bin", xsBuilder );
        xsBuilder.setZAID(92235);
        xsListBuilder.add(xsBuilder.build());
        readInPlaceFromFile( "MonteRayTestFiles/92238-69c_MonteRayCrossSection.bin", xsBuilder );
        xsBuilder.setZAID(92238);
        xsListBuilder.add(xsBuilder.build());
        readInPlaceFromFile( "MonteRayTestFiles/1001-66c_MonteRayCrossSection.bin", xsBuilder );
        xsBuilder.setZAID(1001);
        xsListBuilder.add(xsBuilder.build());
        readInPlaceFromFile( "MonteRayTestFiles/8016-70c_MonteRayCrossSection.bin", xsBuilder );
        xsBuilder.setZAID(8016);
        xsListBuilder.add(xsBuilder.build());

        auto pXsList = std::make_unique<CrossSectionList>(xsListBuilder.build());

        MaterialList::Builder matListBuilder{};
        auto mb = Material::make_builder(*pXsList);
        mb.addIsotope(0.01, 92234);
        mb.addIsotope(0.98, 92235);
        mb.addIsotope(0.01, 92238);
        matListBuilder.addMaterial(2, mb.build() );

        mb.addIsotope(0.667, 1001);
        mb.addIsotope(0.333, 8016);
        matListBuilder.addMaterial(3, mb.build() );
        auto pMatList = std::make_unique<MaterialList>(matListBuilder.build());

        mpb.renumberMaterialIDs( *pMatList );
        auto mp = std::make_unique<MaterialProperties>(mpb.build());

        RayListInterface<1> bank1(100000);
        bool end = false;
        unsigned offset = 0;
        std::cout << "Debug: Reading Bank1 \n";
        end = bank1.readToBank( "MonteRayTestFiles/collisionsGodivaRCart100x100x100InWater_2568016Rays.bin", offset );

        gpuFloatType_t energy = bank1.getEnergy(0);
        unsigned cell = bank1.getIndex(0);
        gpuFloatType_t expected = helper.getTotalXSByMatProp(mp.get(), pMatList.get(), cell, energy );
        CHECK_CLOSE( 4.44875, energy, 1e-5);
        CHECK_EQUAL( 485557, cell);
        CHECK_CLOSE( 0.353442, expected, 1e-6);

        offset += bank1.size();

        RayListInterface<1> bank2(100000);
        bool last = false;

        auto cpuWork1 = [&] (void) -> void {
            if( !end ) {
                std::cout << "Debug: Reading Bank2 \n";
                end = bank2.readToBank( "MonteRayTestFiles/collisionsGodivaRCart100x100x100InWater_2568016Rays.bin", offset );
                offset += bank2.size();
            }
        };
        auto cpuWork2 = [&] (void) -> void {
            if( !end ) {
                std::cout << "Debug: Reading Bank1 \n";
                end = bank1.readToBank( "MonteRayTestFiles/collisionsGodivaRCart100x100x100InWater_2568016Rays.bin", offset );
                offset += bank1.size();
            }
        };

        helper.setupTimers();

        while(true){
            bank1.copyToGPU();
            if( end ) { last = true; }
            MonteRay::tripleTime time = launchRayTraceTally(
                    cpuWork1,
                    1,
                    256,
                    pGrid.get(),
                    &bank1,
                    pMatList.get(),
                    mp.get(),
                    pTally.get());

#ifdef __CUDACC__
            std::cout << "Debug: Time in GPU raytrace kernel=" << time.gpuTime << " secs.\n";
#else
            std::cout << "Debug: Time in CPU raytrace kernel=" << time.gpuTime << " secs.\n";
#endif
            std::cout << "Debug: Time in CPU work =" << time.cpuTime << " secs.\n";
            std::cout << "Debug: Time total time =" << time.totalTime << " secs.\n\n";
            if( last ) { break; }

            bank2.copyToGPU();
            if( end ) { last = true; }
            time = launchRayTraceTally(
                    cpuWork2,
                    1,
                    256,
                    pGrid.get(),
                    &bank2,
                    pMatList.get(),
                    mp.get(),
                    pTally.get());

#ifdef __CUDACC__
            std::cout << "Debug: Time in GPU raytrace kernel=" << time.gpuTime << " secs.\n";
#else
            std::cout << "Debug: Time in CPU raytrace kernel=" << time.gpuTime << " secs.\n";
#endif
            std::cout << "Debug: Time in CPU work =" << time.cpuTime << " secs.\n";
            std::cout << "Debug: Time total time =" << time.totalTime << " secs.\n\n";
            if( last ) { break; }

        };

        helper.stopTimers();

        CHECK_CLOSE( 0.0201584, pTally->getTally(24), 1e-5 );
        CHECK_CLOSE( 0.0504394, pTally->getTally(500182), 1e-4 );
    }

}
#endif
