#include <UnitTest++.h>

#include <iostream>
#include <functional>

#include "fi_genericGPU_test_helper.hh"

#include "GPUUtilityFunctions.hh"
#include "gpuTally.hh"
#include "ExpectedPathLength.hh"
#include "MonteRayMaterial.hh"
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

        MonteRayCrossSectionHost* xs = new MonteRayCrossSectionHost(1);
        xs->read( "MonteRayTestFiles/92235-65c_MonteRayCrossSection.bin");
        xs->copyToGPU();

        gpuFloatType_t energy = points->getEnergy(0);
        gpuFloatType_t expected = getTotalXS(xs->getXSPtr(), energy);
        CHECK_CLOSE(  7.85419f, expected, 1e-5);

        helper.setupTimers();
        helper.launchTallyCrossSection(1, 1024, points, xs);
        helper.stopTimers();

        CHECK_CLOSE( expected, helper.getTally(0), 1e-7 );

        delete xs;
        delete points;
    }

    TEST(load_godiva_metal_from_file_small_file ) {
        RayListInterface<1> points(2);
        points.readToMemory( "MonteRayTestFiles/collisionsGodivaRCart100x100x100InWater_2568016Rays.bin"  );
        FIGenericGPUTestHelper<1> helper(points.size());
        points.copyToGPU();

        MonteRayCrossSectionHost u234s(1);
        MonteRayCrossSectionHost u235s(1);
        MonteRayCrossSectionHost u238s(1);
        MonteRayCrossSectionHost h1s(1);
        MonteRayCrossSectionHost o16s(1);

        u234s.read( "MonteRayTestFiles/92234-69c_MonteRayCrossSection.bin" );
        u235s.read( "MonteRayTestFiles/92235-65c_MonteRayCrossSection.bin" );
        u238s.read( "MonteRayTestFiles/92238-69c_MonteRayCrossSection.bin" );
        h1s.read( "MonteRayTestFiles/1001-66c_MonteRayCrossSection.bin" );
        o16s.read( "MonteRayTestFiles/8016-70c_MonteRayCrossSection.bin" );

        MonteRayMaterialHost metal(3);
        metal.add(0, u234s, 0.01);
        metal.add(1, u235s, 0.98);
        metal.add(2, u238s, 0.01);
        metal.copyToGPU();

        MonteRayMaterialHost water(2);
        water.add(0, h1s, 0.667 );
        water.add(1, o16s, 0.333 );
        water.copyToGPU();

        MonteRayMaterialListHost matList(2,5);
        matList.add( 0, metal, 0 );
        matList.add( 1, water, 1 );
        matList.copyToGPU();

        u234s.copyToGPU();
        u235s.copyToGPU();
        u238s.copyToGPU();
        h1s.copyToGPU();
        o16s.copyToGPU();

        gpuFloatType_t energy = points.getEnergy(0);
        unsigned HashBin = getHashBin( matList.getHashPtr()->getPtr(), energy );
        gpuFloatType_t expected = getTotalXS(matList.getPtr(), 0, matList.getHashPtr()->getPtr(), HashBin, energy, 18.0 );
        CHECK_CLOSE(  0.36215, expected, 1e-5);

        helper.setupTimers();
        helper.launchTallyCrossSection(1, 1024, &points, &matList, 0, 18.0);
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

        MonteRayCrossSectionHost u234s(1);
        MonteRayCrossSectionHost u235s(1);
        MonteRayCrossSectionHost u238s(1);
        MonteRayCrossSectionHost h1s(1);
        MonteRayCrossSectionHost o16s(1);

        u234s.read( "MonteRayTestFiles/92234-69c_MonteRayCrossSection.bin" );
        u235s.read( "MonteRayTestFiles/92235-65c_MonteRayCrossSection.bin" );
        u238s.read( "MonteRayTestFiles/92238-69c_MonteRayCrossSection.bin" );
        h1s.read( "MonteRayTestFiles/1001-66c_MonteRayCrossSection.bin" );
        o16s.read( "MonteRayTestFiles/8016-70c_MonteRayCrossSection.bin" );

        MonteRayMaterialHost metal(3);
        metal.add(0, u234s, 0.01);
        metal.add(1, u235s, 0.98);
        metal.add(2, u238s, 0.01);
        metal.copyToGPU();

        MonteRayMaterialHost water(2);
        water.add(0, h1s, 0.667 );
        water.add(1, o16s, 0.333 );
        water.copyToGPU();

        MonteRayMaterialListHost matList(2,5);
        matList.add( 0, metal, 2 );
        matList.add( 1, water, 3 );
        matList.copyToGPU();

        mpb.renumberMaterialIDs( matList );
        auto mp = std::make_unique<MaterialProperties>(mpb.build());


        u234s.copyToGPU();
        u235s.copyToGPU();
        u238s.copyToGPU();
        h1s.copyToGPU();
        o16s.copyToGPU();

        gpuFloatType_t energy = points.getEnergy(0);
        unsigned cell = points.getIndex(0);
        unsigned HashBin = getHashBin( matList.getHashPtr()->getPtr(), energy );
        gpuFloatType_t expected1 = helper.getTotalXSByMatProp(mp.get(), matList.getPtr(), matList.getHashPtr()->getPtr(), HashBin, cell, energy );
        gpuFloatType_t expected2 = helper.getTotalXSByMatProp(mp.get(), matList.getPtr(), cell, energy );
        CHECK_CLOSE( 4.44875, energy, 1e-5);
        CHECK_EQUAL( 485557, cell);
        CHECK_CLOSE( 0.353442, expected1, 1e-6);
        CHECK_CLOSE( expected2, expected1, 1e-6);

        helper.setupTimers();
        helper.launchTallyCrossSectionAtCollision(1, 1024, &points, &matList, mp.get() );
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

        MonteRayCrossSectionHost u234s(1);
        MonteRayCrossSectionHost u235s(1);
        MonteRayCrossSectionHost u238s(1);
        MonteRayCrossSectionHost h1s(1);
        MonteRayCrossSectionHost o16s(1);

        u234s.read( "MonteRayTestFiles/92234-69c_MonteRayCrossSection.bin" );
        u235s.read( "MonteRayTestFiles/92235-65c_MonteRayCrossSection.bin" );
        u238s.read( "MonteRayTestFiles/92238-69c_MonteRayCrossSection.bin" );
        h1s.read( "MonteRayTestFiles/1001-66c_MonteRayCrossSection.bin" );
        o16s.read( "MonteRayTestFiles/8016-70c_MonteRayCrossSection.bin" );

        MonteRayMaterialHost metal(3);
        metal.add(0, u234s, 0.01);
        metal.add(1, u235s, 0.98);
        metal.add(2, u238s, 0.01);
        metal.copyToGPU();

        MonteRayMaterialHost water(2);
        water.add(0, h1s, 0.667 );
        water.add(1, o16s, 0.333 );
        water.copyToGPU();

        MonteRayMaterialListHost matList(2,5);
        matList.add( 0, metal, 2 );
        matList.add( 1, water, 3 );
        matList.copyToGPU();

        mpb.renumberMaterialIDs( matList );
        auto mp = std::make_unique<MaterialProperties>(mpb.build());

        FIGenericGPUTestHelper<1> helper( mp->numCells() );

        u234s.copyToGPU();
        u235s.copyToGPU();
        u238s.copyToGPU();
        h1s.copyToGPU();
        o16s.copyToGPU();

        gpuFloatType_t energy = points.getEnergy(0);
        unsigned cell = points.getIndex(0);
        unsigned HashBin = getHashBin( matList.getHashPtr()->getPtr(), energy );
        gpuFloatType_t expected = helper.getTotalXSByMatProp(mp.get(), matList.getPtr(), matList.getHashPtr()->getPtr(), HashBin, cell, energy );
        CHECK_CLOSE( 4.44875, energy, 1e-5);
        CHECK_EQUAL( 485557, cell);
        CHECK_CLOSE( 0.353442, expected, 1e-6);

        expected=0.0;
        for( unsigned i=0; i<points.size(); ++i){
            if( points.getIndex(i) == cell ) {
                energy = points.getEnergy(i);
                HashBin = getHashBin( matList.getHashPtr()->getPtr(), energy );
                expected += helper.getTotalXSByMatProp(mp.get(), matList.getPtr(), matList.getHashPtr()->getPtr(), HashBin, cell, energy );
            }
        }
        CHECK_CLOSE( 16.6541, expected, 1e-3);

        helper.setupTimers();
        helper.launchSumCrossSectionAtCollisionLocation(1, 1024, &points, &matList, mp.get() );
        helper.stopTimers();

        CHECK_CLOSE( expected, helper.getTally(cell), 1e-3 );
    }

    TEST( rayTraceTally_GodivaR )
    {
        //cudaReset();

        GridBins* grid_host = new GridBins;
        grid_host->setVertices( 0, -33.5, 33.5, 100);
        grid_host->setVertices( 1, -33.5, 33.5, 100);
        grid_host->setVertices( 2, -33.5, 33.5, 100);
        grid_host->finalize();

        FIGenericGPUTestHelper<1> helper( 0 );
        helper.copyGridtoGPU(grid_host);

        MonteRay_ReadLnk3dnt readerObject( "lnk3dnt/godivaR_lnk3dnt_cartesian_100x100x100.lnk3dnt" );
        readerObject.ReadMatData();

        MaterialProperties::Builder mpb;
        mpb.disableMemoryReduction();
        mpb.setMaterialDescription( readerObject );

        MonteRayCrossSectionHost u234s(1);
        MonteRayCrossSectionHost u235s(1);
        MonteRayCrossSectionHost u238s(1);
        MonteRayCrossSectionHost h1s(1);
        MonteRayCrossSectionHost o16s(1);

        u234s.read( "MonteRayTestFiles/92234-69c_MonteRayCrossSection.bin" );
        u235s.read( "MonteRayTestFiles/92235-65c_MonteRayCrossSection.bin" );
        u238s.read( "MonteRayTestFiles/92238-69c_MonteRayCrossSection.bin" );
        h1s.read( "MonteRayTestFiles/1001-66c_MonteRayCrossSection.bin" );
        o16s.read( "MonteRayTestFiles/8016-70c_MonteRayCrossSection.bin" );

        MonteRayMaterialHost metal(3);
        metal.add(0, u234s, 0.01);
        metal.add(1, u235s, 0.98);
        metal.add(2, u238s, 0.01);
        metal.copyToGPU();

        MonteRayMaterialHost water(2);
        water.add(0, h1s, 0.667 );
        water.add(1, o16s, 0.333 );
        water.copyToGPU();

        MonteRayMaterialListHost matList(2,5);
        matList.add( 0, metal, 2 );
        matList.add( 1, water, 3 );
        matList.copyToGPU();

        mpb.renumberMaterialIDs( matList );
        auto mp = std::make_unique<MaterialProperties>(mpb.build());

        u234s.copyToGPU();
        u235s.copyToGPU();
        u238s.copyToGPU();
        h1s.copyToGPU();
        o16s.copyToGPU();

        RayListInterface<1> points(2);
        points.readToMemory( "MonteRayTestFiles/collisionsGodivaRCart100x100x100InWater_2568016Rays.bin"  );

        points.copyToGPU();
        CHECK_EQUAL(2568016, points.size());

        gpuFloatType_t energy = points.getEnergy(0);
        unsigned cell = points.getIndex(0);
        unsigned HashBin = getHashBin( matList.getHashPtr()->getPtr(), energy );
        gpuFloatType_t expected = helper.getTotalXSByMatProp(mp.get(), matList.getPtr(), matList.getHashPtr()->getPtr(), HashBin, cell, energy );
        CHECK_CLOSE( 4.44875, energy, 1e-6);
        CHECK_EQUAL( 485557, cell);
        CHECK_CLOSE( 0.353442, expected, 1e-6);

        helper.setupTimers();
        helper.launchRayTraceTally(1, 256, &points, &matList, mp.get() );
        helper.stopTimers();

        //    	CHECK_CLOSE( 0.0803215, helper.getTally(0), 1e-5 );
        //    	CHECK_CLOSE( 0.186005, helper.getTally(50+100*100), 1e-4 );

        CHECK_CLOSE( 0.0201584, helper.getTally(24), 1e-5 );
        CHECK_CLOSE( 0.0504394, helper.getTally(500182), 1e-4 );
        //    	for( unsigned i=0; i<grid_host->num[0]*grid_host->num[1]*grid_host->num[2]; ++i) {
        //    		if( helper.getTally(i) > 0.0 ) {
        //    			std::cout << "i = " << i << " tally = " << helper.getTally(i) << "\n";
        //    		}
        //    	}
        delete grid_host;
    }
#endif

}

SUITE( Collision_fi_looping_tester ) {

    TEST( setup ) {
        //gpuCheck();
    }

    TEST( rayTraceTally_GodivaR_wGlobalLauncher )
    {
        std::cout << "Debug: ********************************************* \n";
        std::cout << "Debug: Starting rayTrace tester with Global Launcher \n";
        FIGenericGPUTestHelper<1> helper(  1 );

        //cudaReset();
        GridBins grid;
        grid.setVertices( 0, -33.5, 33.5, 100);
        grid.setVertices( 1, -33.5, 33.5, 100);
        grid.setVertices( 2, -33.5, 33.5, 100);

        grid.copyToGPU();

        gpuTallyHost tally( grid.getNumCells() );
        tally.copyToGPU();

        MonteRay_ReadLnk3dnt readerObject( "lnk3dnt/godivaR_lnk3dnt_cartesian_100x100x100.lnk3dnt" );
        readerObject.ReadMatData();

        MaterialProperties::Builder mpb;
        mpb.disableMemoryReduction();
        mpb.setMaterialDescription( readerObject );

        MonteRayCrossSectionHost u234s(1);
        MonteRayCrossSectionHost u235s(1);
        MonteRayCrossSectionHost u238s(1);
        MonteRayCrossSectionHost h1s(1);
        MonteRayCrossSectionHost o16s(1);

        u234s.read( "MonteRayTestFiles/92234-69c_MonteRayCrossSection.bin" );
        u235s.read( "MonteRayTestFiles/92235-65c_MonteRayCrossSection.bin" );
        u238s.read( "MonteRayTestFiles/92238-69c_MonteRayCrossSection.bin" );
        h1s.read( "MonteRayTestFiles/1001-66c_MonteRayCrossSection.bin" );
        o16s.read( "MonteRayTestFiles/8016-70c_MonteRayCrossSection.bin" );

        MonteRayMaterialHost metal(3);
        metal.add(0, u234s, 0.01);
        metal.add(1, u235s, 0.98);
        metal.add(2, u238s, 0.01);
        metal.copyToGPU();

        MonteRayMaterialHost water(2);
        water.add(0, h1s, 0.667 );
        water.add(1, o16s, 0.333 );
        water.copyToGPU();

        MonteRayMaterialListHost matList(2,5);
        matList.add( 0, metal, 2 );
        matList.add( 1, water, 3 );
        matList.copyToGPU();

        mpb.renumberMaterialIDs( matList );
        auto mp = std::make_unique<MaterialProperties>(mpb.build());

        u234s.copyToGPU();
        u235s.copyToGPU();
        u238s.copyToGPU();
        h1s.copyToGPU();
        o16s.copyToGPU();

        RayListInterface<1> bank1(100000);
        bool end = false;
        unsigned offset = 0;
        std::cout << "Debug: Reading Bank1 \n";
        end = bank1.readToBank( "MonteRayTestFiles/collisionsGodivaRCart100x100x100InWater_2568016Rays.bin", offset );

        gpuFloatType_t energy = bank1.getEnergy(0);
        unsigned cell = bank1.getIndex(0);
        unsigned HashBin = getHashBin( matList.getHashPtr()->getPtr(), energy );
        gpuFloatType_t expected = helper.getTotalXSByMatProp(mp.get(), matList.getPtr(), matList.getHashPtr()->getPtr(), HashBin, cell, energy );
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
                    &grid,
                    &bank1,
                    &matList,
                    mp.get(),
                    &tally);

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
                    &grid,
                    &bank2,
                    &matList,
                    mp.get(),
                    &tally);

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

        tally.copyToCPU();

        CHECK_CLOSE( 0.0201584, tally.getTally(24), 1e-5 );
        CHECK_CLOSE( 0.0504394, tally.getTally(500182), 1e-4 );
        //    for( unsigned i=0; i<grid.getNumCells(); ++i) {
        //        if( tally.getTally(i) > 0.0 ) {
        //            std::cout << "i = " << i << " tally = " << tally.getTally(i) << "\n";
        //        }
        //    }

    }

}
#endif
