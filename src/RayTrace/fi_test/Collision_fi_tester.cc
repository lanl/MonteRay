#include <UnitTest++.h>

#include <iostream>

#include "fi_genericGPU_test_helper.hh"

SUITE( Collision_fi_tester ) {

    TEST(get_total_xs_from_gpu ) {
    	cudaReset();
    	CollisionPointsHost* points = new CollisionPointsHost(2);
    	points->readToMemory( "/usr/projects/mcatk/user/jsweezy/link_files/collisionsGodivaCylInWater1.bin"  );
    	FIGenericGPUTestHelper helper(points->size());
    	points->copyToGPU();

    	SimpleCrossSectionHost* xs = new SimpleCrossSectionHost(1);
    	xs->read( "/usr/projects/mcatk/user/jsweezy/link_files/u235_simpleCrossSection.bin");
    	xs->copyToGPU();

    	gpuFloatType_t energy = points->getEnergy(0);
    	gpuFloatType_t expected = getTotalXS(xs->getXSPtr(), energy);
    	CHECK_CLOSE(  6.8940749168395996f, expected, 1e-6);

    	helper.setupTimers();
    	helper.launchTallyCrossSection(1024, 1024, points, xs);
    	helper.stopTimers();

    	CHECK_CLOSE( expected, helper.getTally(0), 1e-7 );

    	delete xs;
    	delete points;
    }

    TEST(load_godiva_metal_from_file_small_file ) {
    	cudaReset();
    	CollisionPointsHost points(2);
    	points.readToMemory( "/usr/projects/mcatk/user/jsweezy/link_files/collisionsGodivaCylInWater1.bin"  );
    	FIGenericGPUTestHelper helper(points.size());
    	points.copyToGPU();

    	SimpleCrossSectionHost u234s(1);
        SimpleCrossSectionHost u235s(1);
        SimpleCrossSectionHost u238s(1);
        SimpleCrossSectionHost h1s(1);
        SimpleCrossSectionHost o16s(1);

        u234s.read( "/usr/projects/mcatk/user/jsweezy/link_files/u234_simpleCrossSection.bin" );
        u235s.read( "/usr/projects/mcatk/user/jsweezy/link_files/u235_simpleCrossSection.bin" );
        u238s.read( "/usr/projects/mcatk/user/jsweezy/link_files/u238_simpleCrossSection.bin" );
        h1s.read( "/usr/projects/mcatk/user/jsweezy/link_files/h1_simpleCrossSection.bin" );
        o16s.read( "/usr/projects/mcatk/user/jsweezy/link_files/o16_simpleCrossSection.bin" );

        u234s.copyToGPU();
        u235s.copyToGPU();
        u238s.copyToGPU();
        h1s.copyToGPU();
        o16s.copyToGPU();

        SimpleMaterialHost metal(3);
        metal.add(0, u234s, 0.01);
        metal.add(1, u235s, 0.98);
        metal.add(2, u238s, 0.01);
        metal.copyToGPU();

        SimpleMaterialHost water(2);
        water.add(0, h1s, 0.667 );
        water.add(0, o16s, 0.333 );
        water.copyToGPU();

        SimpleMaterialListHost matList(2);
        matList.add( 0, metal, 0 );
        matList.add( 1, water, 1 );
        matList.copyToGPU();

    	gpuFloatType_t energy = points.getEnergy(0);
    	gpuFloatType_t expected = getTotalXS(matList.getPtr(), 0, energy, 18.0 );
    	CHECK_CLOSE(  0.318065, expected, 1e-6);

    	helper.setupTimers();
       	helper.launchTallyCrossSection(1024, 1024, &points, &matList, 0, 18.0);
        helper.stopTimers();

        CHECK_CLOSE( expected, helper.getTally(0), 1e-7 );
    }

    TEST( load_godivaR_materials_godivaR_geom_and_collisions_tally_collision ) {
    	cudaReset();

    	CollisionPointsHost points(2);
    	points.readToMemory( "/usr/projects/mcatk/user/jsweezy/link_files/collisionsGodivaCyl100x100x100InWater.bin"  );
        FIGenericGPUTestHelper helper(points.size());
    	points.copyToGPU();

        SimpleMaterialPropertiesHost mp(2);
        mp.read( "/usr/projects/mcatk/user/jsweezy/link_files/godivaR_geometry_100x100x100.bin" );
        mp.copyToGPU();

    	SimpleCrossSectionHost u234s(1);
        SimpleCrossSectionHost u235s(1);
        SimpleCrossSectionHost u238s(1);
        SimpleCrossSectionHost h1s(1);
        SimpleCrossSectionHost o16s(1);

        u234s.read( "/usr/projects/mcatk/user/jsweezy/link_files/u234_simpleCrossSection.bin" );
        u235s.read( "/usr/projects/mcatk/user/jsweezy/link_files/u235_simpleCrossSection.bin" );
        u238s.read( "/usr/projects/mcatk/user/jsweezy/link_files/u238_simpleCrossSection.bin" );
        h1s.read( "/usr/projects/mcatk/user/jsweezy/link_files/h1_simpleCrossSection.bin" );
        o16s.read( "/usr/projects/mcatk/user/jsweezy/link_files/o16_simpleCrossSection.bin" );

        u234s.copyToGPU();
        u235s.copyToGPU();
        u238s.copyToGPU();
        h1s.copyToGPU();
        o16s.copyToGPU();

        SimpleMaterialHost metal(3);
        metal.add(0, u234s, 0.01);
        metal.add(1, u235s, 0.98);
        metal.add(2, u238s, 0.01);
        metal.copyToGPU();

        SimpleMaterialHost water(2);
        water.add(0, h1s, 0.667 );
        water.add(0, o16s, 0.333 );
        water.copyToGPU();

        SimpleMaterialListHost matList(2);
        matList.add( 0, metal, 0 );
        matList.add( 1, water, 1 );
        matList.copyToGPU();

    	gpuFloatType_t energy = points.getEnergy(0);
    	unsigned cell = points.getIndex(0);
    	gpuFloatType_t expected = helper.getTotalXSByMatProp(mp.getPtr(), matList.getPtr(), cell, energy );
    	CHECK_CLOSE( 0.804852, energy, 1e-6);
    	CHECK_EQUAL( 435859, cell);
    	CHECK_CLOSE( 0.102606, expected, 1e-6);

    	helper.setupTimers();
    	helper.launchTallyCrossSectionAtCollision(1024, 1024, &points, &matList, &mp );
    	helper.stopTimers();

    	CHECK_CLOSE( expected, helper.getTally(0), 1e-7 );
    }

    TEST( sum_crossSection_by_startingCell )  {
    	cudaReset();
        SimpleMaterialPropertiesHost mp(2);
        mp.read( "/usr/projects/mcatk/user/jsweezy/link_files/godivaR_geometry_100x100x100.bin" );
        FIGenericGPUTestHelper helper( mp.getNumCells() );
        mp.copyToGPU();

    	CollisionPointsHost points(2);
    	points.readToMemory( "/usr/projects/mcatk/user/jsweezy/link_files/collisionsGodivaCyl100x100x100InWater.bin"  );
    	points.copyToGPU();

    	SimpleCrossSectionHost u234s(1);
        SimpleCrossSectionHost u235s(1);
        SimpleCrossSectionHost u238s(1);
        SimpleCrossSectionHost h1s(1);
        SimpleCrossSectionHost o16s(1);

        u234s.read( "/usr/projects/mcatk/user/jsweezy/link_files/u234_simpleCrossSection.bin" );
        u235s.read( "/usr/projects/mcatk/user/jsweezy/link_files/u235_simpleCrossSection.bin" );
        u238s.read( "/usr/projects/mcatk/user/jsweezy/link_files/u238_simpleCrossSection.bin" );
        h1s.read( "/usr/projects/mcatk/user/jsweezy/link_files/h1_simpleCrossSection.bin" );
        o16s.read( "/usr/projects/mcatk/user/jsweezy/link_files/o16_simpleCrossSection.bin" );

        u234s.copyToGPU();
        u235s.copyToGPU();
        u238s.copyToGPU();
        h1s.copyToGPU();
        o16s.copyToGPU();

        SimpleMaterialHost metal(3);
        metal.add(0, u234s, 0.01);
        metal.add(1, u235s, 0.98);
        metal.add(2, u238s, 0.01);
        metal.copyToGPU();

        SimpleMaterialHost water(2);
        water.add(0, h1s, 0.667 );
        water.add(0, o16s, 0.333 );
        water.copyToGPU();

        SimpleMaterialListHost matList(2);
        matList.add( 0, metal, 0 );
        matList.add( 1, water, 1 );
        matList.copyToGPU();

    	gpuFloatType_t energy = points.getEnergy(0);
    	unsigned cell = points.getIndex(0);
    	gpuFloatType_t expected = helper.getTotalXSByMatProp(mp.getPtr(), matList.getPtr(), cell, energy );
    	CHECK_CLOSE( 0.804852, energy, 1e-6);
    	CHECK_EQUAL( 435859, cell);
    	CHECK_CLOSE( 0.102606, expected, 1e-6);

    	expected=0.0;
    	for( unsigned i=0; i<points.size(); ++i){
    		if( points.getIndex(i) == cell ) {
    			energy = points.getEnergy(i);
    			expected += helper.getTotalXSByMatProp(mp.getPtr(), matList.getPtr(), cell, energy );
    		}
    	}
    	CHECK_CLOSE( 75.2092, expected, 1e-4);

    	helper.setupTimers();
    	helper.launchSumCrossSectionAtCollisionLocation(1024, 1024, &points, &matList, &mp );
    	helper.stopTimers();

    	CHECK_CLOSE( expected, helper.getTally(cell), 1e-4 );
    }

    TEST( rayTraceTally_GodivaR )
    {
    	cudaReset();
    	GridBins* grid_host;
    	grid_host = (GridBins*) malloc( sizeof(GridBins) );
    	ctor( grid_host );
    	setVertices(grid_host, 0, -33.5, 33.5, 100);
    	setVertices(grid_host, 1, -33.5, 33.5, 100);
    	setVertices(grid_host, 2, -33.5, 33.5, 100);
    	finalize(grid_host);
    	FIGenericGPUTestHelper helper( 0 );
    	helper.copyGridtoGPU(grid_host);

        SimpleMaterialPropertiesHost mp(2);
        mp.read( "/usr/projects/mcatk/user/jsweezy/link_files/godivaR_geometry_100x100x100.bin" );
        mp.copyToGPU();

    	SimpleCrossSectionHost u234s(1);
        SimpleCrossSectionHost u235s(1);
        SimpleCrossSectionHost u238s(1);
        SimpleCrossSectionHost h1s(1);
        SimpleCrossSectionHost o16s(1);

        u234s.read( "/usr/projects/mcatk/user/jsweezy/link_files/u234_simpleCrossSection.bin" );
        u235s.read( "/usr/projects/mcatk/user/jsweezy/link_files/u235_simpleCrossSection.bin" );
        u238s.read( "/usr/projects/mcatk/user/jsweezy/link_files/u238_simpleCrossSection.bin" );
        h1s.read( "/usr/projects/mcatk/user/jsweezy/link_files/h1_simpleCrossSection.bin" );
        o16s.read( "/usr/projects/mcatk/user/jsweezy/link_files/o16_simpleCrossSection.bin" );

        u234s.copyToGPU();
        u235s.copyToGPU();
        u238s.copyToGPU();
        h1s.copyToGPU();
        o16s.copyToGPU();

        SimpleMaterialHost metal(3);
        metal.add(0, u234s, 0.01);
        metal.add(1, u235s, 0.98);
        metal.add(2, u238s, 0.01);
        metal.copyToGPU();

        SimpleMaterialHost water(2);
        water.add(0, h1s, 0.667 );
        water.add(0, o16s, 0.333 );
        water.copyToGPU();

        SimpleMaterialListHost matList(2);
        matList.add( 0, metal, 0 );
        matList.add( 1, water, 1 );
        matList.copyToGPU();

    	CollisionPointsHost points(2);
    	points.readToMemory( "/usr/projects/mcatk/user/jsweezy/link_files/collisionsGodivaCyl100x100x100InWater.bin"  );
    	points.copyToGPU();

    	gpuFloatType_t energy = points.getEnergy(0);
    	unsigned cell = points.getIndex(0);
    	gpuFloatType_t expected = helper.getTotalXSByMatProp(mp.getPtr(), matList.getPtr(), cell, energy );
    	CHECK_CLOSE( 0.804852, energy, 1e-6);
    	CHECK_EQUAL( 435859, cell);
    	CHECK_CLOSE( 0.102606, expected, 1e-6);

     	helper.setupTimers();
    	helper.launchRayTraceTally(1024, 1024, &points, &matList, &mp );
    	helper.stopTimers();

    	CHECK_CLOSE( 9.43997, helper.getTally(0), 1e-5 );
    	CHECK_CLOSE( 16.5143, helper.getTally(50+100*100), 1e-4 );
    }

}
