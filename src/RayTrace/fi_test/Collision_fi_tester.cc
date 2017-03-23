#include <UnitTest++.h>

#include <iostream>
#include <functional>

#include "fi_genericGPU_test_helper.hh"

#include "GPUUtilityFunctions.hh"
#include "gpuTally.h"
#include "ExpectedPathLength.h"
#include "cpuTimer.h"

#if( true )
SUITE( Collision_fi_tester ) {

	TEST( setup ) {
		gpuCheck();
	}

    TEST(get_total_xs_from_gpu ) {
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

        SimpleMaterialHost metal(3);
        metal.add(0, u234s, 0.01);
        metal.add(1, u235s, 0.98);
        metal.add(2, u238s, 0.01);
        metal.copyToGPU();

        SimpleMaterialHost water(2);
        water.add(0, h1s, 0.667 );
        water.add(1, o16s, 0.333 );
        water.copyToGPU();

        SimpleMaterialListHost matList(2,5);
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
    	CHECK_CLOSE(  0.318065, expected, 1e-6);

    	helper.setupTimers();
       	helper.launchTallyCrossSection(1024, 1024, &points, &matList, 0, 18.0);
        helper.stopTimers();

        CHECK_CLOSE( expected, helper.getTally(0), 1e-7 );
    }

    TEST( load_godivaR_materials_godivaR_geom_and_collisions_tally_collision ) {
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

        SimpleMaterialHost metal(3);
        metal.add(0, u234s, 0.01);
        metal.add(1, u235s, 0.98);
        metal.add(2, u238s, 0.01);
        metal.copyToGPU();

        SimpleMaterialHost water(2);
        water.add(0, h1s, 0.667 );
        water.add(1, o16s, 0.333 );
        water.copyToGPU();

        SimpleMaterialListHost matList(2,5);
        matList.add( 0, metal, 0 );
        matList.add( 1, water, 1 );
        matList.copyToGPU();

        u234s.copyToGPU();
        u235s.copyToGPU();
        u238s.copyToGPU();
        h1s.copyToGPU();
        o16s.copyToGPU();

    	gpuFloatType_t energy = points.getEnergy(0);
    	unsigned cell = points.getIndex(0);
    	unsigned HashBin = getHashBin( matList.getHashPtr()->getPtr(), energy );
    	gpuFloatType_t expected1 = helper.getTotalXSByMatProp(mp.getPtr(), matList.getPtr(), matList.getHashPtr()->getPtr(), HashBin, cell, energy );
    	gpuFloatType_t expected2 = helper.getTotalXSByMatProp(mp.getPtr(), matList.getPtr(), cell, energy );
    	CHECK_CLOSE( 0.804852, energy, 1e-6);
    	CHECK_EQUAL( 435859, cell);
    	CHECK_CLOSE( 0.410871, expected1, 1e-6);
    	CHECK_CLOSE( expected2, expected1, 1e-6);

    	helper.setupTimers();
    	helper.launchTallyCrossSectionAtCollision(1024, 1024, &points, &matList, &mp );
    	helper.stopTimers();

    	CHECK_CLOSE( expected1, helper.getTally(0), 1e-7 );
    }

    TEST( sum_crossSection_by_startingCell )  {
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

        SimpleMaterialHost metal(3);
        metal.add(0, u234s, 0.01);
        metal.add(1, u235s, 0.98);
        metal.add(2, u238s, 0.01);
        metal.copyToGPU();

        SimpleMaterialHost water(2);
        water.add(0, h1s, 0.667 );
        water.add(1, o16s, 0.333 );
        water.copyToGPU();

        SimpleMaterialListHost matList(2,5);
        matList.add( 0, metal, 0 );
        matList.add( 1, water, 1 );
        matList.copyToGPU();

        u234s.copyToGPU();
        u235s.copyToGPU();
        u238s.copyToGPU();
        h1s.copyToGPU();
        o16s.copyToGPU();

    	gpuFloatType_t energy = points.getEnergy(0);
    	unsigned cell = points.getIndex(0);
    	unsigned HashBin = getHashBin( matList.getHashPtr()->getPtr(), energy );
    	gpuFloatType_t expected = helper.getTotalXSByMatProp(mp.getPtr(), matList.getPtr(), matList.getHashPtr()->getPtr(), HashBin, cell, energy );
    	CHECK_CLOSE( 0.804852, energy, 1e-6);
    	CHECK_EQUAL( 435859, cell);
    	CHECK_CLOSE( 0.410871, expected, 1e-6);

    	expected=0.0;
    	for( unsigned i=0; i<points.size(); ++i){
    		if( points.getIndex(i) == cell ) {
    			energy = points.getEnergy(i);
    			HashBin = getHashBin( matList.getHashPtr()->getPtr(), energy );
    			expected += helper.getTotalXSByMatProp(mp.getPtr(), matList.getPtr(), matList.getHashPtr()->getPtr(), HashBin, cell, energy );
    		}
    	}
    	CHECK_CLOSE( 942.722, expected, 1e-3);

    	helper.setupTimers();
    	helper.launchSumCrossSectionAtCollisionLocation(1024, 1024, &points, &matList, &mp );
    	helper.stopTimers();

    	CHECK_CLOSE( expected, helper.getTally(cell), 1e-3 );
    }

    TEST( rayTraceTally_GodivaR )
    {
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

        SimpleMaterialHost metal(3);
        metal.add(0, u234s, 0.01);
        metal.add(1, u235s, 0.98);
        metal.add(2, u238s, 0.01);
        metal.copyToGPU();

        SimpleMaterialHost water(2);
        water.add(0, h1s, 0.667 );
        water.add(1, o16s, 0.333 );
        water.copyToGPU();

        SimpleMaterialListHost matList(2,5);
        matList.add( 0, metal, 0 );
        matList.add( 1, water, 1 );
        matList.copyToGPU();

        u234s.copyToGPU();
        u235s.copyToGPU();
        u238s.copyToGPU();
        h1s.copyToGPU();
        o16s.copyToGPU();

    	CollisionPointsHost points(2);
    	points.readToMemory( "/usr/projects/mcatk/user/jsweezy/link_files/collisionsGodivaCyl100x100x100InWater.bin"  );
    	points.copyToGPU();
    	CHECK_EQUAL(32786806, points.size());

    	gpuFloatType_t energy = points.getEnergy(0);
    	unsigned cell = points.getIndex(0);
    	unsigned HashBin = getHashBin( matList.getHashPtr()->getPtr(), energy );
    	gpuFloatType_t expected = helper.getTotalXSByMatProp(mp.getPtr(), matList.getPtr(), matList.getHashPtr()->getPtr(), HashBin, cell, energy );
    	CHECK_CLOSE( 0.804852, energy, 1e-6);
    	CHECK_EQUAL( 435859, cell);
    	CHECK_CLOSE( 0.410871, expected, 1e-6);

     	helper.setupTimers();
    	helper.launchRayTraceTally(1024, 1024, &points, &matList, &mp );
    	helper.stopTimers();

//    	CHECK_CLOSE( 0.0803215, helper.getTally(0), 1e-5 );
//    	CHECK_CLOSE( 0.186005, helper.getTally(50+100*100), 1e-4 );

    	CHECK_CLOSE( 0.0266943, helper.getTally(0), 1e-5 );
    	CHECK_CLOSE( 0.064631, helper.getTally(50+100*100), 1e-4 );
    	free(grid_host);
    }
}

SUITE( Collision_fi_looping_tester ) {

	TEST( setup ) {
		gpuCheck();
	}

    TEST( rayTraceTally_GodivaR_wGlobalLauncher )
    {
    	FIGenericGPUTestHelper helper(  1 );

    	cudaReset();
    	GridBinsHost grid(-33.5, 33.5, 100,
    			          -33.5, 33.5, 100,
    			          -33.5, 33.5, 100);
    	grid.copyToGPU();

    	gpuTallyHost tally( grid.getNumCells() );
    	tally.copyToGPU();

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

        SimpleMaterialHost metal(3);
        metal.add(0, u234s, 0.01);
        metal.add(1, u235s, 0.98);
        metal.add(2, u238s, 0.01);
        metal.copyToGPU();

        SimpleMaterialHost water(2);
        water.add(0, h1s, 0.667 );
        water.add(1, o16s, 0.333 );
        water.copyToGPU();

        SimpleMaterialListHost matList(2,5);
        matList.add( 0, metal, 0 );
        matList.add( 1, water, 1 );
        matList.copyToGPU();

        u234s.copyToGPU();
        u235s.copyToGPU();
        u238s.copyToGPU();
        h1s.copyToGPU();
        o16s.copyToGPU();

    	CollisionPointsHost bank1(1000000);
    	bool end = false;
    	unsigned offset = 0;
		end = bank1.readToBank( "/usr/projects/mcatk/user/jsweezy/link_files/collisionsGodivaCyl100x100x100InWater.bin", offset );

    	gpuFloatType_t energy = bank1.getEnergy(0);
    	unsigned cell = bank1.getIndex(0);
    	unsigned HashBin = getHashBin( matList.getHashPtr()->getPtr(), energy );
    	gpuFloatType_t expected = helper.getTotalXSByMatProp(mp.getPtr(), matList.getPtr(), matList.getHashPtr()->getPtr(), HashBin, cell, energy );
     	CHECK_CLOSE( 0.804852, energy, 1e-6);
    	CHECK_EQUAL( 435859, cell);
    	CHECK_CLOSE( 0.410871, expected, 1e-6);

		offset += bank1.size();

    	CollisionPointsHost bank2(1000000);

    	auto cpuWork1 = [&] (void) -> void {
    		if( !end ) {
    			end = bank2.readToBank( "/usr/projects/mcatk/user/jsweezy/link_files/collisionsGodivaCyl100x100x100InWater.bin", offset );
    			offset += bank2.size();
    		}
    	};
    	auto cpuWork2 = [&] (void) -> void {
    		if( !end ) {
    			end = bank1.readToBank( "/usr/projects/mcatk/user/jsweezy/link_files/collisionsGodivaCyl100x100x100InWater.bin", offset );
    			offset += bank1.size();
    		}
    	};

     	helper.setupTimers();

     	bool last = false;
     	while(true){
     		bank1.copyToGPU();
     		if( end ) { last = true; }
     		MonteRay::tripleTime time = launchRayTraceTally(cpuWork1,
     				1024,
     				1024,
     				&grid,
     				&bank1,
     				&matList,
     				&mp,
     				&tally);

        	std::cout << "Debug: Time in GPU raytrace kernel=" << time.gpuTime << " secs.\n";
        	std::cout << "Debug: Time in CPU work =" << time.cpuTime << " secs.\n";
        	std::cout << "Debug: Time total time =" << time.totalTime << " secs.\n\n";
        	if( last ) { break; }

          	bank2.copyToGPU();
        	if( end ) { last = true; }
     		time = launchRayTraceTally(cpuWork2,
     				1024,
     				1024,
     				&grid,
     				&bank2,
     				&matList,
     				&mp,
     				&tally);

        	std::cout << "Debug: Time in GPU raytrace kernel=" << time.gpuTime << " secs.\n";
        	std::cout << "Debug: Time in CPU work =" << time.cpuTime << " secs.\n";
        	std::cout << "Debug: Time total time =" << time.totalTime << " secs.\n\n";
        	if( last ) { break; }

     	};

    	helper.stopTimers();

    	tally.copyToCPU();

//    	CHECK_CLOSE( 0.0803215, tally.getTally(0), 1e-5 );
//    	CHECK_CLOSE( 0.186005, tally.getTally(50+100*100), 1e-4 );

    	CHECK_CLOSE( 0.0754594, tally.getTally(0), 1e-5 );
    	CHECK_CLOSE( 0.175685, tally.getTally(50+100*100), 1e-4 );
    }

}
#endif
