#include <UnitTest++.h>

#include <cmath>

#include "SimpleMaterialProperties.h"
#include "genericGPU_test_helper.hh"

SUITE( SimpleMaterialProperties_tester ) {
	TEST( SimpleMaterialProperties_read_lnk3dnt ) {
		SimpleMaterialPropertiesHost mp(2);
		mp.read( "/usr/projects/mcatk/user/jsweezy/link_files/godivaR_geometry_100x100x100.bin" );

		CHECK_EQUAL( 100*100*100, mp.getNumCells());

		gpuFloatType_t mass = mp.sumMatDensity(0) * std::pow( ((33.5*2.0)/100), 3);
		CHECK_CLOSE( 2.21573E+04, mass, 1e-1);

		mass = mp.sumMatDensity(1) * std::pow( ((33.5*2.0)/100), 3);
		CHECK_CLOSE( 1.55890E+05, mass, 1);
	}

	TEST_FIXTURE(GenericGPUTestHelper, SimpleMaterialProperties_getNumCells_on_gpu)
	{
	    SimpleMaterialPropertiesHost mp(2);
	    mp.read( "/usr/projects/mcatk/user/jsweezy/link_files/godivaR_geometry_100x100x100.bin" );
	    mp.copyToGPU();

		setupTimers();
		unsigned numCells = mp.launchGetNumCells();
		stopTimers();

		CHECK_CLOSE( 100*100*100, numCells, 1e-1);
	}

	TEST_FIXTURE(GenericGPUTestHelper, SimpleMaterialProperties_sum_on_gpu)
	{
	    SimpleMaterialPropertiesHost mp(2);
	    mp.read( "/usr/projects/mcatk/user/jsweezy/link_files/godivaR_geometry_100x100x100.bin" );
	    mp.copyToGPU();

		setupTimers();
		gpuFloatType_t mass = mp.launchSumMatDensity(0);
		stopTimers();

	    mass *= std::pow( ((33.5*2.0)/100), 3);
	    CHECK_CLOSE( 2.21573E+04, mass, 1e-1);
	}
}
