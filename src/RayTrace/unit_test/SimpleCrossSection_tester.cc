#include <UnitTest++.h>

#include "GPUSync.hh"
#include "GPUUtilityFunctions.hh"
#include "SimpleCrossSection.h"

#include "SimpleCrossSection_test_helper.hh"

SUITE( SimpleCrossSection_tester ) {
	TEST( setup ) {
		gpuCheck();
	}
    TEST( ctor ) {
        SimpleCrossSectionHost xs(10);
        CHECK_EQUAL(10, xs.size() );
    }
    TEST_FIXTURE(SimpleCrossSectionTestHelper, get_id ) {
    	SimpleCrossSectionHost* xs = new SimpleCrossSectionHost(4);
    	xs->setTotalXS(0, 0.0, 4.0 );
    	xs->setTotalXS(1, 1.0, 3.0 );
    	xs->setTotalXS(2, 2.0, 2.0 );
    	xs->setTotalXS(3, 3.0, 1.0 );

    	CHECK_EQUAL(-1, xs->getID() );
    	xs->setID(3);
    	CHECK_EQUAL(3, xs->getID() );

    	// can't set twice
    	xs->setID(4);
    	CHECK_EQUAL(3, xs->getID() );

    	delete xs;
    }
    TEST_FIXTURE(SimpleCrossSectionTestHelper, get_total_xs_from_gpu ) {
    	SimpleCrossSectionHost* xs = new SimpleCrossSectionHost(4);
    	xs->setTotalXS(0, 0.0, 4.0 );
    	xs->setTotalXS(1, 1.0, 3.0 );
    	xs->setTotalXS(2, 2.0, 2.0 );
    	xs->setTotalXS(3, 3.0, 1.0 );

    	xs->copyToGPU();

    	gpuFloatType_t energy = 0.5;

    	setupTimers();
    	gpuFloatType_t totalXS = launchGetTotalXS( xs, energy);
    	stopTimers();

    	CHECK_CLOSE( 3.5f, totalXS, 1e-7 );

    	delete xs;
    }

    TEST_FIXTURE(SimpleCrossSectionTestHelper, load_u235_from_file)
    {
    	SimpleCrossSectionHost* xs = new SimpleCrossSectionHost(1);
    	xs->read( "/usr/projects/mcatk/user/jsweezy/link_files/u235_simpleCrossSection.bin");

    	gpuFloatType_t energy = 2.0;

    	CHECK_EQUAL( 24135, xs->size() );
    	CHECK_CLOSE( 233.025, xs->getAWR(), 1e-3 );
    	double value = getTotalXS(xs->getXSPtr(), energy);
    	CHECK_CLOSE( 7.17639378000f, value, 1e-6);

    	xs->copyToGPU();

    	GPUSync sync;
    	gpuFloatType_t totalXS = launchGetTotalXS( xs, energy);
    	sync.sync();

    	CHECK_CLOSE( 7.17639378000f, totalXS, 1e-7 );

    	delete xs;
    }

}
