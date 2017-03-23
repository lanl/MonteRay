#include <UnitTest++.h>

#include <iostream>
#include <iomanip>

#include "MonteRayDefinitions.hh"
#include "GPUUtilityFunctions.hh"
#include "gpuTally.h"

#include "genericGPU_test_helper.hh"
#include "gpuTally_test_helper.hh"


SUITE( gpuTally_tester ) {
	TEST( setup ) {
		gpuCheck();
	}
    TEST( ctor ) {
        gpuTallyHost tally(5);
        CHECK( true );
    }

    TEST( set_get ) {
        gpuTallyHost tally(5);
        tally.setTally(0, 99.0);
        CHECK_CLOSE( 99.0, tally.getTally(0), 1e-4 );
    }

    TEST( get_size ) {
        gpuTallyHost tally(5);
        CHECK_EQUAL(5, tally.size() );
    }
    TEST( send_to_gpu ) {
        gpuTallyHost tally(5);
        tally.setTally(0, 99.0);
        tally.copyToGPU();
        tally.setTally(0, 0.0);
        tally.copyToCPU();
        CHECK_CLOSE( 99.0, tally.getTally(0), 1e-4 );
    }
    TEST( clear ) {
        gpuTallyHost tally(5);
        tally.setTally(0, 99.0);
        tally.copyToGPU();
        tally.clear();
        tally.copyToCPU();
        CHECK_CLOSE( 0.0, tally.getTally(0), 1e-4 );
    }

    TEST( add_on_GPU ) {
        gpuTallyHost tally(5);
        GPUTallyTestHelper helper;
        helper.launchAddTally(&tally, 0, 2.5, 1.25 );
        CHECK_CLOSE( 3.75, tally.getTally(0), 1e-4 );
    }

}
