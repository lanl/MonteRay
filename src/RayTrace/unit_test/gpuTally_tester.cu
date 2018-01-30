#include <UnitTest++.h>

#include <iostream>
#include <iomanip>

#include "MonteRayDefinitions.hh"
#include "GPUUtilityFunctions.hh"
#include "gpuTally.hh"

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
#ifdef __CUDACC__
        tally.copyToGPU();
        tally.setTally(0, 0.0);
        tally.copyToCPU();
#endif
        CHECK_CLOSE( 99.0, tally.getTally(0), 1e-4 );
    }
    TEST( get_default_values_from_gpu ) {
         gpuTallyHost tally(5);
         tally.copyToGPU();
         tally.copyToCPU();
         CHECK_CLOSE( 0.0, tally.getTally(0), 1e-4 );
         CHECK_CLOSE( 0.0, tally.getTally(1), 1e-4 );
         CHECK_CLOSE( 0.0, tally.getTally(2), 1e-4 );
         CHECK_CLOSE( 0.0, tally.getTally(3), 1e-4 );
         CHECK_CLOSE( 0.0, tally.getTally(4), 1e-4 );
     }
    TEST( clear_all_on_cpu ) {
         gpuTallyHost tally(5);
         tally.setTally(0, 99.0);
         tally.setTally(1, 99.0);
         tally.setTally(2, 99.0);
         tally.setTally(3, 99.0);
         tally.setTally(4, 99.0);
#ifdef __CUDACC__
         tally.copyToGPU();
         tally.setTally(0, 0.0);
         tally.setTally(1, 0.0);
         tally.setTally(2, 0.0);
         tally.setTally(3, 0.0);
         tally.setTally(4, 0.0);
         tally.copyToCPU();
#endif
         CHECK_CLOSE( 99.0, tally.getTally(0), 1e-4 );
         CHECK_CLOSE( 99.0, tally.getTally(1), 1e-4 );
         CHECK_CLOSE( 99.0, tally.getTally(2), 1e-4 );
         CHECK_CLOSE( 99.0, tally.getTally(3), 1e-4 );
         CHECK_CLOSE( 99.0, tally.getTally(4), 1e-4 );
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
    TEST( write_to_file_read_from_file ) {
    	std::string filename = "test_write_tally_to_file_read_from_file.bin";
        gpuTallyHost tally(5);
        tally.setTally(4, 99.0);
        tally.write( filename);

        gpuTallyHost readTally(1);
        CHECK_EQUAL( 1, readTally.size());
        readTally.read( filename );
        CHECK_EQUAL( 5, readTally.size());
        CHECK_CLOSE( 99.0, tally.getTally(4), 1e-4 );
    }

}
