#include <UnitTest++.h>

#include <iostream>
#include <iomanip>

#include "MonteRayDefinitions.hh"
#include "GPUUtilityFunctions.hh"
#include "GPUTiming.hh"

#include "genericGPU_test_helper.hh"
#include "gpuTiming_test_helper.hh"

using namespace MonteRay;

SUITE( gpuTiming_tester ) {
	TEST( setup ) {
		gpuCheck();
	}
    TEST( ctor ) {
        gpuTimingHost timing;
        CHECK( true );
    }

    TEST( check_gpu_rate_set ) {
    	gpuTimingHost timing;
    	CHECK( timing.getRate() != 0 );
    	//std::cout << "Debug: rate=" << timing.getRate() << std::endl;
    	// titanx = 1076000000 = 1.076 GHz
    }

    TEST( check_send_to_gpu ) {
       	gpuTimingHost timing;
       	timing.setClockStop( timing.getRate() );
#ifdef __CUDACC__
       	timing.copyToGPU();
       	timing.setClockStop( 0 ); // reset stop
#endif
       	CHECK_CLOSE( 1.0, timing.getGPUTime(), 1e-6 );
    }

    TEST( check_zero_time ) {
       	gpuTimingHost timing;

       	timing.setClockStart( 0 );
       	timing.setClockStop( 0 );
#ifdef __CUDACC__
       	timing.copyToGPU();
       	timing.setClockStart( 100 );
       	timing.setClockStop( 0 );
#endif
       	CHECK_CLOSE( 0.0, timing.getGPUTime(), 1e-6 );
    }

    TEST( check_default_zero_time ) {
       	gpuTimingHost timing;
#ifdef __CUDACC__
       	timing.setClockStart( 100 );
       	timing.setClockStop( 0 );
#endif
       	CHECK_CLOSE( 0.0, timing.getGPUTime(), 1e-6 );
    }

    TEST( test_gpuTime_via_sleep ) {
    	// Sleep 0.10 secs
       	gpuTimingHost timer;

       	GPUTimingTestHelper helper;

       	helper.launchGPUSleep( timer.getRate()*0.1, &timer );
//      	std::cout << "Debug: delta time=" << std::setprecision(5) << timer.getGPUTime() << std::endl;
//       	std::cout << "Debug: start time=" << timer.getClockStart() << std::endl;
//       	std::cout << "Debug: start time=" << timer.getClockStop() << std::endl;
//       	CHECK_CLOSE( 0.10, timer.getGPUTime(), 1e-2 );
    }

    TEST( test_many_thread_gpuTime_by_stream ) {
    	// Sleep 0.10 secs
       	gpuTimingHost timer;

       	GPUTimingTestHelper helper;
       	gpuFloatType_t gpuTime =  helper.launchGPUStreamSleep(1,1, timer.getRate()*1.0, 5000 );

      	std::cout << "Debug: gpu time=" << std::setprecision(5) << gpuTime << std::endl;
//       	std::cout << "Debug: start time=" << timer.getClockStart() << std::endl;
//       	std::cout << "Debug: start time=" << timer.getClockStop() << std::endl;
       	//CHECK_CLOSE( 0.10, timer.getGPUTime(), 1e-2 );
    }
}
