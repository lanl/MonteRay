#include <UnitTest++.h>

#include <iostream>
#include <iomanip>

#include "MonteRayTally_GPU_test_helper.hh"

namespace MontRayTally_GPU_tester_namespace {

using namespace MonteRay;
using namespace MonteRayTallyGPUTestHelper;

SUITE( MonteRayTally_GPU_tester ) {

    TEST_FIXTURE(MonteRayTallyGPUTester, ctor1 ) {
        MonteRayTallyGPUTester tally;
        CHECK_EQUAL( 1, tally.getNumSpatialBins() );
        CHECK_EQUAL( 6, tally.getNumTimeBins() );
        CHECK_EQUAL( 6, tally.getCPUTallySize() );
        CHECK_EQUAL( 6, tally.getGPUTallySize() );
    }

    TEST_FIXTURE(MonteRayTallyGPUTester, ctor2 ) {
        MonteRayTallyGPUTester tally(2);
        CHECK_EQUAL( 1, tally.getNumSpatialBins() );
        CHECK_EQUAL( 1, tally.getNumTimeBins() );
        CHECK_EQUAL( 1, tally.getCPUTallySize() );
        CHECK_EQUAL( 1, tally.getGPUTallySize() );
    }

    TEST_FIXTURE(MonteRayTallyGPUTester, scoreByIndex ) {
        MonteRayTallyGPUTester tally;

        tally.scoreByIndex(1.0f, 0, 0);
        CHECK_CLOSE( 1.0, tally.getTally(0,0), 1e-6);

        tally.scoreByIndex(1.0f, 0, 0);
        CHECK_CLOSE( 2.0, tally.getTally(0,0), 1e-6);
    }

    TEST_FIXTURE(MonteRayTallyGPUTester, score ) {
        MonteRayTallyGPUTester tally;

        CHECK_CLOSE( 0.0, tally.getTally(0,0), 1e-6);
        CHECK_CLOSE( 0.0, tally.getTally(0,1), 1e-6);
        CHECK_CLOSE( 0.0, tally.getTally(0,2), 1e-6);
        CHECK_CLOSE( 0.0, tally.getTally(0,3), 1e-6);
        CHECK_CLOSE( 0.0, tally.getTally(0,4), 1e-6);
        CHECK_CLOSE( 0.0, tally.getTally(0,5), 1e-6);

        gpuFloatType_t time = 1.5;
        tally.score(1.0f, 0, time);
        CHECK_CLOSE( 0.0, tally.getTally(0,0), 1e-6);
        CHECK_CLOSE( 1.0, tally.getTally(0,1), 1e-6);

        time = 2.5;
        tally.score(2.0f, 0, time);
        CHECK_CLOSE( 0.0, tally.getTally(0,0), 1e-6);
        CHECK_CLOSE( 1.0, tally.getTally(0,1), 1e-6);
        CHECK_CLOSE( 2.0, tally.getTally(0,2), 1e-6);

        time = 400.0;
        tally.score( 4.0f, 0, time);
        CHECK_CLOSE( 0.0, tally.getTally(0,0), 1e-6);
        CHECK_CLOSE( 1.0, tally.getTally(0,1), 1e-6);
        CHECK_CLOSE( 2.0, tally.getTally(0,2), 1e-6);
        CHECK_CLOSE( 0.0, tally.getTally(0,3), 1e-6);
        CHECK_CLOSE( 0.0, tally.getTally(0,4), 1e-6);
        CHECK_CLOSE( 4.0, tally.getTally(0,5), 1e-6);
    }

    TEST_FIXTURE(MonteRayTallyGPUTester, score_noTimeBins ) {
        MonteRayTallyGPUTester tally(2);

        CHECK_CLOSE( 0.0, tally.getTally(0,0), 1e-6);

        gpuFloatType_t time = 1.5;
        tally.score(1.0f, 0, time);
        CHECK_CLOSE( 1.0, tally.getTally(0,0), 1e-6);

        time = 2.5;
        tally.score(2.0f, 0, time);
        CHECK_CLOSE( 3.0, tally.getTally(0,0), 1e-6);

        time = 400.0;
        tally.score( 4.0f, 0, time);
        CHECK_CLOSE( 7.0, tally.getTally(0,0), 1e-6);

    }

    TEST_FIXTURE(MonteRayTallyGPUTester, copyToCPU ) {
        //printf("Debug:  MonteRayTally_GPU_tester.cu -- copyToCPU test\n");
        MonteRayTallyGPUTester tally;

        gpuFloatType_t time = 1.5;
        tally.score(1.0f, 0, time);

        time = 2.5;
        tally.score(2.0f, 0, time);

        time = 400.0;
        tally.score( 4.0f, 0, time);

        CHECK_EQUAL( 5, tally.getTimeIndex(time) );
        CHECK_EQUAL( 0, tally.getIndex(0,0) );
        CHECK_EQUAL( 5, tally.getIndex(0,5) );



        CHECK_EQUAL( 6, tally.getCPUTallySize() );
        CHECK_EQUAL( 6, tally.getGPUTallySize() );

        // On GPU
        CHECK_CLOSE( 0.0, tally.getTally(0,0), 1e-6);
        CHECK_CLOSE( 1.0, tally.getTally(0,1), 1e-6);
        CHECK_CLOSE( 2.0, tally.getTally(0,2), 1e-6);
        CHECK_CLOSE( 0.0, tally.getTally(0,3), 1e-6);
        CHECK_CLOSE( 0.0, tally.getTally(0,4), 1e-6);
        CHECK_CLOSE( 4.0, tally.getTally(0,5), 1e-6);

        tally.copyToCPU();
        // On CPU
        CHECK_CLOSE( 0.0, tally.getCPUTally(0,0), 1e-6);
        CHECK_CLOSE( 1.0, tally.getCPUTally(0,1), 1e-6);
        CHECK_CLOSE( 2.0, tally.getCPUTally(0,2), 1e-6);
        CHECK_CLOSE( 0.0, tally.getCPUTally(0,3), 1e-6);
        CHECK_CLOSE( 0.0, tally.getCPUTally(0,4), 1e-6);
        CHECK_CLOSE( 4.0, tally.getCPUTally(0,5), 1e-6);

    }

    TEST_FIXTURE(MonteRayTallyGPUTester, copyToCPU_noTimeBins ) {
        //printf("Debug:  MonteRayTally_GPU_tester.cu -- copyToCPU_noTimeBins test\n");
        MonteRayTallyGPUTester tally(2);

        gpuFloatType_t time = 1.5;
        tally.score(1.0f, 0, time);

        CHECK_EQUAL( 0, tally.getTimeIndex(time) );
        CHECK_EQUAL( 0, tally.getIndex(0,0));

        // copyToCPU should copy the tally to the CPU and zero the value on the GPU
        CHECK_CLOSE( 1.0, tally.getTally(0,0), 1e-6);
#ifdef __CUDACC__
        CHECK_CLOSE( 0.0, tally.getCPUTally(0,0), 1e-6);
#else
        CHECK_CLOSE( 1.0, tally.getCPUTally(0,0), 1e-6);
#endif
        tally.copyToCPU();
#ifdef __CUDACC__
        CHECK_CLOSE( 0.0, tally.getTally(0,0), 1e-6);
#else
        CHECK_CLOSE( 1.0, tally.getTally(0,0), 1e-6);
#endif
        CHECK_CLOSE( 1.0, tally.getCPUTally(0,0), 1e-6);

        time = 2.5;
        tally.score(2.0f, 0, time);
        tally.copyToCPU();

#ifdef __CUDACC__
        CHECK_CLOSE( 0.0, tally.getTally(0,0), 1e-6);
#else
        CHECK_CLOSE( 3.0, tally.getTally(0,0), 1e-6);
#endif
        CHECK_CLOSE( 3.0, tally.getCPUTally(0,0), 1e-6);
    }

}

} // end namespace
