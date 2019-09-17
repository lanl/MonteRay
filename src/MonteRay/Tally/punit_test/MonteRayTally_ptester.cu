#include <UnitTest++.h>

#include <iostream>
#include <mpi.h>
#include <unistd.h>

#include "../unit_test/MonteRayTally_GPU_test_helper.hh"

namespace MonteRayTally_ptester_namespace{

using namespace MonteRay;
using namespace MonteRayTallyGPUTestHelper;

SUITE( MonteRayTally_ptester ){

    class setup{
    public:
        setup() : PA( MonteRayParallelAssistant::getInstance() ) {}

        ~setup(){}

        const MonteRayParallelAssistant& PA;
    };

    TEST_FIXTURE(setup, score_and_gather ) {
        MonteRayTally tally;
        tally.setupForParallel();
        tally.initialize();

        tally.scoreByIndex(1.0f, 0, 0);
        tally.gatherWorkGroup(); // used for testing only
        tally.gather();

        if( PA.getWorldRank() == 0 ) {
            CHECK_CLOSE( 1.0*PA.getWorldSize(), tally.getTally(0,0), 1e-6);
        }
    }

    TEST_FIXTURE(setup, score_and_gather_twice ) {
        MonteRayTally tally;
        tally.setupForParallel();
        tally.initialize();

        tally.scoreByIndex(1.0f, 0, 0);
        tally.gatherWorkGroup(); // used for testing only
        tally.gather();

        if( PA.getWorldRank() == 0 ) {
            CHECK_CLOSE( 1.0*PA.getWorldSize(), tally.getTally(0,0), 1e-6);
        }

        tally.scoreByIndex(1.0f, 0, 0);
        if( PA.getWorldRank() == 0 ) {
            gpuTallyType_t value = tally.getTally(0,0);
            //std::cout << "Debug:  Rank = " << PA.getWorldRank() << " value=" << value << "\n";
            CHECK_CLOSE( 1.0*PA.getWorldSize()+1.0, value, 1e-6);
        } else {
            gpuTallyType_t value = tally.getTally(0,0);
            //std::cout << "Debug:  Rank = " << PA.getWorldRank() << " value=" << value << "\n";
            CHECK_CLOSE( 1.0, value, 1e-6);
        }

        tally.gatherWorkGroup(); // used for testing only
        tally.gather();

        if( PA.getWorldRank() == 0 ) {
            CHECK_CLOSE( 2.0*PA.getWorldSize(), tally.getTally(0,0), 1e-6);
        }
    }

    TEST_FIXTURE(setup, tally_with_timeBins ) {
        MonteRayTally tally;
        tally.setupForParallel();
        std::vector<MonteRay::gpuFloatType_t> timeEdges= { 1.0, 2.0, 10.0, 99.0, 100.0 };
        tally.setTimeBinEdges(timeEdges);
        tally.initialize();

        gpuFloatType_t time = 1.5;
        tally.score(1.0f, 0, time);
        tally.gatherWorkGroup(); // used for testing only
        tally.gather();

        if( PA.getWorldRank() == 0 ) {
            CHECK_CLOSE( 0.0*PA.getWorldSize(), tally.getTally(0,0), 1e-6);
            CHECK_CLOSE( 1.0*PA.getWorldSize(), tally.getTally(0,1), 1e-6);
            CHECK_CLOSE( 0.0*PA.getWorldSize(), tally.getTally(0,2), 1e-6);
            CHECK_CLOSE( 0.0*PA.getWorldSize(), tally.getTally(0,3), 1e-6);
            CHECK_CLOSE( 0.0*PA.getWorldSize(), tally.getTally(0,4), 1e-6);
            CHECK_CLOSE( 0.0*PA.getWorldSize(), tally.getTally(0,5), 1e-6);
        }

        time = 2.5;
        tally.score(2.0f, 0, time);
        tally.gatherWorkGroup(); // used for testing only
        tally.gather();

        if( PA.getWorldRank() == 0 ) {
            CHECK_CLOSE( 0.0*PA.getWorldSize(), tally.getTally(0,0), 1e-6);
            CHECK_CLOSE( 1.0*PA.getWorldSize(), tally.getTally(0,1), 1e-6);
            CHECK_CLOSE( 2.0*PA.getWorldSize(), tally.getTally(0,2), 1e-6);
            CHECK_CLOSE( 0.0*PA.getWorldSize(), tally.getTally(0,3), 1e-6);
            CHECK_CLOSE( 0.0*PA.getWorldSize(), tally.getTally(0,4), 1e-6);
            CHECK_CLOSE( 0.0*PA.getWorldSize(), tally.getTally(0,5), 1e-6);
        }

        time = 400.0;
        tally.score(4.0f, 0, time);
        tally.gatherWorkGroup(); // used for testing only
        tally.gather();

        if( PA.getWorldRank() == 0 ) {
            CHECK_CLOSE( 0.0*PA.getWorldSize(), tally.getTally(0,0), 1e-6);
            CHECK_CLOSE( 1.0*PA.getWorldSize(), tally.getTally(0,1), 1e-6);
            CHECK_CLOSE( 2.0*PA.getWorldSize(), tally.getTally(0,2), 1e-6);
            CHECK_CLOSE( 0.0*PA.getWorldSize(), tally.getTally(0,3), 1e-6);
            CHECK_CLOSE( 0.0*PA.getWorldSize(), tally.getTally(0,4), 1e-6);
            CHECK_CLOSE( 4.0*PA.getWorldSize(), tally.getTally(0,5), 1e-6);
        }
    }

    TEST_FIXTURE(setup, tally_with_timeBins_scoreOnSharedMemoryRank0 ) {
        const bool debug = false;
        MonteRayTally tally;
        tally.setupForParallel();
        std::vector<MonteRay::gpuFloatType_t> timeEdges= { 1.0, 2.0, 10.0, 99.0, 100.0 };
        tally.setTimeBinEdges(timeEdges);
        tally.initialize();

        char hostname[1024];
        gethostname(hostname, 1024);

        if( debug )
        std::cout << "tally_with_timeBins_scoreOnSharedMemoryRank0 -- hostname = " << hostname <<
                     ", PA.getWorldRank()=" << PA.getWorldRank() <<
                     ", PA.getWorldSize()=" << PA.getWorldSize() <<
                     ", PA.getWorkGroupRank()=" << PA.getWorkGroupRank() <<
                     ", PA.getWorkGroupSize()=" << PA.getWorkGroupSize() <<
                     "\n";

        gpuFloatType_t time = 1.5;
        if( PA.getWorkGroupRank() == 0 ) {
            tally.score(1.0f, 0, time);
        }
        tally.gather();

        if( PA.getWorldRank() == 0 ) {
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,0), 1e-6);
            CHECK_CLOSE( 1.0*PA.getInterWorkGroupSize(), tally.getTally(0,1), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,2), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,3), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,4), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,5), 1e-6);
        } else {
            double value = 0.0;
            for( unsigned i=0; i<6; ++i){
                value += tally.getTally(0,i);
            }
            if( value > 0.0 ) {
                std::cout << "Failure -- value > 0, value =" << value << "\n" <<
                             "---------- tally_with_timeBins_scoreOnSharedMemoryRank0 -- hostname = " << hostname <<
                             ", PA.getWorldRank()=" << PA.getWorldRank() <<
                             ", PA.getWorldSize()=" << PA.getWorldSize() <<
                             ", PA.getWorkGroupRank()=" << PA.getWorkGroupRank() <<
                             ", PA.getWorkGroupSize()=" << PA.getWorkGroupSize() <<
                             "\n";
            }
            CHECK_CLOSE( 0.0, value, 1e-6);
        }

        time = 2.5;
        if( PA.getWorkGroupRank() == 0 ) {
            tally.score(2.0f, 0, time);
        }
        tally.gather();

        if( PA.getWorldRank() == 0 ) {
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,0), 1e-6);
            CHECK_CLOSE( 1.0*PA.getInterWorkGroupSize(), tally.getTally(0,1), 1e-6);
            CHECK_CLOSE( 2.0*PA.getInterWorkGroupSize(), tally.getTally(0,2), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,3), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,4), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,5), 1e-6);
        }

        time = 400.0;
        if( PA.getWorkGroupRank() == 0 ) {
            tally.score(4.0f, 0, time);
        }
        tally.gather();

        if( PA.getWorldRank() == 0 ) {
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,0), 1e-6);
            CHECK_CLOSE( 1.0*PA.getInterWorkGroupSize(), tally.getTally(0,1), 1e-6);
            CHECK_CLOSE( 2.0*PA.getInterWorkGroupSize(), tally.getTally(0,2), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,3), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,4), 1e-6);
            CHECK_CLOSE( 4.0*PA.getInterWorkGroupSize(), tally.getTally(0,5), 1e-6);
        }
    }

    TEST_FIXTURE(setup, tallyOnGPU_with_timeBins ) {
        const bool debug = false;
        MonteRayTallyGPUTester tally;
        CHECK_EQUAL( true, MonteRayParallelAssistant::getInstance().isParallel() );

        char hostname[1024];
        gethostname(hostname, 1024);

        if( debug )
        std::cout << "tallyOnGPU_with_timeBins -- hostname = " << hostname <<
                     ", PA.getWorldRank()=" << PA.getWorldRank() <<
                     ", PA.getWorldSize()=" << PA.getWorldSize() <<
                     ", PA.getWorkGroupRank()=" << PA.getWorkGroupRank() <<
                     ", PA.getWorkGroupSize()=" << PA.getWorkGroupSize() <<
                     "\n";

        gpuFloatType_t time = 1.5;
        if( PA.getWorkGroupRank() == 0 ) {
            // only shared memory rank 0 can score
            tally.score(1.0f, 0, time);
        }
        tally.gather();

        if( PA.getWorldRank() == 0 ) {
#ifdef __CUDACC__
            // GPU is zero'd after the gather.
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,0), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,1), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,2), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,3), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,4), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,5), 1e-6);
#endif

            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getCPUTally(0,0), 1e-6);
            CHECK_CLOSE( 1.0*PA.getInterWorkGroupSize(), tally.getCPUTally(0,1), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getCPUTally(0,2), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getCPUTally(0,3), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getCPUTally(0,4), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getCPUTally(0,5), 1e-6);
        } else {
            double gpuSum = 0.0;
            double cpuSum = 0.0;
            for( unsigned i=0; i<6; ++i){
                gpuSum += tally.getTally(0,i);
                cpuSum += tally.getCPUTally(0,i);
            }
            if( gpuSum > 0.0 ) {
                std::cout << "Failure -- gpuSum > 0, gpuSum =" << gpuSum << "\n" <<
                             "---------- tallyOnGPU_with_timeBins -- hostname = " << hostname <<
                             ", PA.getWorldRank()=" << PA.getWorldRank() <<
                             ", PA.getWorldSize()=" << PA.getWorldSize() <<
                             ", PA.getWorkGroupRank()=" << PA.getWorkGroupRank() <<
                             ", PA.getWorkGroupSize()=" << PA.getWorkGroupSize() <<
                             "\n";
            }
            CHECK_CLOSE( 0.0, gpuSum, 1e-6);

            if( cpuSum > 0.0 ) {
                std::cout << "Failure -- cpuSum > 0, cpuSum =" << cpuSum << "\n" <<
                             "---------- tallyOnGPU_with_timeBins -- hostname = " << hostname <<
                             ", PA.getWorldRank()=" << PA.getWorldRank() <<
                             ", PA.getWorldSize()=" << PA.getWorldSize() <<
                             ", PA.getWorkGroupRank()=" << PA.getWorkGroupRank() <<
                             ", PA.getWorkGroupSize()=" << PA.getWorkGroupSize() <<
                              "\n";
            }
            CHECK_CLOSE( 0.0, cpuSum, 1e-6);
        }

        time = 2.5;
        if( PA.getWorkGroupRank() == 0 ) {
            tally.score(2.0f, 0, time);
        }
        tally.gather();

        if( PA.getWorldRank() == 0 ) {
#ifdef __CUDACC__
            // GPU is zero'd after the gather.
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,0), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,1), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,2), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,3), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,4), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,5), 1e-6);
#endif

            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getCPUTally(0,0), 1e-6);
            CHECK_CLOSE( 1.0*PA.getInterWorkGroupSize(), tally.getCPUTally(0,1), 1e-6);
            CHECK_CLOSE( 2.0*PA.getInterWorkGroupSize(), tally.getCPUTally(0,2), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getCPUTally(0,3), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getCPUTally(0,4), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getCPUTally(0,5), 1e-6);
        }

        time = 400.0;
        if( PA.getWorkGroupRank() == 0 ) {
            tally.score(4.0f, 0, time);
        }
        tally.gather();

        if( PA.getWorldRank() == 0 ) {
#ifdef __CUDACC__
            // GPU is zero'd after the gather.
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,0), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,1), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,2), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,3), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,4), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getTally(0,5), 1e-6);
#endif

            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getCPUTally(0,0), 1e-6);
            CHECK_CLOSE( 1.0*PA.getInterWorkGroupSize(), tally.getCPUTally(0,1), 1e-6);
            CHECK_CLOSE( 2.0*PA.getInterWorkGroupSize(), tally.getCPUTally(0,2), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getCPUTally(0,3), 1e-6);
            CHECK_CLOSE( 0.0*PA.getInterWorkGroupSize(), tally.getCPUTally(0,4), 1e-6);
            CHECK_CLOSE( 4.0*PA.getInterWorkGroupSize(), tally.getCPUTally(0,5), 1e-6);
        }
    }

}

} // end namespace
