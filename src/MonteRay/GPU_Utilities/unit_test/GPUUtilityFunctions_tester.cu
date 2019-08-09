#include <UnitTest++.h>

#include <iostream>

#include "GPUUtilityFunctions.hh"
#include "MonteRayTypes.hh"

SUITE( gpu_utility_functions_simple_tests ) {
    using namespace MonteRay;

    TEST( setLaunchBounds_pos_threads_pos_nRaysPerThread ) {
        auto bounds = setLaunchBounds( 1, 1, 100000);
        CHECK_EQUAL( 3125, bounds.first ); // blocks
        CHECK_EQUAL( 32, bounds.second ); // threads
    }

    TEST( setLaunchBounds_neg_threads_neg_nRaysPerThread ) {
        auto bounds = setLaunchBounds( -1, -1, 100000);
        CHECK_EQUAL( 1, bounds.first ); // blocks
        CHECK_EQUAL( 1, bounds.second ); // threads
    }

    TEST( setLaunchBounds_neg_threads_pos_nRaysPerThread ) {
        auto bounds = setLaunchBounds( -1, 10, 100000);
        CHECK_EQUAL( 10000, bounds.first ); // blocks
        CHECK_EQUAL( 1, bounds.second ); // threads
    }

    TEST( setLaunchBounds_pos_threads_neg_nRaysPerThread ) {
        auto bounds = setLaunchBounds( 64, -10, 100000);
        CHECK_EQUAL( 10, bounds.first ); // blocks
        CHECK_EQUAL( 64, bounds.second ); // threads
    }

    TEST( setLaunchBounds_num_threads_non32multiple ) {
        auto bounds = setLaunchBounds( 63, -10, 100000);
        CHECK_EQUAL( 10, bounds.first ); // blocks
        CHECK_EQUAL( 64, bounds.second ); // threads
    }

    TEST( setLaunchBounds_more_threads_than_rays ) {
        auto bounds = setLaunchBounds( 512, 10, 100 );
        CHECK_EQUAL( 1, bounds.first ); // blocks
        CHECK_EQUAL( 128, bounds.second ); // threads
    }

    TEST( setLaunchBounds_more_threads_than_MONTERAY_MAX_THREADS_PER_BLOCK ) {
        auto bounds = setLaunchBounds( MONTERAY_MAX_THREADS_PER_BLOCK+32, 10, 1000 );
        CHECK_EQUAL( 1, bounds.first ); // blocks
        CHECK_EQUAL( MONTERAY_MAX_THREADS_PER_BLOCK, bounds.second ); // threads
    }

    TEST( setLaunchBounds_1_256_2568016 ) {
        auto bounds = setLaunchBounds( 256, 1, 2568016 );
        CHECK_EQUAL( 390, bounds.first ); // blocks
        CHECK_EQUAL( 256, bounds.second ); // threads
    }
}
