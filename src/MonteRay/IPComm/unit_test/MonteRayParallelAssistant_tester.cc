#include <UnitTest++.h>

#include <iostream>

#include "MonteRayParallelAssistant.hh"

namespace MonteRayParallelAssistant_tester_namespace{

SUITE( MonteRayParallelAssistant_tester ){
    using namespace MonteRay;

    TEST( ctor ) {
        const MonteRayParallelAssistant& PA( MonteRayParallelAssistant::getInstance() );
        CHECK_EQUAL( false, PA.isParallel() );
        CHECK_EQUAL( false, PA.usingSingleProcWorkGroup() );
        CHECK_EQUAL(     0, PA.getWorldRank() );
        CHECK_EQUAL(     1, PA.getWorldSize() );
        CHECK_EQUAL(     0, PA.getSharedMemoryRank() );
        CHECK_EQUAL(     1, PA.getSharedMemorySize() );
        CHECK_EQUAL(     0, PA.getWorkGroupRank() );
        CHECK_EQUAL(     1, PA.getWorkGroupSize() );
        CHECK_EQUAL(     0, PA.getInterWorkGroupRank() );
        CHECK_EQUAL(     1, PA.getInterWorkGroupSize() );
    }

    TEST( getDeviceAssignmentMapping_1GPU ) {
        const MonteRayParallelAssistant& PA( MonteRayParallelAssistant::getInstance() );

        CHECK_EQUAL(0, PA.calcDeviceID( 4, 1, 0 ) );
        CHECK_EQUAL(0, PA.calcDeviceID( 4, 1, 1 ) );
        CHECK_EQUAL(0, PA.calcDeviceID( 4, 1, 2 ) );
        CHECK_EQUAL(0, PA.calcDeviceID( 4, 1, 3 ) );
    }

    TEST( getDeviceAssignmentMapping_2GPUs ) {
        const MonteRayParallelAssistant& PA( MonteRayParallelAssistant::getInstance() );

        CHECK_EQUAL(0, PA.calcDeviceID( 7, 2, 0 ) );
        CHECK_EQUAL(0, PA.calcDeviceID( 7, 2, 1 ) );
        CHECK_EQUAL(0, PA.calcDeviceID( 7, 2, 2 ) );
        CHECK_EQUAL(0, PA.calcDeviceID( 7, 2, 3 ) );
        CHECK_EQUAL(1, PA.calcDeviceID( 7, 2, 4 ) );
        CHECK_EQUAL(1, PA.calcDeviceID( 7, 2, 5 ) );
        CHECK_EQUAL(1, PA.calcDeviceID( 7, 2, 6 ) );
    }

    TEST( getDeviceAssignmentMapping_3GPUs ) {
        const MonteRayParallelAssistant& PA( MonteRayParallelAssistant::getInstance() );

        CHECK_EQUAL(0, PA.calcDeviceID( 7, 3, 0 ) );
        CHECK_EQUAL(0, PA.calcDeviceID( 7, 3, 1 ) );
        CHECK_EQUAL(0, PA.calcDeviceID( 7, 3, 2 ) );
        CHECK_EQUAL(1, PA.calcDeviceID( 7, 3, 3 ) );
        CHECK_EQUAL(1, PA.calcDeviceID( 7, 3, 4 ) );
        CHECK_EQUAL(2, PA.calcDeviceID( 7, 3, 5 ) );
        CHECK_EQUAL(2, PA.calcDeviceID( 7, 3, 6 ) );
    }

    TEST( getDeviceAssignmentMapping_20cores_2GPUs ) {
        const MonteRayParallelAssistant& PA( MonteRayParallelAssistant::getInstance() );
        int nGPUS = 2;
        int nCores = 20;

        std::vector<int> ids;

        for( unsigned rank=0; rank<nCores; ++rank){
            ids.push_back( PA.calcDeviceID( nCores, nGPUS, rank ) );
        }

        std::vector<int> expected;
        for( unsigned rank=0; rank<10; ++rank){
            expected.push_back(0);
        }
        for( unsigned rank=0; rank<10; ++rank){
            expected.push_back(1);
        }

        CHECK_ARRAY_EQUAL( expected, ids, 20 );
    }

    TEST( getDeviceAssignmentMapping_21cores_3GPUs ) {
        const MonteRayParallelAssistant& PA( MonteRayParallelAssistant::getInstance() );
        int nGPUS = 3;
        int nCores = 22;

        std::vector<int> ids;

        for( unsigned rank=0; rank<nCores; ++rank){
            ids.push_back( PA.calcDeviceID( nCores, nGPUS, rank ) );
        }

        std::vector<int> expected;
        for( unsigned rank=0; rank<8; ++rank){
            expected.push_back(0);
        }
        for( unsigned rank=0; rank<7; ++rank){
            expected.push_back(1);
        }
        for( unsigned rank=0; rank<7; ++rank){
            expected.push_back(2);
        }

        CHECK_ARRAY_EQUAL( expected, ids, nCores );
    }

    TEST( getDevice ) {
        const MonteRayParallelAssistant& PA( MonteRayParallelAssistant::getInstance() );
        CHECK_EQUAL( 0, PA.getDeviceID() );
    }

}

} // end namespace
