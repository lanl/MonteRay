#include <UnitTest++.h>

#include <iostream>
#include <string>

#include "MonteRayDefinitions.hh"
#include "GPUUtilityFunctions.hh"

#include "RayWorkInfo.hh"

SUITE( RayWorkInfo_tests ) {

    // data is not normally allocated on the CPU
    const bool allocateOnCPU = true;

    using namespace MonteRay;

    TEST( RayWorkInfo_ctor_0 ) {
        RayWorkInfo info(0,allocateOnCPU);
        CHECK_EQUAL( 1, info.capacity() );
        CHECK_EQUAL( 0, info.getRayCastSize(0) );
        CHECK_EQUAL( 0, info.getCrossingSize(0,0) );
        CHECK_EQUAL( 0, info.getCrossingSize(1,0) );
        CHECK_EQUAL( 0, info.getCrossingSize(2,0) );
        CHECK(true);
    }

    TEST( RayWorkInfo_ctor_w_num ) {
        RayWorkInfo info(10,allocateOnCPU);
        CHECK_EQUAL( 10, info.capacity() );
    }

    TEST( get_set_indices ) {
        RayWorkInfo info(3,allocateOnCPU);

        info.setIndex(0,0,11);
        info.setIndex(1,0,12);
        info.setIndex(2,0,13);
        info.setIndex(0,1,21);
        info.setIndex(1,1,22);
        info.setIndex(2,1,23);
        info.setIndex(0,2,31);
        info.setIndex(1,2,32);
        info.setIndex(2,2,33);

        CHECK_EQUAL( 11, info.getIndex(0,0) );
        CHECK_EQUAL( 12, info.getIndex(1,0) );
        CHECK_EQUAL( 13, info.getIndex(2,0) );
        CHECK_EQUAL( 21, info.getIndex(0,1) );
        CHECK_EQUAL( 22, info.getIndex(1,1) );
        CHECK_EQUAL( 23, info.getIndex(2,1) );
        CHECK_EQUAL( 31, info.getIndex(0,2) );
        CHECK_EQUAL( 32, info.getIndex(1,2) );
        CHECK_EQUAL( 33, info.getIndex(2,2) );
    }

    TEST( addRayCastCell ) {
         RayWorkInfo info(3,allocateOnCPU);
         CHECK_EQUAL( 0, info.getRayCastSize(0) );

         info.addRayCastCell( 0, 11, 21.0 );
         CHECK_EQUAL( 1, info.getRayCastSize(0) );
         info.addRayCastCell( 0, 12, 22.0 );
         CHECK_EQUAL( 2, info.getRayCastSize(0) );

         CHECK_EQUAL( 11, info.getRayCastCell(0,0) );
         CHECK_CLOSE( 21.0, info.getRayCastDist(0,0), 1e-5 );

         CHECK_EQUAL( 12, info.getRayCastCell(0,1) );
         CHECK_CLOSE( 22.0, info.getRayCastDist(0,1), 1e-5 );
     }

    TEST( addCrossingCell ) {
         RayWorkInfo info(3,allocateOnCPU);
         CHECK_EQUAL( 0, info.getCrossingSize(0,0) );
         CHECK_EQUAL( 0, info.getCrossingSize(1,0) );
         CHECK_EQUAL( 0, info.getCrossingSize(2,0) );

         unsigned dim = 0;
         unsigned pid = 0; // particle id
         info.addCrossingCell( dim, pid, 11, 21.0 );
         CHECK_EQUAL( 1, info.getCrossingSize(dim, pid) );
         info.addCrossingCell( dim, pid, 12, 22.0 );
         CHECK_EQUAL( 2, info.getCrossingSize(dim, pid) );

         CHECK_EQUAL( 11, info.getCrossingCell(dim, pid, 0) );
         CHECK_CLOSE( 21.0, info.getCrossingDist(dim, pid, 0), 1e-5 );

         CHECK_EQUAL( 12, info.getCrossingCell(dim, pid , 1) );
         CHECK_CLOSE( 22.0, info.getCrossingDist(dim, pid, 1), 1e-5 );
     }

    TEST( clear ) {
         RayWorkInfo info(3,allocateOnCPU);

         unsigned dim = 0;
         unsigned pid = 0; // particle id

         info.addRayCastCell( pid, 11, 21.0 );
         info.addRayCastCell( pid, 12, 22.0 );
         CHECK_EQUAL( 2, info.getRayCastSize(0) );

         dim = 0;
         info.addCrossingCell( dim, pid, 11, 21.0 );
         info.addCrossingCell( dim, pid, 12, 22.0 );
         CHECK_EQUAL( 2, info.getCrossingSize(dim, pid) );

         dim = 1;
         info.addCrossingCell( dim, pid, 11, 21.0 );
         info.addCrossingCell( dim, pid, 12, 22.0 );
         CHECK_EQUAL( 2, info.getCrossingSize(dim, pid) );

         dim = 2;
         info.addCrossingCell( dim, pid, 11, 21.0 );
         info.addCrossingCell( dim, pid, 12, 22.0 );
         CHECK_EQUAL( 2, info.getCrossingSize(dim, pid) );

         info.clear( pid );
         CHECK_EQUAL( 0, info.getRayCastSize(0) );
         CHECK_EQUAL( 0, info.getCrossingSize(0, pid) );
         CHECK_EQUAL( 0, info.getCrossingSize(1, pid) );
         CHECK_EQUAL( 0, info.getCrossingSize(2, pid) );
     }

}
