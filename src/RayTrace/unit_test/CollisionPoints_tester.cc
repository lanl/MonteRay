#include <UnitTest++.h>

#include <iostream>

#include "CollisionPoints.h"
#include "CollisionPoints_test_helper.hh"
#include "gpuGlobal.h"

SUITE( CollisionPoints_simple_tests ) {
    TEST( setup_host ) {
    	cudaReset();
    	CollisionPointsHost points(10);
    	CHECK_EQUAL(10, points.capacity() );
    	CHECK_EQUAL(0, points.size() );
    }

    TEST( send_to_gpu_getCapacity) {
    	cudaReset();

    	CollisionPointsTester tester;

    	CollisionPointsHost points(2);

    	points.add( 1.0, 2.0, 3.0,
    			4.0, 5.0, 6.0,
    			14.0, 0.9, 99);
    	points.add( 1.0, 2.0, 3.0,
    			4.0, 5.0, 6.0,
    			14.0, 0.9, 99);
        CHECK_EQUAL(2, points.size() );

        points.CopyToGPU();
        tester.setupTimers();
        CollisionPointsSize_t result = tester.launchGetCapacity(1,1,points);
        tester.stopTimers();
        CHECK_EQUAL( 2, unsigned(result) );
    }

    TEST( send_to_gpu_getTotalEnergy) {
    	cudaReset();
    	CollisionPointsTester tester;

    	CollisionPointsHost points(2);

    	points.add( 1.0, 2.0, 3.0,
    			4.0, 5.0, 6.0,
    			14.0, 0.9, 99);
    	points.add( 1.0, 2.0, 3.0,
    			4.0, 5.0, 6.0,
    			24.0, 0.9, 99);
        CHECK_EQUAL(2, points.size() );

        points.CopyToGPU();
        tester.setupTimers();
        gpuFloatType_t result = tester.launchTestSumEnergy(1,1,points);
        tester.stopTimers();
        CHECK_CLOSE( 38.0, float(result), 1e-7 );
    }

    TEST( read_file_getCapacity ) {
    	cudaReset();
     	CollisionPointsTester tester;

     	CollisionPointsHost points(2);
     	points.readToMemory( "/usr/projects/mcatk/user/jsweezy/link_files/collisionsGodivaCylInWater1.bin"  );
     	CHECK_EQUAL(285735, points.size() );

     	points.CopyToGPU();
     	tester.setupTimers();
     	CollisionPointsSize_t result = tester.launchGetCapacity(1,1,points);
     	tester.stopTimers();
     	CHECK_EQUAL( 285735, unsigned(result) );
    }
}


