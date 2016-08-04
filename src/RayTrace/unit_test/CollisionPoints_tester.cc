#include <UnitTest++.h>

#include <iostream>

#include "CollisionPoints.h"
#include "CollisionPoints_test_helper.hh"
#include "gpuGlobal.h"

SUITE( CollisionPoints_simple_tests ) {
	TEST( test_setup ) {
		gpuCheck();
	}
    TEST( setup_host ) {
    	CollisionPointsHost points(10);
    	CHECK_EQUAL(10, points.capacity() );
    	CHECK_EQUAL(0, points.size() );
    }

    TEST( send_to_gpu_getCapacity) {
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
     	CollisionPointsTester tester;

     	CollisionPointsHost points(2);
     	points.readToMemory( "/usr/projects/mcatk/user/jsweezy/link_files/collisionsGodivaCylInWater1.bin"  );
     	CHECK_EQUAL(285735, points.size() );

     	points.CopyToGPU();
     	tester.setupTimers();
     	CollisionPointsSize_t result = tester.launchGetCapacity(1,1,points);
     	tester.stopTimers();
     	CHECK_EQUAL( 285735, unsigned(result) );

     	CHECK_CLOSE(0.948374, points.getEnergy(0), 1e-6);
     	CHECK_CLOSE(-3.05307, points.getX(0), 1e-5);
     	CHECK_CLOSE(5.3902, points.getY(0), 1e-4);
     	CHECK_CLOSE(9.21146, points.getZ(0), 1e-5);

     	CHECK_CLOSE(2.12997e-08, points.getEnergy(1000), 1e-13);
     	CHECK_CLOSE(-8.08183, points.getX(1000), 1e-5);
     	CHECK_CLOSE(-6.36573, points.getY(1000), 1e-5);
     	CHECK_CLOSE(4.8127, points.getZ(1000), 1e-4);
    }
}

SUITE( CollisionPoints_bank_tests ) {
	TEST( readToBank ) {
		CollisionPointsHost bank(1000);
		unsigned offset=0;
		bool end = bank.readToBank( "/usr/projects/mcatk/user/jsweezy/link_files/collisionsGodivaCylInWater1.bin", offset );
     	CHECK_CLOSE(0.948374, bank.getEnergy(0), 1e-6);
     	CHECK_CLOSE(-3.05307, bank.getX(0), 1e-5);
     	CHECK_CLOSE(5.3902, bank.getY(0), 1e-4);
     	CHECK_CLOSE(9.21146, bank.getZ(0), 1e-5);
     	CHECK_EQUAL( false, end);


     	offset += bank.capacity();
     	end = bank.readToBank( "/usr/projects/mcatk/user/jsweezy/link_files/collisionsGodivaCylInWater1.bin", offset );
    	CHECK_CLOSE(2.12997e-08, bank.getEnergy(0), 1e-13);
     	CHECK_CLOSE(-8.08183, bank.getX(0), 1e-5);
     	CHECK_CLOSE(-6.36573, bank.getY(0), 1e-5);
     	CHECK_CLOSE(4.8127, bank.getZ(0), 1e-4);
     	CHECK_EQUAL( false, end);
	}

	TEST( nicely_read_end_of_bank ) {
		CollisionPointsHost bank(1000);
		unsigned offset=0;
		bool end = bank.readToBank( "/usr/projects/mcatk/user/jsweezy/link_files/collisionsGodivaCylInWater1.bin", offset );
     	CHECK_CLOSE(0.948374, bank.getEnergy(0), 1e-6);
     	CHECK_CLOSE(-3.05307, bank.getX(0), 1e-5);
     	CHECK_CLOSE(5.3902, bank.getY(0), 1e-4);
     	CHECK_CLOSE(9.21146, bank.getZ(0), 1e-5);
     	CHECK_EQUAL( false, end);

     	offset = 285735 - 500;
     	end = bank.readToBank( "/usr/projects/mcatk/user/jsweezy/link_files/collisionsGodivaCylInWater1.bin", offset );
     	CHECK_EQUAL(500, bank.size());
     	CHECK_EQUAL( true, end);
	}

	TEST( read_collisions_to_bank_in_a_loop ) {
		CollisionPointsHost bank(1000);
		unsigned offset=0;

		bool end = false;
		while( ! end ) {
			end = bank.readToBank( "/usr/projects/mcatk/user/jsweezy/link_files/collisionsGodivaCylInWater1.bin", offset );
			offset += bank.size();
		}

     	CHECK_EQUAL(735, bank.size());
     	CHECK_EQUAL(true, end);
	}


}


