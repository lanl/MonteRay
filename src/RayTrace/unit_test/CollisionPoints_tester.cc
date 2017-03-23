#include <UnitTest++.h>

#include <iostream>

#include "MonteRayDefinitions.hh"
#include "GPUUtilityFunctions.hh"

#include "CollisionPoints.h"
#include "CollisionPoints_test_helper.hh"

SUITE( CollisionPoints_simple_tests ) {
	TEST( test_setup ) {
		gpuCheck();
	}
    TEST( setup_host ) {
    	CollisionPointsHost points(10);
    	CHECK_EQUAL(10, points.capacity() );
    	CHECK_EQUAL(0, points.size() );
    }

    TEST( add_two_particle_via_components ) {
    	CollisionPointsHost points(2);

    	points.add( 1.0, 2.0, 3.0,
    			4.0, 5.0, 6.0,
    			14.0, 0.9, 99);
    	points.add( 1.0, 2.0, 3.0,
    			4.0, 5.0, 6.0,
    			14.0, 0.9, 99);
    	CHECK_EQUAL(2, points.size() );
    }

    TEST( add_two_particle_via_particle ) {
    	CollisionPointsHost points(2);
    	gpuParticle_t particle;
    	particle.pos[0] = 1.0;
    	particle.pos[1] = 2.0;
    	particle.pos[2] = 3.0;
    	particle.dir[0] = 4.0;
    	particle.dir[1] = 5.0;
    	particle.dir[2] = 6.0;
    	particle.energy = 7.0;
    	particle.weight = 8.0;
    	particle.index = 9;

    	points.add( particle );
    	CHECK_EQUAL(1, points.size() );
    	particle.index = 19;
    	points.add( particle );
    	CHECK_EQUAL(2, points.size() );

    	gpuParticle_t particle2 = points.getParticle(0);
    	CHECK_EQUAL(9, particle2.index );
    	particle2 = points.getParticle(1);
    	CHECK_EQUAL(19, particle2.index );

    }

    TEST( add_two_particles_via_array ) {
      	CollisionPointsHost points(2);
      	gpuParticle_t particle[2];
      	particle[0].pos[0] = 1.0;
      	particle[0].pos[1] = 2.0;
      	particle[0].pos[2] = 3.0;
      	particle[0].dir[0] = 4.0;
      	particle[0].dir[1] = 5.0;
      	particle[0].dir[2] = 6.0;
      	particle[0].energy = 7.0;
      	particle[0].weight = 8.0;
      	particle[0].index = 9;

      	particle[1].pos[0] = 11.0;
      	particle[1].pos[1] = 12.0;
      	particle[1].pos[2] = 13.0;
      	particle[1].dir[0] = 14.0;
      	particle[1].dir[1] = 15.0;
      	particle[1].dir[2] = 16.0;
      	particle[1].energy = 17.0;
      	particle[1].weight = 18.0;
      	particle[1].index = 19;

      	points.add( particle, 2 );
      	CHECK_EQUAL(2, points.size() );

      	gpuParticle_t particle2 = points.getParticle(0);
      	CHECK_EQUAL(9, particle2.index );
      	particle2 = points.getParticle(1);
      	CHECK_EQUAL(19, particle2.index );

      }

    TEST( add_two_particles_via_voidPtr ) {
       	CollisionPointsHost points(2);
       	gpuParticle_t particle[2];
       	particle[0].pos[0] = 1.0;
       	particle[0].pos[1] = 2.0;
       	particle[0].pos[2] = 3.0;
       	particle[0].dir[0] = 4.0;
       	particle[0].dir[1] = 5.0;
       	particle[0].dir[2] = 6.0;
       	particle[0].energy = 7.0;
       	particle[0].weight = 8.0;
       	particle[0].index = 9;

       	particle[1].pos[0] = 11.0;
       	particle[1].pos[1] = 12.0;
       	particle[1].pos[2] = 13.0;
       	particle[1].dir[0] = 14.0;
       	particle[1].dir[1] = 15.0;
       	particle[1].dir[2] = 16.0;
       	particle[1].energy = 17.0;
       	particle[1].weight = 18.0;
       	particle[1].index = 19;

       	void* voidPtr = static_cast<void*>( particle );
       	points.add( voidPtr, 2 );
       	CHECK_EQUAL(2, points.size() );

       	gpuParticle_t particle2 = points.getParticle(0);
       	CHECK_EQUAL(9, particle2.index );
       	particle2 = points.getParticle(1);
       	CHECK_EQUAL(19, particle2.index );

       }

    TEST( send_to_gpu_getCapacity) {
    	std::cout << "Debug: CollisionPoints_tester -- send_to_gpu_getCapacity \n";
    	MonteRay::gpuInfo();
    	CollisionPointsTester tester;

    	CollisionPointsHost points(2);

    	points.add( 1.0, 2.0, 3.0,
    			4.0, 5.0, 6.0,
    			14.0, 0.9, 99);
    	points.add( 1.0, 2.0, 3.0,
    			4.0, 5.0, 6.0,
    			14.0, 0.9, 99);
        CHECK_EQUAL(2, points.size() );
        CHECK_EQUAL(2, points.capacity() );

        CHECK_EQUAL( false, points.isCudaCopyMade() );
        points.copyToGPU();
        CHECK_EQUAL( true, points.isCudaCopyMade() );
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
    	std::cout << "Debug: CollisionPoints_tester -- read_file_getCapacity \n";
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


