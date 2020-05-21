#include <UnitTest++.h>

#include <iostream>

#include "MonteRayDefinitions.hh"
#include "GPUUtilityFunctions.hh"

#include "RayListInterface.hh"
#include "RayListInterface_test_helper.hh"

SUITE( RayListInterface_simple_tests ) {
    using namespace MonteRay;

    typedef MonteRay::ParticleRay_t ParticleRay_t;

    TEST( test_setup ) {
        //gpuCheck();
        //		CHECK(false);
    }
    TEST( setup_host ) {
        RayListInterface<1> points(10);
        CHECK_EQUAL(10, points.capacity() );
        CHECK_EQUAL(0, points.size() );
    }

    TEST( add_two_particle_via_components ) {
        RayListInterface<1> points(2);

        points.add( 1.0, 2.0, 3.0,
                    4.0, 5.0, 6.0,
                    4.0, 0.9, 100.0,
                    99, 199, 0 );
        points.add( 1.0, 2.0, 3.0,
                    4.0, 5.0, 6.0,
                    14.0, 0.9, 100.0,
                    99, 199, 0 );
        CHECK_EQUAL(2, points.size() );
    }

    TEST( add_two_particle_via_particle ) {
        RayListInterface<1> points(2);
        ParticleRay_t particle;
        particle.pos[0] = 1.0;
        particle.pos[1] = 2.0;
        particle.pos[2] = 3.0;
        particle.dir[0] = 4.0;
        particle.dir[1] = 5.0;
        particle.dir[2] = 6.0;
        particle.energy[0] = 7.0;
        particle.weight[0] = 8.0;
        particle.index = 9;
        particle.detectorIndex = 99;
        particle.particleType = 0;
        particle.time = 11.0;

        points.add( particle );
        CHECK_EQUAL(1, points.size() );
        particle.index = 19;
        points.add( particle );
        CHECK_EQUAL(2, points.size() );

        ParticleRay_t particle2 = points.getParticle(0);
        CHECK_CLOSE(1.0, particle2.pos[0], 1e-5 );
        CHECK_EQUAL(9, particle2.index );

        particle2 = points.getParticle(1);
        CHECK_EQUAL(19, particle2.index );

    }

    TEST( add_two_particles_via_array ) {
        RayListInterface<1> points(2);
        ParticleRay_t particle[2];
        particle[0].pos[0] = 1.0;
        particle[0].pos[1] = 2.0;
        particle[0].pos[2] = 3.0;
        particle[0].dir[0] = 4.0;
        particle[0].dir[1] = 5.0;
        particle[0].dir[2] = 6.0;
        particle[0].energy[0] = 7.0;
        particle[0].weight[0] = 8.0;
        particle[0].time = 10.0;
        particle[0].index = 9;
        particle[0].detectorIndex = 99;
        particle[0].particleType = 0;

        particle[1].pos[0] = 11.0;
        particle[1].pos[1] = 12.0;
        particle[1].pos[2] = 13.0;
        particle[1].dir[0] = 14.0;
        particle[1].dir[1] = 15.0;
        particle[1].dir[2] = 16.0;
        particle[1].energy[0] = 17.0;
        particle[1].weight[0] = 18.0;
        particle[1].time = 20.0;
        particle[1].index = 19;
        particle[1].detectorIndex = 99;
        particle[1].particleType = 0;

        points.add( particle, 2 );
        CHECK_EQUAL(2, points.size() );

        ParticleRay_t particle2 = points.getParticle(0);
        CHECK_CLOSE(1.0, particle2.pos[0], 1e-5 );
        CHECK_CLOSE(10.0, particle2.time, 1e-5 );
        CHECK_EQUAL(9, particle2.index );

        particle2 = points.getParticle(1);
        CHECK_EQUAL(19, particle2.index );
        CHECK_CLOSE(20.0, particle2.time, 1e-5 );

    }

    TEST( add_two_particles_via_voidPtr ) {
        RayListInterface<1> points(2);
        ParticleRay_t particle[2];
        particle[0].pos[0] = 1.0;
        particle[0].pos[1] = 2.0;
        particle[0].pos[2] = 3.0;
        particle[0].dir[0] = 4.0;
        particle[0].dir[1] = 5.0;
        particle[0].dir[2] = 6.0;
        particle[0].energy[0] = 7.0;
        particle[0].weight[0] = 8.0;
        particle[0].time = 10.0;
        particle[0].index = 9;
        particle[0].detectorIndex = 99;
        particle[0].particleType = 0;

        particle[1].pos[0] = 11.0;
        particle[1].pos[1] = 12.0;
        particle[1].pos[2] = 13.0;
        particle[1].dir[0] = 14.0;
        particle[1].dir[1] = 15.0;
        particle[1].dir[2] = 16.0;
        particle[1].energy[0] = 17.0;
        particle[1].weight[0] = 18.0;
        particle[1].time = 10.0;
        particle[1].index = 19;
        particle[1].detectorIndex = 99;
        particle[1].particleType = 0;

        void* voidPtr = static_cast<void*>( particle );
        points.add( voidPtr, 2 );
        CHECK_EQUAL(2, points.size() );

        ParticleRay_t particle2 = points.getParticle(0);
        CHECK_CLOSE(1.0, particle2.pos[0], 1e-5 );
        CHECK_EQUAL(9, particle2.index );
        particle2 = points.getParticle(1);
        CHECK_EQUAL(19, particle2.index );

    }

    TEST( send_to_gpu_only) {
        std::cout << "Debug: RayListInterface_tester -- send_to_gpu_only \n";
        MonteRay::gpuInfo();
        RayListInterfaceTester<1> tester;

        RayListInterface<1> points(2);

        points.add( 1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                14.0, 0.9, 100.0,
                99, 199, 0);
        points.add( 1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                14.0, 0.9, 100.0,
                99, 199, 0);
        CHECK_EQUAL(2, points.size() );
        CHECK_EQUAL(2, points.capacity() );

        CHECK_EQUAL( false, points.isCudaCopyMade() );
    }

    TEST( send_to_gpu_getCapacity) {
        std::cout << "Debug: RayListInterface_tester -- send_to_gpu_getCapacity \n";
        MonteRay::gpuInfo();
        RayListInterfaceTester<1> tester;

        RayListInterface<1> points(2);

        points.add( 1.0, 2.0, 3.0,
                    4.0, 5.0, 6.0,
                    14.0, 0.9, 100.0,
                    99, 199, 0);
        points.add( 1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                14.0, 0.9, 100.0,
                99, 199, 0);
        CHECK_EQUAL(2, points.size() );
        CHECK_EQUAL(2, points.capacity() );

        CHECK_EQUAL( false, points.isCudaCopyMade() );
        tester.setupTimers();
        RayListSize_t result = tester.launchGetCapacity(1,1,points);
        tester.stopTimers();
        CHECK_EQUAL( 2, unsigned(result) );
    }

    TEST( send_to_gpu_getTotalEnergy) {
        RayListInterfaceTester<1> tester;

        RayListInterface<1> points(2);

        points.add( 1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                14.0, 0.9, 100.0,
                99, 199, 0);
        points.add( 1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                24.0, 0.9, 100.0,
                99, 199, 0);
        CHECK_EQUAL(2, points.size() );

        tester.setupTimers();
        gpuFloatType_t result = tester.launchTestSumEnergy(1,1,points);
        tester.stopTimers();
        CHECK_CLOSE( 38.0, float(result), 1e-7 );
    }

    TEST( write_And_read_a_particle ) {
        RayListInterface<1> collisions(1);

        std::string filename = "writeAndReadAParticleTest.bin";
        collisions.openOutput( filename );

        ParticleRay_t particle;
        particle.pos[0] = 1.0;
        particle.pos[1] = 2.0;
        particle.pos[2] = 3.0;
        particle.dir[0] = 4.0;
        particle.dir[1] = 5.0;
        particle.dir[2] = 6.0;
        particle.energy[0] = 14.0;
        particle.weight[0] = 15.0;
        particle.time = 20.0;
        particle.index = 16;
        particle.detectorIndex = 17;
        particle.particleType = 18;

        collisions.writeParticle( particle );
        CHECK_EQUAL(1, collisions.getNumCollisionsOnFile() );
        collisions.closeOutput();

        RayListInterface<1> collisions2(1);
        collisions2.openInput( filename );

        CHECK_EQUAL(1, collisions2.getVersion() );
        CHECK_EQUAL(1, collisions2.getNumCollisionsOnFile() );

        ParticleRay_t particle2;
        particle2 = collisions2.readParticle();
        CHECK_CLOSE( 1.0, particle2.pos[0], 1e-7 );
        CHECK_CLOSE( 2.0, particle2.pos[1], 1e-7 );
        CHECK_CLOSE( 3.0, particle2.pos[2], 1e-7 );
        CHECK_CLOSE( 4.0, particle2.dir[0], 1e-7 );
        CHECK_CLOSE( 5.0, particle2.dir[1], 1e-7 );
        CHECK_CLOSE( 6.0, particle2.dir[2], 1e-7 );
        CHECK_CLOSE( 14.0, particle2.energy[0], 1e-7 );
        CHECK_CLOSE( 15.0, particle2.weight[0], 1e-7 );
        CHECK_CLOSE( 20.0, particle2.time, 1e-7 );
        CHECK_EQUAL( 16, particle2.index );
        CHECK_EQUAL( 17, particle2.detectorIndex );
        CHECK_EQUAL( 18, particle2.particleType );
        collisions2.closeInput();
    }

    TEST( read_mcatk_written_test_collision_file ) {
        std::cout << "Debug: RayListInterface_tester -- read_mcatk_written_test_collision_file \n";
        std::cout << "Debug:    reading MonteRayTestFiles/MCATKWriteParticleRayListTest.bin\n";
        RayListInterfaceTester<1> tester;

        std::string filename = "MonteRayTestFiles/MCATKWriteParticleRayListTest.bin";
        RayListInterface<1> points(1);

        points.openInput( filename );
        CHECK_EQUAL(0, points.getVersion() );
        CHECK_EQUAL(1, points.getNumCollisionsOnFile() );

        ParticleRay_t particle;
        particle = points.readParticle();

        CHECK_CLOSE(1.0, particle.getPosition()[0], 1e-5);
        CHECK_CLOSE(2.0, particle.getPosition()[1], 1e-5);
        CHECK_CLOSE(3.0, particle.getPosition()[2], 1e-5);
        CHECK_CLOSE(4.0, particle.getDirection()[0], 1e-5);
        CHECK_CLOSE(5.0, particle.getDirection()[1], 1e-5);
        CHECK_CLOSE(6.0, particle.getDirection()[2], 1e-5);
        CHECK_CLOSE(14.0, particle.getEnergy(), 1e-5);
        CHECK_CLOSE(15.0, particle.getWeight(), 1e-5);
        CHECK_EQUAL(16, particle.getIndex() );
        CHECK_EQUAL(17, particle.getDetectorIndex() );
        CHECK_EQUAL(18, particle.getParticleType() );
    }

    TEST( big_collision_file_read ) {
        std::cout << "Debug: RayListInterface_tester -- big_collision_file_read \n";
        std::cout << "Debug:    reading MonteRayTestFiles/collisionsGodivaRCart100x100x100InWater_2568016Rays.bin \n";
        RayListInterfaceTester<1> tester;

        RayListInterface<1> points(2);
        points.readToMemory( "MonteRayTestFiles/collisionsGodivaRCart100x100x100InWater_2568016Rays.bin"  );
        CHECK_EQUAL(2568016, points.size() );

        tester.setupTimers();
        RayListSize_t result = tester.launchGetCapacity(1,1,points);
        tester.stopTimers();
        CHECK_EQUAL( 2568016, unsigned(result) );

        // x =  5.09468442; y= 3.39810971; z=-0.75147645; u=0.73015572; v=-0.62438205; w=-0.27752420; energy=4.44875086e+00; weight= 0.13102725; index=485557;

        CHECK_CLOSE(5.09468, points.getX(0), 1e-5);
        CHECK_CLOSE(3.39811, points.getY(0), 1e-5);
        CHECK_CLOSE(-0.751476, points.getZ(0), 1e-5);
        CHECK_CLOSE(0.73015572, points.getU(0), 1e-5);
        CHECK_CLOSE(-0.62438205, points.getV(0), 1e-5);
        CHECK_CLOSE(-0.27752420, points.getW(0), 1e-5);
        CHECK_CLOSE(4.44875, points.getEnergy(0), 1e-5);
        CHECK_CLOSE(0.13102725, points.getWeight(0), 1e-5);
        CHECK_EQUAL(485557, points.getIndex(0) );

        // x =  5.09468442; y= 3.39810971; z=-0.75147645; u=0.24749477; v=-0.46621658; w=-0.84934589; energy=4.46686655e+00; weight= 0.13102725; index=485557;

        CHECK_CLOSE(5.09468, points.getX(1), 1e-5);
        CHECK_CLOSE(3.39811, points.getY(1), 1e-5);
        CHECK_CLOSE(-0.751476, points.getZ(1), 1e-5);
        CHECK_CLOSE(0.24749477, points.getU(1), 1e-5);
        CHECK_CLOSE(-0.46621658, points.getV(1), 1e-5);
        CHECK_CLOSE(-0.84934589, points.getW(1), 1e-5);
        CHECK_CLOSE(4.46686655, points.getEnergy(1), 1e-5);
        CHECK_CLOSE(0.13102725, points.getWeight(1), 1e-5);
        CHECK_EQUAL(485557, points.getIndex(1) );

        //  x = -4.03245961; y=-2.47439122; z=-13.36526208; u=0.02893844; v= 0.51430749; w=-0.85711748; energy=3.00844045e-08; weight= 0.13102725; index=304643;

        CHECK_CLOSE(-4.03245961, points.getX(1000), 1e-5);
        CHECK_CLOSE(-2.47439122, points.getY(1000), 1e-5);
        CHECK_CLOSE(-13.36526208, points.getZ(1000), 1e-5);
        CHECK_CLOSE(0.02893844, points.getU(1000), 1e-5);
        CHECK_CLOSE(0.51430749, points.getV(1000), 1e-5);
        CHECK_CLOSE(-0.85711748, points.getW(1000), 1e-5);
        CHECK_CLOSE(3.00844045e-08, points.getEnergy(1000), 1e-5);
        CHECK_CLOSE(0.13102725, points.getWeight(1000), 1e-5);
        CHECK_EQUAL(304643, points.getIndex(1000) );

    }

    TEST( read_next_event_estimator_ray_file ) {
        std::cout << "Debug: RayListInterface_tester -- read_next_event_estimator_ray_file \n";

        std::cout << "Debug:    reading MonteRayTestFiles/U-04p_slab_single_source_ray_collisionFile.bin \n";
        RayListInterfaceTester<3> tester;

        RayListInterface<3> points(2);
        points.readToMemory( "MonteRayTestFiles/U-04p_slab_single_source_ray_collisionFile.bin"  );
        CHECK_EQUAL(1, points.size() );

        tester.setupTimers();
        RayListSize_t result = tester.launchGetCapacity(1,1,points);
        tester.stopTimers();
        CHECK_EQUAL( 1, unsigned(result) );

        CHECK_CLOSE(-0.001, points.getX(0), 1e-5);
        CHECK_CLOSE(0.0, points.getY(0), 1e-5);
        CHECK_CLOSE(0.0, points.getZ(0), 1e-5);
        CHECK_CLOSE(1.0, points.getU(0), 1e-5);
        CHECK_CLOSE(0.0, points.getV(0), 1e-5);
        CHECK_CLOSE(0.0, points.getW(0), 1e-5);
        CHECK_CLOSE(1.0, points.getEnergy(0,0), 1e-5);
        CHECK_CLOSE(0.5, points.getWeight(0,0), 1e-5);
        CHECK_CLOSE(0.0, points.getEnergy(0,1), 1e-5);
        CHECK_CLOSE(0.0, points.getWeight(0,1), 1e-5);
        CHECK_CLOSE(0.0, points.getEnergy(0,2), 1e-5);
        CHECK_CLOSE(0.0, points.getWeight(0,2), 1e-5);
        CHECK_EQUAL(0, points.getIndex(0) );
        CHECK_EQUAL(0, points.getDetectorIndex(0) );
        CHECK_EQUAL(1, points.getParticleType(0) );
    }
}

SUITE( RayListInterface_bank_tests ) {
    using namespace MonteRay;

    TEST( readToBank ) {
        RayListInterface<1> bank(1000);
        unsigned offset=0;
        bool end = bank.readToBank( "MonteRayTestFiles/collisionsGodivaRCart100x100x100InWater_2568016Rays.bin", offset );

        CHECK_CLOSE(5.09468, bank.getX(0), 1e-5);
        CHECK_CLOSE(3.39811, bank.getY(0), 1e-5);
        CHECK_CLOSE(-0.751476, bank.getZ(0), 1e-5);
        CHECK_CLOSE(0.73015572, bank.getU(0), 1e-5);
        CHECK_CLOSE(-0.62438205, bank.getV(0), 1e-5);
        CHECK_CLOSE(-0.27752420, bank.getW(0), 1e-5);
        CHECK_CLOSE(4.44875, bank.getEnergy(0), 1e-5);
        CHECK_CLOSE(0.13102725, bank.getWeight(0), 1e-5);
        CHECK_EQUAL(485557, bank.getIndex(0) );

        CHECK_CLOSE(5.09468, bank.getX(1), 1e-5);
        CHECK_CLOSE(3.39811, bank.getY(1), 1e-5);
        CHECK_CLOSE(-0.751476, bank.getZ(1), 1e-5);
        CHECK_CLOSE(0.24749477, bank.getU(1), 1e-5);
        CHECK_CLOSE(-0.46621658, bank.getV(1), 1e-5);
        CHECK_CLOSE(-0.84934589, bank.getW(1), 1e-5);
        CHECK_CLOSE(4.46686655, bank.getEnergy(1), 1e-5);
        CHECK_CLOSE(0.13102725, bank.getWeight(1), 1e-5);
        CHECK_EQUAL(485557, bank.getIndex(1) );

        CHECK_EQUAL( false, end);

        offset += bank.capacity();
        CHECK_EQUAL( 1000, offset );
        end = bank.readToBank( "MonteRayTestFiles/collisionsGodivaRCart100x100x100InWater_2568016Rays.bin", offset );

        CHECK_CLOSE(-4.03245961, bank.getX(0), 1e-5);
        CHECK_CLOSE(-2.47439122, bank.getY(0), 1e-5);
        CHECK_CLOSE(-13.36526208, bank.getZ(0), 1e-5);
        CHECK_CLOSE(0.02893844, bank.getU(0), 1e-5);
        CHECK_CLOSE(0.51430749, bank.getV(0), 1e-5);
        CHECK_CLOSE(-0.85711748, bank.getW(0), 1e-5);
        CHECK_CLOSE(3.00844045e-08, bank.getEnergy(0), 1e-5);
        CHECK_CLOSE(0.13102725, bank.getWeight(0), 1e-5);
        CHECK_EQUAL(304643, bank.getIndex(0) );

        CHECK_EQUAL( false, end);
    }

    TEST( nicely_read_end_of_bank ) {
        RayListInterface<1> bank(1000);
        unsigned offset=0;
        bool end = bank.readToBank( "MonteRayTestFiles/collisionsGodivaRCart100x100x100InWater_2568016Rays.bin", offset );
        CHECK_EQUAL( false, end);

        offset = 2568016 - 500;
        end = bank.readToBank( "MonteRayTestFiles/collisionsGodivaRCart100x100x100InWater_2568016Rays.bin", offset );
        CHECK_EQUAL(500, bank.size());
        CHECK_EQUAL( true, end);
    }

    TEST( read_collisions_to_bank_in_a_loop ) {
        RayListInterface<1> bank(1000);
        unsigned offset=0;

        bool end = false;
        while( ! end ) {
            end = bank.readToBank( "MonteRayTestFiles/collisionsGodivaRCart100x100x100InWater_2568016Rays.bin", offset );
            offset += bank.size();
        }

        CHECK_EQUAL(16, bank.size());
        CHECK_EQUAL(true, end);
    }

}


