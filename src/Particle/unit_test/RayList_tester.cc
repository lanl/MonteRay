#include <UnitTest++.h>

#include <iostream>

#include "MonteRayDefinitions.hh"
#include "GPUUtilityFunctions.hh"

#include "RayList.hh"

SUITE( RayList_simple_tests ) {

    using namespace MonteRay;

    TEST( ParticleRayList_default_ctor ) {
        ParticleRayList list;
        // std::cout << "Debug: running RayList_tester.cc -- ctor test\n";
        CHECK(true);
    }

    TEST( getN ) {
        RayList_t<> raylist;
        CHECK_EQUAL( 1, raylist.getN());
    }

    TEST( ParticleRayList_getN ) {
        ParticleRayList raylist;
        CHECK_EQUAL( 1, raylist.getN());
    }

    TEST( PointDetRayList_getN ) {
        PointDetRayList raylist;
        CHECK_EQUAL( 3, raylist.getN());
    }

    TEST( ParticleRayList_ctor_w_num ) {
        ParticleRayList list(10);
        CHECK_EQUAL( 10, list.capacity() );
        CHECK_EQUAL( 0, list.size() );
    }

    TEST( ParticleRayList_add_propertyGetters ) {
        ParticleRayList list(10);
        CHECK_EQUAL( 10, list.capacity() );
        CHECK_EQUAL( 0, list.size() );

        ParticleRayList::RAY_T ray;

        ray.pos[0] = 1.0;
        ray.pos[1] = 2.0;
        ray.pos[2] = 3.0;
        ray.dir[0] = 4.0;
        ray.dir[1] = 5.0;
        ray.dir[2] = 6.0;
        ray.energy[0] = 10.0;
        ray.weight[0] = 11.0;
        ray.index = 12;
        ray.detectorIndex = 13;
        ray.particleType = 14;
        ray.time = 15.0;
        list.add( ray );

        ray.pos[0] = 101.0;
        ray.pos[1] = 102.0;
        ray.pos[2] = 103.0;
        ray.dir[0] = 104.0;
        ray.dir[1] = 105.0;
        ray.dir[2] = 106.0;
        ray.energy[0] = 1010.0;
        ray.weight[0] = 1011.0;
        ray.index = 1012;
        ray.detectorIndex = 1013;
        ray.particleType = 1014;
        ray.time = 1015.0;
        list.add( ray );

        CHECK_EQUAL( 2, list.size() );

        CHECK_CLOSE( 1.0, list.getPosition(0)[0], 1e-11 );
        CHECK_CLOSE( 2.0, list.getPosition(0)[1], 1e-11 );
        CHECK_CLOSE( 3.0, list.getPosition(0)[2], 1e-11 );
        CHECK_CLOSE( 4.0, list.getDirection(0)[0], 1e-11 );
        CHECK_CLOSE( 5.0, list.getDirection(0)[1], 1e-11 );
        CHECK_CLOSE( 6.0, list.getDirection(0)[2], 1e-11 );
        CHECK_CLOSE( 10.0, list.getEnergy(0,0), 1e-11 );
        CHECK_CLOSE( 11.0, list.getWeight(0,0), 1e-11 );
        CHECK_EQUAL( 12.0, list.getIndex(0) );
        CHECK_EQUAL( 13.0, list.getDetectorIndex(0) );
        CHECK_EQUAL( 14.0, list.getParticleType(0) );
        CHECK_EQUAL( 15.0, list.getTime(0) );

        CHECK_CLOSE( 101.0, list.getPosition(1)[0], 1e-11 );
        CHECK_CLOSE( 102.0, list.getPosition(1)[1], 1e-11 );
        CHECK_CLOSE( 103.0, list.getPosition(1)[2], 1e-11 );
        CHECK_CLOSE( 104.0, list.getDirection(1)[0], 1e-11 );
        CHECK_CLOSE( 105.0, list.getDirection(1)[1], 1e-11 );
        CHECK_CLOSE( 106.0, list.getDirection(1)[2], 1e-11 );
        CHECK_CLOSE( 1010.0, list.getEnergy(1,0), 1e-11 );
        CHECK_CLOSE( 1011.0, list.getWeight(1,0), 1e-11 );
        CHECK_EQUAL( 1012.0, list.getIndex(1) );
        CHECK_EQUAL( 1013.0, list.getDetectorIndex(1) );
        CHECK_EQUAL( 1014.0, list.getParticleType(1) );
        CHECK_EQUAL( 1015.0, list.getTime(1) );

    }

    TEST( ParticleRayList_add_get ) {
        ParticleRayList list(10);
        CHECK_EQUAL( 10, list.capacity() );
        CHECK_EQUAL( 0, list.size() );

        ParticleRayList::RAY_T ray;
        ray.pos[0] = 1.0;
        ray.pos[1] = 2.0;
        ray.pos[2] = 3.0;
        ray.dir[0] = 4.0;
        ray.dir[1] = 5.0;
        ray.dir[2] = 6.0;
        ray.energy[0] = 10.0;
        ray.weight[0] = 11.0;
        ray.index = 12;
        ray.detectorIndex = 13;
        ray.particleType = 14;
        ray.time = 15.0;
        list.add( ray );
        CHECK_EQUAL( 1, list.size() );

        ParticleRayList::RAY_T rayCopy;
        rayCopy = list.getParticle( 0 );
        CHECK_CLOSE( 1.0, rayCopy.getPosition()[0], 1e-11 );
        CHECK_CLOSE( 2.0, rayCopy.getPosition()[1], 1e-11 );
        CHECK_CLOSE( 3.0, rayCopy.getPosition()[2], 1e-11 );
        CHECK_CLOSE( 4.0, rayCopy.getDirection()[0], 1e-11 );
        CHECK_CLOSE( 5.0, rayCopy.getDirection()[1], 1e-11 );
        CHECK_CLOSE( 6.0, rayCopy.getDirection()[2], 1e-11 );
        CHECK_CLOSE( 10.0, rayCopy.getEnergy(0), 1e-11 );
        CHECK_CLOSE( 11.0, rayCopy.getWeight(0), 1e-11 );
        CHECK_EQUAL( 12.0, rayCopy.getIndex() );
        CHECK_EQUAL( 13.0, rayCopy.getDetectorIndex() );
        CHECK_EQUAL( 14.0, rayCopy.getParticleType() );
        CHECK_EQUAL( 15.0, rayCopy.getTime() );
    }

    TEST( ParticleRayList_add_pop ) {
        ParticleRayList list(10);
        CHECK_EQUAL( 10, list.capacity() );
        CHECK_EQUAL( 0, list.size() );

        ParticleRayList::RAY_T ray;
        ray.pos[0] = 1.0;
        ray.pos[1] = 2.0;
        ray.pos[2] = 3.0;
        ray.dir[0] = 4.0;
        ray.dir[1] = 5.0;
        ray.dir[2] = 6.0;
        ray.energy[0] = 10.0;
        ray.weight[0] = 11.0;
        ray.index = 12;
        ray.detectorIndex = 13;
        ray.particleType = 14;
        ray.time = 15.0;
        list.add( ray );
        CHECK_EQUAL( 1, list.size() );

        ParticleRayList::RAY_T rayCopy;
        rayCopy = list.pop();
        CHECK_EQUAL( 0, list.size() );

        CHECK_CLOSE( 1.0, rayCopy.getPosition()[0], 1e-11 );
        CHECK_CLOSE( 2.0, rayCopy.getPosition()[1], 1e-11 );
        CHECK_CLOSE( 3.0, rayCopy.getPosition()[2], 1e-11 );
        CHECK_CLOSE( 4.0, rayCopy.getDirection()[0], 1e-11 );
        CHECK_CLOSE( 5.0, rayCopy.getDirection()[1], 1e-11 );
        CHECK_CLOSE( 6.0, rayCopy.getDirection()[2], 1e-11 );
        CHECK_CLOSE( 10.0, rayCopy.getEnergy(0), 1e-11 );
        CHECK_CLOSE( 11.0, rayCopy.getWeight(0), 1e-11 );
        CHECK_EQUAL( 12.0, rayCopy.getIndex() );
        CHECK_EQUAL( 13.0, rayCopy.getDetectorIndex() );
        CHECK_EQUAL( 14.0, rayCopy.getParticleType() );
        CHECK_EQUAL( 15.0, rayCopy.getTime() );
    }

    TEST( ParticleRayList_copy_ctor ) {
        ParticleRayList oldlist(10);
        CHECK_EQUAL( 10, oldlist.capacity() );
        CHECK_EQUAL( 0, oldlist.size() );

        ParticleRayList::RAY_T ray;

        ray.pos[0] = 1.0;
        ray.pos[1] = 2.0;
        ray.pos[2] = 3.0;
        ray.dir[0] = 4.0;
        ray.dir[1] = 5.0;
        ray.dir[2] = 6.0;
        ray.energy[0] = 10.0;
        ray.weight[0] = 11.0;
        ray.index = 12;
        ray.detectorIndex = 13;
        ray.particleType = 14;
        ray.time = 15.0;
        oldlist.add( ray );

        ray.pos[0] = 101.0;
        ray.pos[1] = 102.0;
        ray.pos[2] = 103.0;
        ray.dir[0] = 104.0;
        ray.dir[1] = 105.0;
        ray.dir[2] = 106.0;
        ray.energy[0] = 1010.0;
        ray.weight[0] = 1011.0;
        ray.index = 1012;
        ray.detectorIndex = 1013;
        ray.particleType = 1014;
        ray.time =1015.0;
        oldlist.add( ray );

        ParticleRayList list(oldlist);
        CHECK_EQUAL( 2, list.size() );

        CHECK_CLOSE( 1.0, list.getPosition(0)[0], 1e-11 );
        CHECK_CLOSE( 2.0, list.getPosition(0)[1], 1e-11 );
        CHECK_CLOSE( 3.0, list.getPosition(0)[2], 1e-11 );
        CHECK_CLOSE( 4.0, list.getDirection(0)[0], 1e-11 );
        CHECK_CLOSE( 5.0, list.getDirection(0)[1], 1e-11 );
        CHECK_CLOSE( 6.0, list.getDirection(0)[2], 1e-11 );
        CHECK_CLOSE( 10.0, list.getEnergy(0,0), 1e-11 );
        CHECK_CLOSE( 11.0, list.getWeight(0,0), 1e-11 );
        CHECK_EQUAL( 12.0, list.getIndex(0) );
        CHECK_EQUAL( 13.0, list.getDetectorIndex(0) );
        CHECK_EQUAL( 14.0, list.getParticleType(0) );
        CHECK_EQUAL( 15.0, list.getTime(0) );

        CHECK_CLOSE( 101.0, list.getPosition(1)[0], 1e-11 );
        CHECK_CLOSE( 102.0, list.getPosition(1)[1], 1e-11 );
        CHECK_CLOSE( 103.0, list.getPosition(1)[2], 1e-11 );
        CHECK_CLOSE( 104.0, list.getDirection(1)[0], 1e-11 );
        CHECK_CLOSE( 105.0, list.getDirection(1)[1], 1e-11 );
        CHECK_CLOSE( 106.0, list.getDirection(1)[2], 1e-11 );
        CHECK_CLOSE( 1010.0, list.getEnergy(1,0), 1e-11 );
        CHECK_CLOSE( 1011.0, list.getWeight(1,0), 1e-11 );
        CHECK_EQUAL( 1012.0, list.getIndex(1) );
        CHECK_EQUAL( 1013.0, list.getDetectorIndex(1) );
        CHECK_EQUAL( 1014.0, list.getParticleType(1) );
        CHECK_EQUAL( 1015.0, list.getTime(1) );

    }
}


