#include <UnitTest++.h>

#include <cmath>

#include "GPUUtilityFunctions.hh"

#include "MonteRayCrossSection.hh"
#include "MonteRayConstants.hh"

#include "MonteRayCrossSection_test_helper.hh"

SUITE( MonteRayCrossSection_tester ) {
    TEST( setup ) {
        //gpuCheck();
    }
    TEST( ctor ) {
        MonteRayCrossSectionHost xs(10);
        CHECK_EQUAL(10, xs.size() );
    }
    TEST_FIXTURE(MonteRayCrossSectionTestHelper, get_id ) {
        MonteRayCrossSectionHost* xs = new MonteRayCrossSectionHost(4);
        xs->setTotalXS(0, 0.0, 4.0 );
        xs->setTotalXS(1, 1.0, 3.0 );
        xs->setTotalXS(2, 2.0, 2.0 );
        xs->setTotalXS(3, 3.0, 1.0 );

        CHECK_EQUAL(-1, xs->getID() );
        xs->setID(3);
        CHECK_EQUAL(3, xs->getID() );

        // can't set twice
        xs->setID(4);
        CHECK_EQUAL(3, xs->getID() );

        delete xs;
    }
    TEST_FIXTURE(MonteRayCrossSectionTestHelper, get_total_xs_from_gpu ) {
        MonteRayCrossSectionHost* xs = new MonteRayCrossSectionHost(4);
        xs->setTotalXS(0, 0.0, 4.0 );
        xs->setTotalXS(1, 1.0, 3.0 );
        xs->setTotalXS(2, 2.0, 2.0 );
        xs->setTotalXS(3, 3.0, 1.0 );

        xs->copyToGPU();

        gpuFloatType_t energy = 0.5;

        setupTimers();
        gpuFloatType_t totalXS = launchGetTotalXS( xs, energy);
        stopTimers();

        CHECK_CLOSE( 3.5f, xs->getTotalXS(0.5), 1e-7 );
        CHECK_CLOSE( 3.5f, totalXS, 1e-7 );

        delete xs;
    }

    TEST_FIXTURE(MonteRayCrossSectionTestHelper, load_u235_from_file)
    {
        MonteRayCrossSectionHost* xs = new MonteRayCrossSectionHost(1);
        xs->read( "MonteRayTestFiles/92235-70c_MonteRayCrossSection.bin");

        gpuFloatType_t energy = 2.0;

        CHECK_EQUAL( 76525, xs->size() );
        CHECK_CLOSE( 233.025, xs->getAWR(), 1e-3 );
        double value = getTotalXS(xs->getXSPtr(), energy);
        CHECK_CLOSE( 7.14769f, value, 1e-5);

        xs->copyToGPU();

        gpuFloatType_t totalXS = launchGetTotalXS( xs, energy);
#ifdef __CUDACC__
        cudaStreamSynchronize(0);
#endif

        CHECK_CLOSE( 7.14769f, totalXS, 1e-5 );

        delete xs;
    }

    TEST_FIXTURE(MonteRayCrossSectionTestHelper, set_photon_ParticleType ) {
        MonteRayCrossSectionHost* xs = new MonteRayCrossSectionHost(4);

        xs->setParticleType( photon );
        CHECK_EQUAL( photon, xs->getParticleType() );

        xs->setTotalXS(0, 1e-11,  4.0  );
        xs->setTotalXS(1, 1.0, 3.0 );
        xs->setTotalXS(2, 2.0, 2.0 );
        xs->setTotalXS(3, 3.0, 1.0 );

        CHECK_CLOSE( 2.0, xs->getTotalXS( 2.0 ), 1e-7);

        delete xs;
    }

    TEST_FIXTURE(MonteRayCrossSectionTestHelper, load_1000_04p_from_file)
    {
        MonteRayCrossSectionHost* xs = new MonteRayCrossSectionHost(1);
        xs->read( "MonteRayTestFiles/1000-04p_MonteRayCrossSection.bin");
        CHECK_EQUAL( photon, xs->getParticleType());
        CHECK_CLOSE( 0.211261, xs->getTotalXS( 1.0 ), 1e-6);
        CHECK_EQUAL(217, xs->size());

        delete xs;
    }

    TEST_FIXTURE(MonteRayCrossSectionTestHelper, load_92000_04p_from_file)
    {
        MonteRayCrossSectionHost* xs = new MonteRayCrossSectionHost(1);
        xs->read( "MonteRayTestFiles/92000-04p_MonteRayCrossSection.bin");
        CHECK_EQUAL( photon, xs->getParticleType());
        CHECK_CLOSE( 30.9887, xs->getTotalXS( 1.0 ), 1e-4);
        CHECK_EQUAL( 451, xs->size());

        delete xs;
    }

}
