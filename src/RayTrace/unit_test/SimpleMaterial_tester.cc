#include <UnitTest++.h>

#include "SimpleMaterial.h"
#include "genericGPU_test_helper.hh"

SUITE( SimpleMaterial_tester ) {
    TEST( ctor ) {
        SimpleMaterialHost mat(1);

        SimpleCrossSectionHost xs(4);
        xs.setTotalXS(0, 0.0, 4.0 );
        xs.setTotalXS(1, 1.0, 3.0 );
        xs.setTotalXS(2, 2.0, 2.0 );
        xs.setTotalXS(3, 3.0, 1.0 );

        gpuFloatType_t fraction = 0.95;
        mat.add( 0, xs, fraction);

        CHECK_CLOSE( 0.95, mat.getFraction(0), 1e-7 );
        CHECK_EQUAL( 1, mat.getNumIsotopes() );
    }

    TEST_FIXTURE(GenericGPUTestHelper, send_to_gpu_testGetNumberOfIsotopes)
    {
        SimpleMaterialHost mat(1);

        SimpleCrossSectionHost xs(4);
        xs.setTotalXS(0, 0.0, 4.0 );
        xs.setTotalXS(1, 1.0, 3.0 );
        xs.setTotalXS(2, 2.0, 2.0 );
        xs.setTotalXS(3, 3.0, 1.0 );

        xs.copyToGPU();

        gpuFloatType_t fraction = 0.95;
        mat.add( 0, xs, fraction);

        mat.copyToGPU();

        setupTimers();
        unsigned numIsotopes = mat.launchGetNumIsotopes();
        stopTimers();
        CHECK_EQUAL( 1, numIsotopes );
    }

}
