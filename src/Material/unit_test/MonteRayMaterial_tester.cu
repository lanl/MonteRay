#include <UnitTest++.h>

#include "MonteRayMaterial.hh"
#include "MonteRayCrossSection.hh"
#include "Material_test_helper.hh"

SUITE( MonteRayMaterial_tester ) {
    TEST( ctor ) {
        MonteRayMaterialHost mat(1);

        MonteRayCrossSectionHost xs(4);
        xs.setTotalXS(0, 0.0, 4.0 );
        xs.setTotalXS(1, 1.0, 3.0 );
        xs.setTotalXS(2, 2.0, 2.0 );
        xs.setTotalXS(3, 3.0, 1.0 );
        xs.setAWR( gpu_AvogadroBarn / ( 0.95 * gpu_neutron_molar_mass ) );
        CHECK_CLOSE( 2.5, xs.getTotalXS(1.5), 1e-7);

        gpuFloatType_t fraction = 0.95;
        mat.add( 0, xs, fraction);
        mat.setID( 0, 3);

        CHECK_CLOSE( 0.95, mat.getFraction(0), 1e-7 );
        CHECK_EQUAL( 3, xs.getID() );
        CHECK_EQUAL( 1, mat.getNumIsotopes() );
        CHECK_CLOSE( 2.5*0.95, mat.getTotalXS(1.5), 1e-7);
        CHECK(true);
    }

    TEST_FIXTURE(MaterialTestHelper, send_to_gpu_testGetNumberOfIsotopes)
    {
        MonteRayMaterialHost mat(1);

        MonteRayCrossSectionHost xs(4);
        xs.setTotalXS(0, 0.0, 4.0 );
        xs.setTotalXS(1, 1.0, 3.0 );
        xs.setTotalXS(2, 2.0, 2.0 );
        xs.setTotalXS(3, 3.0, 1.0 );

        gpuFloatType_t fraction = 0.95;
        mat.add( 0, xs, fraction);

        mat.copyToGPU();
        xs.copyToGPU();

        setupTimers();
        unsigned numIsotopes = mat.launchGetNumIsotopes();
        stopTimers();
        CHECK_EQUAL( 1, numIsotopes );
    }

}
