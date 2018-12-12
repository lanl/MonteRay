#include <UnitTest++.h>

#include <fstream>

#include "MonteRayMaterial.hh"
#include "MonteRayCrossSection.hh"
#include "Material_test_helper.hh"
#include "HashLookup.hh"

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

    TEST( read_write ) {
        MonteRayMaterialHost write_mat(1);

        MonteRayCrossSectionHost xs(4);
        xs.setTotalXS(0.0001, 0.001, 4.0 );
        xs.setTotalXS(1, 1.0, 3.0 );
        xs.setTotalXS(2, 2.0, 2.0 );
        xs.setTotalXS(3, 3.0, 1.0 );
        xs.setAWR( gpu_AvogadroBarn / ( 0.95 * gpu_neutron_molar_mass ) );
        CHECK_CLOSE( 2.5, xs.getTotalXS(1.5), 1e-7);

        gpuFloatType_t fraction = 0.95;
        write_mat.add( 0, xs, fraction);
        write_mat.setID( 0, 3);
        write_mat.writeToFile( "MonteRayMaterialHost_save_test1.bin" );

        MonteRayMaterialHost mat(1);
        mat.readFromFile( "MonteRayMaterialHost_save_test1.bin" );
        CHECK_CLOSE( 0.95, mat.getFraction(0), 1e-7 );
        CHECK_EQUAL( 3, xs.getID() );
        CHECK_EQUAL( 1, mat.getNumIsotopes() );
        CHECK_CLOSE( 2.5*0.95, mat.getTotalXS(1.5), 1e-7);
        CHECK(true);
    }

    TEST( read ) {
        // test file exists
        std::ifstream exists("MonteRayMaterialHost_save_test1.bin");
        CHECK_EQUAL( true, exists.good() );
        exists.close();

        MonteRayMaterialHost mat(1);
        mat.readFromFile( "MonteRayMaterialHost_save_test1.bin" );
        CHECK_CLOSE( 0.95, mat.getFraction(0), 1e-7 );
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
        mat.setID( 0, 3);

        mat.copyToGPU();
        xs.copyToGPU();

        setupTimers();
        unsigned numIsotopes = mat.launchGetNumIsotopes();
        stopTimers();
        CHECK_EQUAL( 1, numIsotopes );
    }

    TEST_FIXTURE(MaterialTestHelper, read_send_to_gpu )
    {
        // test file exists
        std::ifstream exists("MonteRayMaterialHost_save_test1.bin");
        CHECK_EQUAL( true, exists.good() );
        exists.close();

        MonteRayMaterialHost mat(1);
        mat.readFromFile( "MonteRayMaterialHost_save_test1.bin" );

        mat.copyToGPU();

        setupTimers();
        unsigned numIsotopes = mat.launchGetNumIsotopes();
        stopTimers();
        CHECK_EQUAL( 1, numIsotopes );

        setupTimers();
        gpuFloatType_t xs = mat.launchGetTotalXS(1.5, 1.0);
        stopTimers();
        CHECK_CLOSE( 2.5*0.95, xs, 1e-7);
    }

    TEST_FIXTURE(MaterialTestHelper, read_with_hash_send_to_gpu )
    {
        // test file exists
        std::ifstream exists("MonteRayMaterialHost_save_test1.bin");
        CHECK_EQUAL( true, exists.good() );
        exists.close();

        HashLookupHost hash(1);

        MonteRayMaterialHost mat(1);
        mat.readFromFile( "MonteRayMaterialHost_save_test1.bin", &hash );

        mat.copyToGPU();

        setupTimers();
        unsigned numIsotopes = mat.launchGetNumIsotopes();
        stopTimers();
        CHECK_EQUAL( 1, numIsotopes );

        setupTimers();
        gpuFloatType_t xs = mat.launchGetTotalXSViaHash(hash, 1.5, 1.0);
        stopTimers();
        CHECK_CLOSE( 2.5*0.95, xs, 1e-7);
    }

}
