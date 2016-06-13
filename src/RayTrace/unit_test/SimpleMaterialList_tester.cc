#include <UnitTest++.h>

#include "SimpleMaterialList.h"
#include "genericGPU_test_helper.hh"

SUITE( SimpleMaterialList_tester ) {
    TEST( ctor ) {
        SimpleMaterialHost mat(1);

        SimpleCrossSectionHost xs(4);
        xs.setTotalXS(0, 0.0, 4.0 );
        xs.setTotalXS(1, 1.0, 3.0 );
        xs.setTotalXS(2, 2.0, 2.0 );
        xs.setTotalXS(3, 3.0, 1.0 );

        gpuFloatType_t fraction = 0.95;
        mat.add( 0, xs, fraction);

        SimpleMaterialListHost matList(1);
        matList.add(0, mat, 9);

        CHECK_EQUAL(9, matList.getMaterialID(0) );
    }

    TEST( load_single_material_from_file ) {
    	SimpleCrossSectionHost u235s(1);
    	SimpleCrossSectionHost h1s(1);
    	u235s.read( "/usr/projects/mcatk/user/jsweezy/link_files/u235_simpleCrossSection.bin" );
    	h1s.read( "/usr/projects/mcatk/user/jsweezy/link_files/h1_simpleCrossSection.bin" );

    	SimpleMaterialHost mat(2);
    	mat.add(0, u235s, 0.5);
    	mat.add(1, h1s, 0.5);

    	SimpleMaterialListHost matList(1);
    	matList.add( 0, mat, 0 );

    	gpuFloatType_t energy=2.0;
    	gpuFloatType_t density = 1.0;

    	CHECK_CLOSE( 0.0257431, matList.getTotalXS(0, energy, density ), 1e-6);
    }

    TEST( load_two_materials_from_file ) {
    	SimpleCrossSectionHost u235s(1);
    	SimpleCrossSectionHost h1s(1);
    	u235s.read( "/usr/projects/mcatk/user/jsweezy/link_files/u235_simpleCrossSection.bin" );
    	h1s.read( "/usr/projects/mcatk/user/jsweezy/link_files/h1_simpleCrossSection.bin" );

    	SimpleMaterialHost mat1(2);
    	mat1.add(0, u235s, 0.5);
    	mat1.add(1, h1s, 0.5);

    	SimpleMaterialHost mat2(1);
    	mat2.add(0, u235s, 1.0);

    	SimpleMaterialListHost matList(2);
    	matList.add( 0, mat1, 0 );
    	matList.add( 1, mat2, 1 );

    	gpuFloatType_t energy=2.0;
    	gpuFloatType_t density = 1.0;

    	CHECK_CLOSE( 0.0257431, matList.getTotalXS(0, energy, density ), 1e-6);
    	CHECK_CLOSE( 1.8386868760e-02, matList.getTotalXS(1, energy, density ), 1e-6);
    }

    TEST_FIXTURE(GenericGPUTestHelper, sent_to_gpu_getTotalXS )
    {
        SimpleCrossSectionHost u235s(1);
        SimpleCrossSectionHost h1s(1);
        u235s.read( "/usr/projects/mcatk/user/jsweezy/link_files/u235_simpleCrossSection.bin" );
        h1s.read( "/usr/projects/mcatk/user/jsweezy/link_files/h1_simpleCrossSection.bin" );

        u235s.copyToGPU();
        h1s.copyToGPU();

        SimpleMaterialHost mat1(2);
        mat1.add(0, u235s, 0.5);
        mat1.add(1, h1s, 0.5);
        mat1.copyToGPU();

        SimpleMaterialHost mat2(1);
        mat2.add(0, u235s, 1.0);
        mat2.copyToGPU();

        SimpleMaterialListHost matList(2);
        matList.add( 0, mat1, 0 );
        matList.add( 1, mat2, 1 );
        matList.copyToGPU();

        gpuFloatType_t energy=2.0;
        gpuFloatType_t density = 1.0;

    	setupTimers();
    	gpuFloatType_t result = matList.launchGetTotalXS(0, energy, density );
    	stopTimers();
    	CHECK_CLOSE( 0.0257431f, result, 1e-7 );
    }

}
