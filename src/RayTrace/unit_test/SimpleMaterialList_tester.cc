#include <UnitTest++.h>

#include "SimpleMaterialList.h"
#include "genericGPU_test_helper.hh"

SUITE( SimpleMaterialList_tester ) {
    TEST( ctor ) {
        SimpleMaterialHost mat(1);

        SimpleCrossSectionHost xs(4);
        xs.setTotalXS(0, 0.1, 4.0 );
        xs.setTotalXS(1, 1.0, 3.0 );
        xs.setTotalXS(2, 2.0, 2.0 );
        xs.setTotalXS(3, 3.0, 1.0 );

        gpuFloatType_t fraction = 0.95;
        mat.add( 0, xs, fraction);

        SimpleMaterialListHost matList(1);
        matList.add(0, mat, 9);

        CHECK_EQUAL(9, matList.getMaterialID(0) );
    }

    TEST( load_single_isotope_material_from_file ) {
    	SimpleCrossSectionHost u235s(1);
    	u235s.read( "/usr/projects/mcatk/user/jsweezy/link_files/u235_simpleCrossSection.bin" );

    	SimpleMaterialHost mat(1);
    	mat.add(0, u235s, 1.0);

    	SimpleMaterialListHost matList(1,1,8192);
    	matList.add( 0, mat, 0 );

    	gpuFloatType_t energy=2.0;
    	gpuFloatType_t density = 1.0;

    	CHECK_CLOSE(  7.17639378000f*gpu_AvogadroBarn / (u235s.getAWR()*gpu_neutron_molar_mass), matList.getTotalXS(0, energy, density ), 1e-6);
    	CHECK_EQUAL(0, u235s.getID() );
    }
    TEST( load_two_single_isotope_materials_from_file ) {
    	SimpleCrossSectionHost u235s(1);
    	u235s.read( "/usr/projects/mcatk/user/jsweezy/link_files/u235_simpleCrossSection.bin" );

    	SimpleCrossSectionHost h1s(1);
    	h1s.read( "/usr/projects/mcatk/user/jsweezy/link_files/h1_simpleCrossSection.bin" );

    	SimpleMaterialHost mat1(1);
    	mat1.add(0, u235s, 1.0);

    	SimpleMaterialHost mat2(1);
    	mat2.add(0, h1s, 1.0);

    	SimpleMaterialListHost matList(2,2,8192);
    	matList.add( 0, mat1, 0 );
    	matList.add( 1, mat2, 1 );

    	gpuFloatType_t energy=2.0;
    	gpuFloatType_t density = 1.0;

    	unsigned HashBin = getHashBin( matList.getHashPtr()->getPtr(), energy);
    	CHECK_CLOSE(  7.17639378000f, getTotalXS(u235s.getXSPtr(), matList.getHashPtr()->getPtr(), HashBin, energy), 1e-6);
    	CHECK_CLOSE(  7.17639378000f*gpu_AvogadroBarn / (u235s.getAWR()*gpu_neutron_molar_mass), matList.getTotalXS(0, energy, density ), 1e-6);
    	CHECK_EQUAL(0, u235s.getID() );
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
    	CHECK_EQUAL(0, u235s.getID() );
    	CHECK_EQUAL(1, h1s.getID() );
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

    	CHECK_EQUAL(0, u235s.getID() );
    	CHECK_EQUAL(1, h1s.getID() );
    }
    TEST( load_two_materials_from_file2 ) {
    	SimpleCrossSectionHost u235s(1);
    	SimpleCrossSectionHost h1s(1);
    	SimpleCrossSectionHost o16s(1);
    	u235s.read( "/usr/projects/mcatk/user/jsweezy/link_files/u235_simpleCrossSection.bin" );
    	h1s.read( "/usr/projects/mcatk/user/jsweezy/link_files/h1_simpleCrossSection.bin" );
    	o16s.read( "/usr/projects/mcatk/user/jsweezy/link_files/o16_simpleCrossSection.bin" );

    	SimpleMaterialHost mat1(2);
    	mat1.add(0, u235s, 0.5);
    	mat1.add(1, h1s, 0.5);

    	SimpleMaterialHost mat2(2);
    	mat2.add(0, u235s, 0.5);
    	mat2.add(1, o16s, 0.5);

    	SimpleMaterialListHost matList(2);
    	matList.add( 0, mat1, 0 );
    	matList.add( 1, mat2, 1 );

    	gpuFloatType_t energy=2.0;
    	gpuFloatType_t density = 1.0;

    	CHECK_EQUAL(0, u235s.getID() );
    	CHECK_EQUAL(1, h1s.getID() );
    	CHECK_EQUAL(2, o16s.getID() );
    }

    TEST_FIXTURE(GenericGPUTestHelper, sent_to_gpu_getTotalXS )
    {
        SimpleCrossSectionHost u235s(1);
        SimpleCrossSectionHost h1s(1);
        u235s.read( "/usr/projects/mcatk/user/jsweezy/link_files/u235_simpleCrossSection.bin" );
        h1s.read( "/usr/projects/mcatk/user/jsweezy/link_files/h1_simpleCrossSection.bin" );

        SimpleMaterialHost mat1(2);
        mat1.add(0, u235s, 0.5);
        mat1.add(1, h1s, 0.5);
        mat1.copyToGPU();

        SimpleMaterialHost mat2(1);
        mat2.add(0, u235s, 1.0);
        mat2.copyToGPU();

        SimpleMaterialListHost matList(2,2,8192);
        matList.add( 0, mat1, 0 );
        matList.add( 1, mat2, 1 );
        matList.copyToGPU();
        u235s.copyToGPU();
        h1s.copyToGPU();

        gpuFloatType_t energy=2.0;
        gpuFloatType_t density = 1.0;

    	setupTimers();
//    	printf("Debug: Calling matList.launchGetTotalXS -- %s %d\n", __FILE__, __LINE__);
    	gpuFloatType_t result = matList.launchGetTotalXS(0, energy, density );
//    	printf("Debug: return from matList.launchGetTotalXS -- %s %d\n", __FILE__, __LINE__);
    	stopTimers();
    	CHECK_CLOSE( 0.0257431f, result, 1e-7 );
    }
    TEST( get_hash_grom_materialList ){
    	SimpleCrossSectionHost* xs = new SimpleCrossSectionHost(10);
    	xs->setTotalXS(0, 1.0, 1.0 );
    	xs->setTotalXS(1, 1.25, 4.0 );
    	xs->setTotalXS(2, 2.0, 3.0 );
    	xs->setTotalXS(3, 2.5, 5.0 );
    	xs->setTotalXS(4, 3.0, 4.0 );
    	xs->setTotalXS(5, 4.0, 4.0 );
    	xs->setTotalXS(6, 5.0, 4.0 );
    	xs->setTotalXS(7, 7.0, 4.0 );
    	xs->setTotalXS(8, 9.0, 4.0 );
    	xs->setTotalXS(9, 10.0, 10.0 );
    	CHECK_EQUAL( -1, xs->getID());

    	SimpleMaterialHost mat(1);

    	gpuFloatType_t fraction = 1.00;
    	mat.add( 0, *xs, fraction);

    	SimpleMaterialListHost matList(1,1,10);
    	matList.add(0, mat, 9);

    	HashLookupHost* hash = matList.getHashPtr();

    	CHECK_EQUAL( 0, xs->getID());
    	CHECK_EQUAL( 1, hash->getNumIsotopes() );

    	delete xs;
    }

}
