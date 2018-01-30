#include <UnitTest++.h>

#include "MonteRayMaterialList.hh"
#include "Material_test_helper.hh"

SUITE( MonteRayMaterialList_tester ) {
    TEST( ctor ) {
        MonteRayMaterialHost mat(1);

        MonteRayCrossSectionHost xs(4);
        xs.setTotalXS(0, 0.1, 4.0 );
        xs.setTotalXS(1, 1.0, 3.0 );
        xs.setTotalXS(2, 2.0, 2.0 );
        xs.setTotalXS(3, 3.0, 1.0 );

        gpuFloatType_t fraction = 0.95;
        mat.add( 0, xs, fraction);

        MonteRayMaterialListHost matList(1);
        matList.add(0, mat, 9);

        CHECK_EQUAL(9, matList.getMaterialID(0) );
    }

    TEST( load_single_isotope_material_from_file ) {
    	MonteRayCrossSectionHost u235s(1);
    	u235s.read( "MonteRayTestFiles/92235-70c_MonteRayCrossSection.bin" );

    	MonteRayMaterialHost mat(1);
    	mat.add(0, u235s, 1.0);

    	MonteRayMaterialListHost matList(1,1,8192);
    	matList.add( 0, mat, 0 );

    	gpuFloatType_t energy=2.0;
    	gpuFloatType_t density = 1.0;
    	gpuFloatType_t microXS = u235s.getTotalXS( energy );
    	CHECK_CLOSE( 7.14769f, microXS, 1e-5);
    	CHECK_CLOSE( 7.14769f*gpu_AvogadroBarn / (u235s.getAWR()*gpu_neutron_molar_mass), matList.getTotalXS(0, energy, density ), 1e-6);
    	CHECK_EQUAL(0, u235s.getID() );
    }
    TEST( load_two_single_isotope_materials_from_file ) {
    	MonteRayCrossSectionHost u235s(1);
    	u235s.read( "MonteRayTestFiles/92235-70c_MonteRayCrossSection.bin" );

    	MonteRayCrossSectionHost h1s(1);
    	h1s.read( "MonteRayTestFiles/1001-70c_MonteRayCrossSection.bin" );

    	MonteRayMaterialHost mat1(1);
    	mat1.add(0, u235s, 1.0);

    	MonteRayMaterialHost mat2(1);
    	mat2.add(0, h1s, 1.0);

    	MonteRayMaterialListHost matList(2,2,8192);
    	matList.add( 0, mat1, 0 );
    	matList.add( 1, mat2, 1 );

    	gpuFloatType_t energy=2.0;
    	gpuFloatType_t density = 1.0;

    	unsigned HashBin = getHashBin( matList.getHashPtr()->getPtr(), energy);
    	CHECK_CLOSE(  7.14769f, getTotalXS(u235s.getXSPtr(), matList.getHashPtr()->getPtr(), HashBin, energy), 1e-5);
    	CHECK_CLOSE(  7.14769f*gpu_AvogadroBarn / (u235s.getAWR()*gpu_neutron_molar_mass), matList.getTotalXS(0, energy, density ), 1e-6);
    	CHECK_EQUAL(0, u235s.getID() );
    }
    TEST( load_single_material_from_file ) {
    	MonteRayCrossSectionHost u235s(1);
    	MonteRayCrossSectionHost h1s(1);
    	u235s.read( "MonteRayTestFiles/92235-70c_MonteRayCrossSection.bin" );
    	h1s.read( "MonteRayTestFiles/1001-70c_MonteRayCrossSection.bin" );

    	MonteRayMaterialHost mat(2);
    	mat.add(0, u235s, 0.5);
    	mat.add(1, h1s, 0.5);

    	MonteRayMaterialListHost matList(1);
    	matList.add( 0, mat, 0 );

    	gpuFloatType_t energy=2.0;
    	gpuFloatType_t density = 1.0;

    	CHECK_CLOSE( 0.025643, matList.getTotalXS(0, energy, density ), 1e-6);
    	CHECK_EQUAL(0, u235s.getID() );
    	CHECK_EQUAL(1, h1s.getID() );
    }

    TEST( load_two_materials_from_file ) {
    	MonteRayCrossSectionHost u235s(1);
    	MonteRayCrossSectionHost h1s(1);
    	u235s.read( "MonteRayTestFiles/92235-70c_MonteRayCrossSection.bin" );
    	h1s.read( "MonteRayTestFiles/1001-70c_MonteRayCrossSection.bin" );

    	MonteRayMaterialHost mat1(2);
    	mat1.add(0, u235s, 0.5);
    	mat1.add(1, h1s, 0.5);

    	MonteRayMaterialHost mat2(1);
    	mat2.add(0, u235s, 1.0);

    	MonteRayMaterialListHost matList(2);
    	matList.add( 0, mat1, 0 );
    	matList.add( 1, mat2, 1 );

    	gpuFloatType_t energy=2.0;
    	gpuFloatType_t density = 1.0;

    	CHECK_CLOSE( 0.025643, matList.getTotalXS(0, energy, density ), 1e-6);
    	CHECK_CLOSE( 0.0183133, matList.getTotalXS(1, energy, density ), 1e-6);

    	CHECK_EQUAL(0, u235s.getID() );
    	CHECK_EQUAL(1, h1s.getID() );
    }
    TEST( load_two_materials_from_file2 ) {
    	MonteRayCrossSectionHost u235s(1);
    	MonteRayCrossSectionHost h1s(1);
    	MonteRayCrossSectionHost o16s(1);
    	u235s.read( "MonteRayTestFiles/92235-70c_MonteRayCrossSection.bin" );
    	h1s.read( "MonteRayTestFiles/1001-70c_MonteRayCrossSection.bin" );
    	o16s.read( "MonteRayTestFiles/8016-70c_MonteRayCrossSection.bin" );

    	MonteRayMaterialHost mat1(2);
    	mat1.add(0, u235s, 0.5);
    	mat1.add(1, h1s, 0.5);

    	MonteRayMaterialHost mat2(2);
    	mat2.add(0, u235s, 0.5);
    	mat2.add(1, o16s, 0.5);

    	MonteRayMaterialListHost matList(2);
    	matList.add( 0, mat1, 0 );
    	matList.add( 1, mat2, 1 );

    	gpuFloatType_t energy=2.0;
    	gpuFloatType_t density = 1.0;

    	CHECK_EQUAL(0, u235s.getID() );
    	CHECK_EQUAL(1, h1s.getID() );
    	CHECK_EQUAL(2, o16s.getID() );
    }

    TEST_FIXTURE(MaterialTestHelper, sent_to_gpu_getTotalXS )
    {
        MonteRayCrossSectionHost u235s(1);
        MonteRayCrossSectionHost h1s(1);
        u235s.read( "MonteRayTestFiles/92235-70c_MonteRayCrossSection.bin" );
        h1s.read( "MonteRayTestFiles/1001-70c_MonteRayCrossSection.bin" );

        MonteRayMaterialHost mat1(2);
        mat1.add(0, u235s, 0.5);
        mat1.add(1, h1s, 0.5);
        mat1.copyToGPU();

        MonteRayMaterialHost mat2(1);
        mat2.add(0, u235s, 1.0);
        mat2.copyToGPU();

        MonteRayMaterialListHost matList(2,2,8192);
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
    	CHECK_CLOSE( 0.025643f, result, 1e-7 );
    }
    TEST( get_hash_grom_materialList ){
    	MonteRayCrossSectionHost* xs = new MonteRayCrossSectionHost(10);
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

    	MonteRayMaterialHost mat(1);

    	gpuFloatType_t fraction = 1.00;
    	mat.add( 0, *xs, fraction);

    	MonteRayMaterialListHost matList(1,1,10);
    	matList.add(0, mat, 9);

    	const HashLookupHost* hash = matList.getHashPtr();

    	CHECK_EQUAL( 0, xs->getID());
    	CHECK_EQUAL( 1, hash->getNumIsotopes() );

    	delete xs;
    }

}
