#include <UnitTest++.h>

#include <cmath>

#include "MonteRay_MaterialProperties.hh"
#include "MonteRay_ReadLnk3dnt.hh"
#include "genericGPU_test_helper.hh"

namespace MonteRay_MaterialProperties_GPUData_tester {
using namespace MonteRay;

SUITE( MaterialProperties_GPUData_tester ) {
	TEST( read_lnk3dnt ) {
		MonteRay_ReadLnk3dnt readerObject( "lnk3dnt/godivaR_lnk3dnt_cartesian_100x100x100.lnk3dnt" );
		readerObject.ReadMatData();

		MonteRay_MaterialProperties mp;
		mp.setMaterialDescription( readerObject );

		CHECK_EQUAL( 100*100*100, mp.size());

		gpuFloatType_t mass = mp.sumMatDensity(2) * std::pow( ((33.5*2.0)/100), 3);
		CHECK_CLOSE( 2.21573E+04, mass, 1e-1);

		mass = mp.sumMatDensity(3) * std::pow( ((33.5*2.0)/100), 3);
		CHECK_CLOSE( 1.55890E+05, mass, 1);


		unsigned cell = 100*100*50 + 100*50 + 50;

		unsigned numMats = mp.getNumMaterials(cell);
		MonteRay_MaterialProperties::MatID_t ID = mp.getMaterialID(cell, 0);
		MonteRay_MaterialProperties::Density_t den = mp.getMaterialDensity(cell,0);

		CHECK_EQUAL( 1, numMats);
		CHECK_EQUAL( 2, ID );
		CHECK_CLOSE( 18.7922, den, 1e-4);
	}

	TEST_FIXTURE(GenericGPUTestHelper, test_copy_to_gpu)
	{
		MonteRay_ReadLnk3dnt readerObject( "lnk3dnt/godivaR_lnk3dnt_cartesian_100x100x100.lnk3dnt" );
		readerObject.ReadMatData();

		MonteRay_MaterialProperties mp;
		mp.disableReduction();
		mp.setMaterialDescription( readerObject );

	    mp.copyToGPU();

		unsigned numCells = mp.launchGetNumCells();

		CHECK_CLOSE( 100*100*100, numCells, 1e-1);

		unsigned cell = 100*100*50 + 100*50 + 50;

		unsigned numMats = mp.launchGetNumMaterials(cell);
		MonteRay_MaterialProperties::MatID_t ID = mp.launchGetMaterialID(cell, 0);
		MonteRay_MaterialProperties::Density_t den = mp.launchGetMaterialDensity(cell,0);

		CHECK_EQUAL( 1, numMats);
		CHECK_EQUAL( 2, ID );
		CHECK_CLOSE( 18.7922, den, 1e-4);

		// explicit checking of data by kernal calls - very very long, but useful for debugging issues
//		for( unsigned i = 0; i< mp.size(); ++i){
//			unsigned numGPUMaterials = mp.launchGetNumMaterials(i);
//			if( numGPUMaterials != mp.getNumMaterials(i) ) {
//				std::cout << "Debug: Number of Material mismatch: cell=" << i << " cpu= " <<  mp.getNumMaterials(i) << " gpu= " << numGPUMaterials << "\n\n";
//			}
//		}
//
//		for( unsigned i = 0; i< mp.size(); ++i){
//			CHECK_EQUAL( mp.getNumMaterials(i), mp.launchGetNumMaterials(i));
//			for( unsigned j=0; j < mp.getNumMaterials(i); ++j ) {
//				CHECK_EQUAL( mp.getMaterialID(i,j), mp.launchGetMaterialID(i,j));
//				CHECK_CLOSE( mp.getMaterialDensity(i,j), mp.launchGetMaterialDensity(i,j), 1e-5);
//			}
//		}

		gpuFloatType_t mass = mp.launchSumMatDensity(2);

		mass *= std::pow( ((33.5*2.0)/100), 3);
		CHECK_CLOSE( 2.21573E+04, mass, 1e-1);
	}
}

}
