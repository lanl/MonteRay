#include <UnitTest++.h>

#include "MonteRay_SpatialGrid.hh"
#include "GPUSync.hh"

using namespace MonteRay;

SUITE( MonteRay_SpatialGrid_Tester ) {
	typedef MonteRay_SpatialGrid Grid_t;

	TEST( ctor ) {
		CHECK(true);
		Grid_t grid;
	}

	TEST( ctor_ptr ) {
		CHECK(true);
        std::unique_ptr<Grid_t> pGridInfo = std::unique_ptr<Grid_t>( new Grid_t() );
    }

    TEST( setgetCoordinateSystem ) {
        Grid_t grid;
        CHECK_EQUAL( TransportMeshTypeEnum::NONE, grid.getCoordinateSystem() );
        grid.setCoordinateSystem( TransportMeshTypeEnum::Cartesian );
        CHECK_EQUAL( TransportMeshTypeEnum::Cartesian, grid.getCoordinateSystem() );
    }

    TEST( checkCoordinateSystemLimit ) {
        Grid_t grid;

        bool exception=false;
        try{
            grid.setCoordinateSystem( TransportMeshTypeEnum::MAX );
        }
        catch( ... ) {
        	exception=true;
        }

        CHECK_EQUAL(true, exception );

    }

    TEST( setgetDimension ) {
         Grid_t grid;
         CHECK_EQUAL( 0U, grid.getDimension() );
         grid.setDimension( 3 );
         CHECK_EQUAL( 3U, grid.getDimension() );
     }

     TEST( checkDimensionUpperLimit ) {
         Grid_t grid;
         CHECK_EQUAL( 0U, grid.getDimension() );

         bool exception=false;
         try{
             grid.setDimension(4);
         }
         catch( ... ) {
        	 exception=true;
         }
         CHECK_EQUAL(true, exception );
     }

     TEST( checkDimensionLowerLimit ) {
         Grid_t grid;

         bool exception=false;
         try{
             grid.setDimension(0);
         }
         catch( ... ) {
        	 exception=true;
         }
         CHECK_EQUAL(true, exception );
     }

//  Disable assignment operator until needed
//     TEST( assignmentOperator ){
//         Grid_t grid;
//         grid.setCoordinateSystem( TransportMeshTypeEnum::Cartesian );
//         grid.setDimension( 3 );
//         grid.setGrid( MonteRay_SpatialGrid::CART_X, -10.0, 10.0, 100);
//         grid.setGrid( MonteRay_SpatialGrid::CART_Y, -10.0, 10.0, 100);
//         grid.setGrid( MonteRay_SpatialGrid::CART_Z, -10.0, 10.0, 100);
//         grid.initialize();
//
//         Grid_t newGrid;
//
//         CHECK_EQUAL( 0, newGrid.getDimension() );
//         CHECK( !newGrid.isInitialized() );
//
//         newGrid = grid;
//
//         CHECK_EQUAL( grid.getCoordinateSystem(), newGrid.getCoordinateSystem() );
//         CHECK_EQUAL( grid.getDimension(), newGrid.getDimension() );
//         CHECK_EQUAL( -10.0, newGrid.getVertex(0,0) );
//         CHECK_EQUAL( -9.8, newGrid.getVertex(0,1) );
//         CHECK_EQUAL( -10.0, newGrid.getVertex(1,0) );
//         CHECK_EQUAL( -10.0, newGrid.getVertex(2,0) );
//         CHECK( newGrid.isInitialized() );
//     }

     TEST( write_to_file_Cartesian ) {
         Grid_t grid;
         grid.setCoordinateSystem( TransportMeshTypeEnum::Cartesian );
         grid.setDimension( 3 );
         grid.setGrid( MonteRay_SpatialGrid::CART_X, -10.0, 10.0, 100);
         grid.setGrid( MonteRay_SpatialGrid::CART_Y, -10.0, 10.0, 100);
         grid.setGrid( MonteRay_SpatialGrid::CART_Z, -10.0, 10.0, 100);
         unsigned ExpectedNumCells = 100 * 100 * 100;
         CHECK_EQUAL(ExpectedNumCells, grid.getNumCells() );

         grid.write( "test_spatialGrid_XYZ.bin" );

         {
             Grid_t newGrid;
             newGrid.read( "test_spatialGrid_XYZ.bin" );

             CHECK_EQUAL( 3U, newGrid.getDimension() );
             CHECK_EQUAL( TransportMeshTypeEnum::Cartesian, newGrid.getCoordinateSystem() );
             CHECK_EQUAL(ExpectedNumCells, newGrid.getNumCells() );
         }
     }

     TEST( write_to_file_Spherical ) {
         Grid_t grid;
         grid.setCoordinateSystem( TransportMeshTypeEnum::Spherical );
         grid.setDimension( 1 );
         grid.setGrid( MonteRay_SpatialGrid::SPH_R, 0.0, 10.0, 100);
         unsigned ExpectedNumCells = 100;
         CHECK_EQUAL(ExpectedNumCells, grid.getNumCells() );

         grid.write( "test_spatialGrid_SphR.bin" );

         {
             Grid_t newGrid;
             newGrid.read( "test_spatialGrid_SphR.bin" );

             CHECK_EQUAL( 1U, newGrid.getDimension() );
             CHECK_EQUAL( TransportMeshTypeEnum::Spherical, newGrid.getCoordinateSystem() );
             CHECK_EQUAL(ExpectedNumCells, newGrid.getNumCells() );
         }
     }

//     TEST( write_to_file_Cylindrical ) {
//          const int DIM=2;
//          unsigned NumRCells = 10;
//          unsigned NumZCells = 100;
//
//          SpatialGrid gridOut;
//          gridOut.setDimension( DIM );
//          gridOut.setCoordinateSystem( TransportMeshTypeEnum::Spherical );
//          gridOut.setGrid( SpatialGrid::CYLR_R, 0.0, 1.0, NumRCells);
//          gridOut.setGrid( SpatialGrid::CYLR_Z, 0.0, 10.0, NumZCells);
//          gridOut.initialize();
//
//          CHECK_EQUAL( NumRCells*NumZCells, gridOut.numCells() );
//
//          SpatialGrid gridIn;
//
//          gridOut.write("SpatialGrid_2D.xml");
//          gridIn.read("SpatialGrid_2D.xml");
//
//          CHECK_EQUAL( gridOut.numCells(), gridIn.numCells() );
//      }

}
