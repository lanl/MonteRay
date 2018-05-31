#include <UnitTest++.h>

#include <memory>
#include <vector>
#include <array>

#include "MonteRay_SphericalGrid.hh"
#include "MonteRay_SpatialGrid.hh"
#include "GPUSync.hh"
#include "MonteRayVector3D.hh"
#include "MonteRayConstants.hh"

using namespace MonteRay;

namespace MonteRay_SphericalGrid_tester{

SUITE( MonteRay_SphericalGrid_basic_tests ) {
	using Grid_t = MonteRay_SphericalGrid;
	using GridBins_t = MonteRay_GridBins;
	using GridBins_t = Grid_t::GridBins_t;
	using pGridInfo_t = GridBins_t*;
	using pArrayOfpGridInfo_t = Grid_t::pArrayOfpGridInfo_t;

    typedef MonteRay::Vector3D<gpuRayFloat_t> Position_t;

    class gridTestData {
    public:
        enum coord {R,DIM};
        gridTestData(){
            std::vector<gpuRayFloat_t> vertices = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

            pGridInfo[0] = new GridBins_t();

            pGridInfo[0]->initialize( vertices );

        }
        ~gridTestData(){
        	delete pGridInfo[0];
        }

        pArrayOfpGridInfo_t pGridInfo;
    };

    TEST( Ctor_ArrayOfpGridInfo ) {
        gridTestData data;
        Grid_t grid(1,data.pGridInfo);

        CHECK( true );
    }
    TEST( getDim ) {
        gridTestData data;

        Grid_t grid(1, data.pGridInfo );
        CHECK_EQUAL( 1, grid.getDimension() );
    }

    TEST( ctor_pGridInfo ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(1, data.pGridInfo[0] ));

    	CHECK_EQUAL( 10, pGrid->getNumBins(0) );
    }

    TEST( special_case_with_1_R_vertex ) {
        std::vector<double> Rverts { 2.0 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[0] = new GridBins_t();
        pGridInfo[0]->initialize( Rverts );

        Grid_t grid( 1, pGridInfo );
        CHECK_EQUAL( 1, grid.getDimension() );

        CHECK_EQUAL( 1, grid.getNumRBins() );

        CHECK_CLOSE( 2.0, grid.getRVertex(0), 1e-11 );

        delete pGridInfo[0];
    }

    TEST( special_case_remove_zero_R_entry ) {
        std::vector<double> Rverts { 0.0, 1.5, 2.0 };


        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[0] = new GridBins_t();
        pGridInfo[0]->initialize( Rverts );

        Grid_t grid( 1, pGridInfo );
        CHECK_EQUAL( 1, grid.getDimension() );

        CHECK_EQUAL( 2, grid.getNumRBins() );

        CHECK_CLOSE( 1.5, grid.getRVertex(0), 1e-11 );
        CHECK_CLOSE( 2.0, grid.getRVertex(1), 1e-11 );

        delete pGridInfo[0];
     }

    TEST( convertFromCartesian ){
        std::vector<double> Rverts { 0.0, 1.5, 2.0, 5.0, 6.0 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[0] = new GridBins_t();
        pGridInfo[0]->initialize( Rverts );

        Grid_t grid( 1, pGridInfo );
        CHECK_EQUAL( 1, grid.getDimension() );

        unsigned dim = grid.getDimension();

        Vector3D<double> pos = grid.convertFromCartesian( Vector3D<double>( 1.0, 0.0, 5.0) );
        CHECK_CLOSE( 5.099019513593, pos[0], 1e-11);
        CHECK_CLOSE( 0.0, pos[1], 1e-11);
        CHECK_CLOSE( 0.0, pos[2], 1e-11);

        pos = grid.convertFromCartesian( Vector3D<double>( 2.0, 1.0, 6.0) );
        CHECK_CLOSE( 6.403124237433, pos[0], 1e-11);
        CHECK_CLOSE( 0.0, pos[1], 1e-11);
        CHECK_CLOSE( 0.0, pos[2], 1e-11);

        delete pGridInfo[0];
    }

    TEST( getNumBins ) {
    	gridTestData data;
    	CHECK_EQUAL( 10, data.pGridInfo[0]->getNumBins() );

    	std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(1,data.pGridInfo));

    	CHECK_EQUAL( 10, pGrid->getNumBins(0) );
    }

    TEST( isIndexOutside_R ) {
    	std::vector<double> Rverts { 1.0, 2.0, 3.0 };

    	pArrayOfpGridInfo_t pGridInfo;
    	pGridInfo[0] = new GridBins_t();
    	pGridInfo[0]->initialize( Rverts );

    	Grid_t grid( 1, pGridInfo );

    	CHECK_EQUAL(   false, grid.isIndexOutside(0, 0 ) );
    	CHECK_EQUAL(   false, grid.isIndexOutside(0, 1 ) );
    	CHECK_EQUAL(   false, grid.isIndexOutside(0, 2 ) );
    	CHECK_EQUAL(    true, grid.isIndexOutside(0, 3 ) );

        delete pGridInfo[0];
    }

    TEST( getIndex ) {
        gridTestData data;
        std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(1,data.pGridInfo));

        Position_t pos1( -0.5, -0.5, -0.5 );
        Position_t pos2( -1.5,  0.0,  0.0 );
        Position_t pos3(  2.5,  0.0,  0.0 );
        Position_t pos4(  0.0, -3.5,  0.0 );
        Position_t pos5(  0.0,  4.5,  0.0 );
        Position_t pos6(  0.0,  0.0, -5.5 );
        Position_t pos7(  0.0,  0.0,  6.5 );
        Position_t pos8(  5.5,  5.5,  5.5 );
        Position_t pos9( 10.0, 10.0, 10.0 );

        CHECK_EQUAL( 0, pGrid->getIndex( pos1 ) );
        CHECK_EQUAL( 1, pGrid->getIndex( pos2 ) );
        CHECK_EQUAL( 2, pGrid->getIndex( pos3 ) );
        CHECK_EQUAL( 3, pGrid->getIndex( pos4 ) );
        CHECK_EQUAL( 4, pGrid->getIndex( pos5 ) );
        CHECK_EQUAL( 5, pGrid->getIndex( pos6 ) );
        CHECK_EQUAL( 6, pGrid->getIndex( pos7 ) );
        CHECK_EQUAL( 9, pGrid->getIndex( pos8 ) );
        CHECK_EQUAL( UINT_MAX, pGrid->getIndex( pos9 ) );
    }

    TEST( getRadialIndexFromR_outside ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(1,data.pGridInfo));
        CHECK_EQUAL( 10, pGrid->getRadialIndexFromR( 10.5 ) );
    }
    TEST( getRadialIndexFromR_inside ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(1,data.pGridInfo));
        CHECK_EQUAL( 9, pGrid->getRadialIndexFromR( 9.5 ) );
    }
    TEST( getRadialIndexFromR_insideOnVertex ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(1,data.pGridInfo));
        CHECK_EQUAL( 9, pGrid->getRadialIndexFromR( 9.0 ) );
    }
    TEST( getRadialIndexFromR_center ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(1,data.pGridInfo));
        CHECK_EQUAL( 0, pGrid->getRadialIndexFromR( 0.0 ) );
    }

    TEST( getRadialIndexFromRSq_outside ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(1,data.pGridInfo));
        CHECK_EQUAL( 10, pGrid->getRadialIndexFromRSq( 10.5*10.5 ) );
    }
    TEST( getRadialIndexFromRSq_inside ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(1,data.pGridInfo));
        CHECK_EQUAL( 9, pGrid->getRadialIndexFromRSq( 9.5*9.5 ) );
    }
    TEST( getRadialIndexFromRSq_insideOnVertex ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(1,data.pGridInfo));
        CHECK_EQUAL( 9, pGrid->getRadialIndexFromRSq( 9.0*9.0 ) );
    }
    TEST( getRadialIndexFromRSq_center ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(1,data.pGridInfo));
        CHECK_EQUAL( 0, pGrid->getRadialIndexFromRSq( 0.0 ) );
    }

    TEST( isOutside_index ) {
    	std::vector<double> Rverts { 1.0, 2.0, 3.0 };

    	pArrayOfpGridInfo_t pGridInfo;
    	pGridInfo[0] = new GridBins_t();
    	pGridInfo[0]->initialize( Rverts );

    	Grid_t grid( 1, pGridInfo );

        int indices[] = {3,0,0};
        CHECK_EQUAL( true, grid.isOutside( indices ) );

        delete pGridInfo[0];
    }

    TEST( isOutside_Radius_false ) {
    	std::vector<double> Rverts { 1.0, 2.0, 3.0 };

    	pArrayOfpGridInfo_t pGridInfo;
    	pGridInfo[0] = new GridBins_t();
    	pGridInfo[0]->initialize( Rverts );

    	Grid_t grid( 1, pGridInfo );

        int indices[] = {2,0,0};
        CHECK_EQUAL( false, grid.isOutside( indices ) );

        indices[0] = 0;
        CHECK_EQUAL( false, grid.isOutside( indices ) );

        delete pGridInfo[0];
    }

    TEST( calcIJK ) {
    	std::vector<double> Rverts { 1.0, 2.0, 3.0 };

    	pArrayOfpGridInfo_t pGridInfo;
    	pGridInfo[0] = new GridBins_t();
    	pGridInfo[0]->initialize( Rverts );

    	Grid_t grid( 1, pGridInfo );

        uint3 indices;

        indices = grid.calcIJK( 0 );
        CHECK_EQUAL( 0, indices.x );
        CHECK_EQUAL( 0, indices.y );
        CHECK_EQUAL( 0, indices.z );

        indices = grid.calcIJK( 1 );
        CHECK_EQUAL( 1, indices.x );
        CHECK_EQUAL( 0, indices.y );
        CHECK_EQUAL( 0, indices.z );

        indices = grid.calcIJK( 2 );
        CHECK_EQUAL( 2, indices.x );
        CHECK_EQUAL( 0, indices.y );
        CHECK_EQUAL( 0, indices.z );

        delete pGridInfo[0];
    }

    TEST( getVolume ) {
    	std::vector<double> Rverts { 1.0, 2.0, 3.0 };

    	pArrayOfpGridInfo_t pGridInfo;
    	pGridInfo[0] = new GridBins_t();
    	pGridInfo[0]->initialize( Rverts );

    	Grid_t grid( 1, pGridInfo );

        CHECK_CLOSE( (1.0)*(4.0/3.0)*pi, grid.getVolume(0), 1e-11 );
        CHECK_CLOSE( (8.0-1.0)*(4.0/3.0)*pi, grid.getVolume(1), 1e-11 );

        delete pGridInfo[0];
    }

}

} // end namespace
