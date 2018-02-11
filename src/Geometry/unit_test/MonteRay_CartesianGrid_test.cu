#include <UnitTest++.h>

#include <memory>
#include <vector>
#include <array>

#include "MonteRay_CartesianGrid.hh"
#include "MonteRay_SpatialGrid.hh"
#include "GPUSync.hh"
#include "MonteRayVector3D.hh"

using namespace MonteRay;

namespace MonteRay_CartesianGrid_tester{

SUITE( MonteRay_CartesianGrid_basic_tests ) {
	using Grid_t = MonteRay_CartesianGrid;
	using GridBins_t = MonteRay_GridBins;
	using GridBins_t = Grid_t::GridBins_t;
	using pGridInfo_t = GridBins_t*;
	using pArrayOfpGridInfo_t = Grid_t::pArrayOfpGridInfo_t;

    typedef MonteRay::Vector3D<gpuFloatType_t> Position_t;

    class gridTestData {
    public:
        enum coord {X,Y,Z,DIM};
        gridTestData(){
            std::vector<gpuFloatType_t> vertices = { -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
                                                       0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10 };

            pGridInfo[X] = new GridBins_t();
            pGridInfo[Y] = new GridBins_t();
            pGridInfo[Z] = new GridBins_t();

            pGridInfo[X]->initialize( vertices );
            pGridInfo[Y]->initialize( vertices );
            pGridInfo[Z]->initialize( vertices );

        }
        ~gridTestData(){
        	delete pGridInfo[X];
        	delete pGridInfo[Y];
        	delete pGridInfo[Z];
        }

        pArrayOfpGridInfo_t pGridInfo;
    };

    TEST( Ctor ) {
        gridTestData data;
        Grid_t cart(3,data.pGridInfo);

        CHECK( true );
    }

    TEST( ctor_4args ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pCart = std::unique_ptr<Grid_t>( new Grid_t(3, data.pGridInfo[0],data.pGridInfo[1],data.pGridInfo[2] ));

    	CHECK_EQUAL( 20, pCart->getNumBins(0) );
    	CHECK_EQUAL( 20, pCart->getNumBins(1) );
    	CHECK_EQUAL( 20, pCart->getNumBins(2) );
    }

    TEST( getNumBins ) {
    	gridTestData data;
    	CHECK_EQUAL( 20, data.pGridInfo[0]->getNumBins() );
    	CHECK_EQUAL( 20, data.pGridInfo[1]->getNumBins() );
    	CHECK_EQUAL( 20, data.pGridInfo[2]->getNumBins() );


    	std::unique_ptr<Grid_t> pCart = std::unique_ptr<Grid_t>( new Grid_t(3,data.pGridInfo));

    	CHECK_EQUAL( 20, pCart->getNumBins(0) );
    	CHECK_EQUAL( 20, pCart->getNumBins(1) );
    	CHECK_EQUAL( 20, pCart->getNumBins(2) );
    }

    TEST( getIndex ) {
        gridTestData data;
        std::unique_ptr<Grid_t> pCart = std::unique_ptr<Grid_t>( new Grid_t(3,data.pGridInfo));

        Position_t pos1( -9.5, -9.5, -9.5 );
        Position_t pos2( -8.5, -9.5, -9.5 );
        Position_t pos3( -9.5, -8.5, -9.5 );
        Position_t pos4( -9.5, -9.5, -8.5 );
        Position_t pos5( -9.5, -7.5, -9.5 );

        CHECK_EQUAL(   0, pCart->getIndex( pos1 ) );
        CHECK_EQUAL(   1, pCart->getIndex( pos2 ) );
        CHECK_EQUAL(  20, pCart->getIndex( pos3 ) );
        CHECK_EQUAL(  40, pCart->getIndex( pos5 ) );
        CHECK_EQUAL( 400, pCart->getIndex( pos4 ) );
    }

    TEST( getDimIndex_negX ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pCart = std::unique_ptr<Grid_t>( new Grid_t(3,data.pGridInfo));
        CHECK_EQUAL( -1, pCart->getDimIndex( 0, -10.5 ) );
    }
    TEST( getDimIndex_posX ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pCart = std::unique_ptr<Grid_t>( new Grid_t(3,data.pGridInfo));
        CHECK_EQUAL( 20, pCart->getDimIndex( 0, 10.5 ) );
    }
    TEST( getDimIndex_inside_negSide_X ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pCart = std::unique_ptr<Grid_t>( new Grid_t(3,data.pGridInfo));
        CHECK_EQUAL( 0, pCart->getDimIndex( 0, -9.5 ) );
    }
    TEST( getDimIndex_inside_posSide_X ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pCart = std::unique_ptr<Grid_t>( new Grid_t(3,data.pGridInfo));
        CHECK_EQUAL( 19, pCart->getDimIndex( 0, 9.5 ) );
    }

    TEST( getDimIndex_negY ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pCart = std::unique_ptr<Grid_t>( new Grid_t(3,data.pGridInfo));
        CHECK_EQUAL( -1, pCart->getDimIndex( 1, -10.5 ) );
    }
    TEST( getDimIndex_posY ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pCart = std::unique_ptr<Grid_t>( new Grid_t(3,data.pGridInfo));
        CHECK_EQUAL( 20, pCart->getDimIndex( 1, 10.5 ) );
    }
    TEST( getDimIndex_inside_negSide_Y ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pCart = std::unique_ptr<Grid_t>( new Grid_t(3,data.pGridInfo));
        CHECK_EQUAL( 0, pCart->getDimIndex( 1, -9.5 ) );
    }
    TEST( getDimIndex_inside_posSide_Y ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pCart = std::unique_ptr<Grid_t>( new Grid_t(3,data.pGridInfo));
        CHECK_EQUAL( 19, pCart->getDimIndex( 1, 9.5 ) );
    }

    TEST( getDimIndex_negZ ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pCart = std::unique_ptr<Grid_t>( new Grid_t(3,data.pGridInfo));
        CHECK_EQUAL( -1, pCart->getDimIndex( 2, -10.5 ) );
    }
    TEST( getDimIndex_posZ ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pCart = std::unique_ptr<Grid_t>( new Grid_t(3,data.pGridInfo));
        CHECK_EQUAL( 20, pCart->getDimIndex( 2, 10.5 ) );
    }
    TEST( getDimIndex_inside_negSide_Z ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pCart = std::unique_ptr<Grid_t>( new Grid_t(3,data.pGridInfo));
        CHECK_EQUAL( 0, pCart->getDimIndex( 2, -9.5 ) );
    }
    TEST( getDimIndex_inside_posSide_Z ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pCart = std::unique_ptr<Grid_t>( new Grid_t(3,data.pGridInfo));
        CHECK_EQUAL( 19, pCart->getDimIndex( 2, 9.5 ) );
    }

    TEST( PositionOutOfBoundsToGrid ) {
        gridTestData data;
        Grid_t cart(3,data.pGridInfo);

        Position_t posNegX( -10.5, -9.5, -9.5 );
        Position_t posPosX(  10.5, -9.5, -9.5 );

        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, cart.getIndex( posNegX ) );
        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, cart.getIndex( posPosX ) );

        Position_t posNegY( -9.5, -10.5, -9.5 );
        Position_t posPosY( -9.5,  10.5, -9.5 );

        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, cart.getIndex( posNegY ) );
        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, cart.getIndex( posPosY ) );

        Position_t posNegZ( -9.5, -9.5, -10.5 );
        Position_t posPosZ( -9.5, -9.5,  10.5 );

        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, cart.getIndex( posNegZ ) );
        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, cart.getIndex( posPosZ ) );
    }

    TEST( PositionOnTheBoundsToGrid_WeDefineOutside ) {
        gridTestData data;
        Grid_t cart(3,data.pGridInfo);

        Position_t posNegX( -10.0, -9.5, -9.5 );
        Position_t posPosX(  10.0, -9.5, -9.5 );

        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, cart.getIndex( posNegX ) );
        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, cart.getIndex( posPosX ) );

        Position_t posNegY( -9.5, -10.0, -9.5 );
        Position_t posPosY( -9.5,  10.0, -9.5 );

        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, cart.getIndex( posNegY ) );
        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, cart.getIndex( posPosY ) );

        Position_t posNegZ( -9.5, -9.5, -10.0 );
        Position_t posPosZ( -9.5, -9.5,  10.0 );

        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, cart.getIndex( posNegZ ) );
        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, cart.getIndex( posPosZ ) );
    }

    TEST( isIndexOutside_negX ) {
        gridTestData data;
        Grid_t cart(3,data.pGridInfo);

        CHECK_EQUAL( true, cart.isIndexOutside(0, -1) );
    }

    TEST( isIndexOutside_posX ) {
        gridTestData data;
        Grid_t cart(3,data.pGridInfo);

        CHECK_EQUAL( true, cart.isIndexOutside(0, 20) );
    }
    TEST( isIndexOutside_false_negEnd ) {
        gridTestData data;
        Grid_t cart(3,data.pGridInfo);

        CHECK_EQUAL( false, cart.isIndexOutside(0, 0) );
    }
    TEST( isIndexOutside_false_posEnd ) {
        gridTestData data;
        Grid_t cart(3,data.pGridInfo);

        CHECK_EQUAL( false, cart.isIndexOutside(0, 19) );
    }

    TEST( isOutside_negX ) {
        gridTestData data;
        Grid_t cart(3,data.pGridInfo);

        int indices[] = {-1,0,0};
        CHECK_EQUAL( true, cart.isOutside( indices ) );
    }
    TEST( isOutside_posX ) {
        gridTestData data;
        Grid_t cart(3,data.pGridInfo);

        int indices[] = {20,0,0};
        CHECK_EQUAL( true, cart.isOutside( indices ) );
    }

    TEST( isOutside_negY ) {
        gridTestData data;
        Grid_t cart(3,data.pGridInfo);

        int indices[] = {0,-1,0};
        CHECK_EQUAL( true, cart.isOutside( indices ) );
    }
    TEST( isOutside_posY ) {
        gridTestData data;
        Grid_t cart(3,data.pGridInfo);

        int indices[] = {0,20,0};
        CHECK_EQUAL( true, cart.isOutside( indices ) );
    }
    TEST( isOutside_negZ ) {
        gridTestData data;
        Grid_t cart(3,data.pGridInfo);

        int indices[] = {0,0,-1};
        CHECK_EQUAL( true, cart.isOutside( indices ) );
    }
    TEST( isOutside_posZ ) {
        gridTestData data;
        Grid_t cart(3,data.pGridInfo);

        int indices[] = {0,0,20};
        CHECK_EQUAL( true, cart.isOutside( indices ) );
    }
    TEST( isOutside_false1 ) {
        gridTestData data;
        Grid_t cart(3,data.pGridInfo);

        int indices[] = {19,0,0};
        CHECK_EQUAL( false, cart.isOutside( indices ) );
    }
    TEST( isOutside_false2 ) {
        gridTestData data;
        Grid_t cart(3,data.pGridInfo);

        int indices[] = {0,0,0};
        CHECK_EQUAL( false, cart.isOutside( indices ) );
    }

    TEST( getVolume ) {
    	pGridInfo_t* pGridInfo = new pGridInfo_t[3];
		pGridInfo[0] = new GridBins_t();
		pGridInfo[1] = new GridBins_t();
		pGridInfo[2] = new GridBins_t();

    	std::vector<gpuFloatType_t> vertices = {-3, -1, 0};

    	pGridInfo[0]->initialize( vertices );
    	pGridInfo[1]->initialize( vertices );
    	pGridInfo[2]->initialize( vertices );

    	Grid_t cart(3,pGridInfo);

    	CHECK_CLOSE( 8.0, cart.getVolume(0), 1e-11 );
    	CHECK_CLOSE( 4.0, cart.getVolume(1), 1e-11 );
    	CHECK_CLOSE( 4.0, cart.getVolume(2), 1e-11 );
    	CHECK_CLOSE( 2.0, cart.getVolume(3), 1e-11 );
    	CHECK_CLOSE( 4.0, cart.getVolume(4), 1e-11 );
    	CHECK_CLOSE( 2.0, cart.getVolume(5), 1e-11 );
    	CHECK_CLOSE( 2.0, cart.getVolume(6), 1e-11 );
    	CHECK_CLOSE( 1.0, cart.getVolume(7), 1e-11 );

    	delete pGridInfo[0];
    	delete pGridInfo[1];
    	delete pGridInfo[2];

    	delete[] pGridInfo;
    }

}

} // end namespace
