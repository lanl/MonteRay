#include <UnitTest++.h>

#include <memory>
#include <vector>
#include <array>

#include "MonteRay_CylindricalGrid.hh"
#include "MonteRay_SpatialGrid.hh"
#include "MonteRayVector3D.hh"
#include "MonteRayConstants.hh"
#include "MonteRayCopyMemory.t.hh"

using namespace MonteRay;

namespace MonteRay_CylindricalGrid_tester{

SUITE( MonteRay_CylindricalGrid_basic_tests ) {
    using Grid_t = MonteRay_CylindricalGrid;
    using GridBins_t = MonteRay_GridBins;
    using GridBins_t = Grid_t::GridBins_t;
    using pGridInfo_t = GridBins_t*;
    using pArrayOfpGridInfo_t = Grid_t::pArrayOfpGridInfo_t;

    typedef MonteRay::Vector3D<gpuRayFloat_t> Position_t;

    enum coord {R=0,Z=1,Theta=2,DIM=3};

    class gridTestData {
    public:
        gridTestData(){
            std::vector<gpuRayFloat_t> Rverts = { 1.3, 2.2, 5.0 };

            pGridInfo[R] = new GridBins_t();
            pGridInfo[Z] = new GridBins_t();

            pGridInfo[R]->initialize( Rverts );
            pGridInfo[Z]->initialize( -10.1, 10.2, 20 );

        }
        ~gridTestData(){
            delete pGridInfo[R];
            delete pGridInfo[Z];
        }

        pArrayOfpGridInfo_t pGridInfo;
    };

    TEST( Ctor ) {
        gridTestData data;
        Grid_t grid(2, data.pGridInfo);

        CHECK( true );
    }

    TEST( getDim ) {
        gridTestData data;

        Grid_t grid(2, data.pGridInfo );
        CHECK_EQUAL( 2, grid.getDimension() );
    }

    TEST( ctor_3args ) {
        gridTestData data;
        std::unique_ptr<Grid_t> pCylindrical = std::unique_ptr<Grid_t>( new Grid_t(2, data.pGridInfo[R], data.pGridInfo[Z] ));

        CHECK_EQUAL(  3, pCylindrical->getNumBins(0) );
        CHECK_EQUAL( 20, pCylindrical->getNumBins(1) );
        //CHECK_EQUAL(  1, pCylindrical->getNumBins(2) );
    }

    TEST( ctor_2args ) {
        gridTestData data;

        std::unique_ptr<Grid_t> pCylindrical = std::unique_ptr<Grid_t>( new Grid_t(2, data.pGridInfo ));

        CHECK_EQUAL(  3, pCylindrical->getNumBins(0) );
        CHECK_EQUAL( 20, pCylindrical->getNumBins(1) );
        //CHECK_EQUAL(  1, pCylindrical->getNumBins(2) );

        CHECK_CLOSE(   1.3, pCylindrical->getRVertex(0), 1e-5 );
        CHECK_CLOSE( -10.1, pCylindrical->getZVertex(0), 1e-5 );
    }

    TEST( CylindricalGrid_modifies_numRBins_of_GridBins ) {
        std::vector<double> Rverts = { 1.5, 2.0 };
        std::vector<double> Zverts = { -10, -5, 0, 5, 10 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[R] = new GridBins_t();
        pGridInfo[Z] = new GridBins_t();
        pGridInfo[R]->initialize(  Rverts );
        pGridInfo[Z]->initialize(  Zverts );

        CHECK_EQUAL( 1, pGridInfo[R]->getNumBins() );

        Grid_t grid( 2, pGridInfo );
        CHECK_EQUAL( 2, pGridInfo[R]->getNumBins() );
        CHECK_EQUAL( 2, grid.getNumRBins() );

        delete pGridInfo[R];
        delete pGridInfo[Z];
    }

    TEST( special_case_with_1_R_vertex ) {

        std::vector<double> Rverts = { 2.0 };
        std::vector<double> Zverts = { -10, -5, 0, 5, 10 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[R] = new GridBins_t();
        pGridInfo[Z] = new GridBins_t();
        pGridInfo[R]->initialize(  Rverts );
        pGridInfo[Z]->initialize(  Zverts );

        Grid_t grid( 2, pGridInfo );
        CHECK_EQUAL( 2, grid.getDimension() );

        CHECK_EQUAL( 1, grid.getNumRBins() );
        CHECK_EQUAL( 4, grid.getNumZBins() );
        //CHECK_EQUAL( 0, grid.getNumThetaBins() );

        CHECK_CLOSE( 2.0, grid.getRVertex(0), 1e-11 );
        CHECK_CLOSE(-10.0, grid.getZVertex(0), 1e-11 );
        CHECK_CLOSE( 5.0, grid.getZVertex(3), 1e-11 );

        delete pGridInfo[R];
        delete pGridInfo[Z];
    }

    TEST( special_case_remove_zero_R_entry ) {

        std::vector<double> Rverts = { 0.0, 1.5, 2.0 };
        std::vector<double> Zverts = { -10, -5, 0, 5, 10 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[R] = new GridBins_t();
        pGridInfo[Z] = new GridBins_t();
        pGridInfo[R]->initialize(  Rverts );
        pGridInfo[Z]->initialize(  Zverts );

        Grid_t grid( 2, pGridInfo );
        CHECK_EQUAL( 2, grid.getDimension() );

        CHECK_EQUAL( 2, grid.getNumRBins() );
        CHECK_EQUAL( 4, grid.getNumZBins() );
        //CHECK_EQUAL( 0, grid.getNumThetaBins() );

        CHECK_CLOSE( 1.5, grid.getRVertex(0), 1e-11 );
        CHECK_CLOSE( 2.0, grid.getRVertex(1), 1e-11 );
        CHECK_CLOSE(-10.0, grid.getZVertex(0), 1e-11 );
        CHECK_CLOSE( 5.0, grid.getZVertex(3), 1e-11 );

        delete pGridInfo[R];
        delete pGridInfo[Z];
    }

    TEST( convertFromCartesian ) {

        std::vector<double> Rverts = { 0.0, 1.5, 2.0 };
        std::vector<double> Zverts = { -10, -5, 0, 5, 10 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[R] = new GridBins_t();
        pGridInfo[Z] = new GridBins_t();
        pGridInfo[R]->initialize(  Rverts );
        pGridInfo[Z]->initialize(  Zverts );

        Grid_t grid( 2, pGridInfo );
        CHECK_EQUAL( 2, grid.getDimension() );

        Vector3D<gpuRayFloat_t> pos = grid.convertFromCartesian( Vector3D<gpuRayFloat_t>( 1.0, 0.0, 5.0) );
        CHECK_CLOSE( 1.0, pos[0], 1e-11);
        CHECK_CLOSE( 5.0, pos[1], 1e-11);
        CHECK_CLOSE( 0.0, pos[2], 1e-11);

        pos = grid.convertFromCartesian( Vector3D<gpuRayFloat_t>( 1.0, 1.0, 5.0) );
        CHECK_CLOSE( std::sqrt(2.0), pos[0], 1e-5);
        CHECK_CLOSE( 5.0, pos[1], 1e-11);
        CHECK_CLOSE( 0.0, pos[2], 1e-11);

        delete pGridInfo[R];
        delete pGridInfo[Z];
    }

    TEST( getRadialIndexFromR ) {

        std::vector<double> Rverts = { 1.0, 2.0, 3.0 };
        std::vector<double> Zverts = { -10,  0,  10 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[R] = new GridBins_t();
        pGridInfo[Z] = new GridBins_t();
        pGridInfo[R]->initialize(  Rverts );
        pGridInfo[Z]->initialize(  Zverts );

        Grid_t grid( 2, pGridInfo );
        CHECK_EQUAL( 2, grid.getDimension() );

        CHECK_EQUAL(   0, pGridInfo[R]->getRadialIndexFromR( 0.5 ) );
        CHECK_EQUAL(   1, pGridInfo[R]->getRadialIndexFromR( 1.5 ) );
        CHECK_EQUAL(   2, pGridInfo[R]->getRadialIndexFromR( 2.5 ) );
        CHECK_EQUAL(   3, pGridInfo[R]->getRadialIndexFromR( 3.5 ) );
        CHECK_EQUAL(   3, pGridInfo[R]->getRadialIndexFromR( 30.5 ) );

        CHECK_EQUAL(   0, grid.getRadialIndexFromR( 0.5 ) );
        CHECK_EQUAL(   1, grid.getRadialIndexFromR( 1.5 ) );
        CHECK_EQUAL(   2, grid.getRadialIndexFromR( 2.5 ) );
        CHECK_EQUAL(   3, grid.getRadialIndexFromR( 3.5 ) );
        CHECK_EQUAL(   3, grid.getRadialIndexFromR( 30.5 ) );

        delete pGridInfo[R];
        delete pGridInfo[Z];
    }

    TEST( getRadialIndexFromRSq ) {

        std::vector<double> Rverts = { 1.0, 2.0, 3.0 };
        std::vector<double> Zverts = { -10,  0,  10 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[R] = new GridBins_t();
        pGridInfo[Z] = new GridBins_t();
        pGridInfo[R]->initialize(  Rverts );
        pGridInfo[Z]->initialize(  Zverts );

        Grid_t grid( 2, pGridInfo );
        CHECK_EQUAL( 2, grid.getDimension() );

        CHECK_EQUAL(   0, pGridInfo[R]->getRadialIndexFromRSq( 0.5*0.5 ) );
        CHECK_EQUAL(   1, pGridInfo[R]->getRadialIndexFromRSq( 1.5*1.5 ) );
        CHECK_EQUAL(   2, pGridInfo[R]->getRadialIndexFromRSq( 2.5*2.5 ) );
        CHECK_EQUAL(   3, pGridInfo[R]->getRadialIndexFromRSq( 3.5*3.4 ) );
        CHECK_EQUAL(   3, pGridInfo[R]->getRadialIndexFromRSq( 30.5*30.5 ) );

        CHECK_EQUAL(   0, grid.getRadialIndexFromRSq( 0.5*0.5 ) );
        CHECK_EQUAL(   1, grid.getRadialIndexFromRSq( 1.5*1.5 ) );
        CHECK_EQUAL(   2, grid.getRadialIndexFromRSq( 2.5*2.5 ) );
        CHECK_EQUAL(   3, grid.getRadialIndexFromRSq( 3.5*3.4 ) );
        CHECK_EQUAL(   3, grid.getRadialIndexFromRSq( 30.5*30.5 ) );

        delete pGridInfo[R];
        delete pGridInfo[Z];
    }

    TEST( getAxialIndex ) {
        std::vector<double> Rverts = { 1.0, 2.0, 3.0 };
        std::vector<double> Zverts = { -10,  0,  10 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[R] = new GridBins_t();
        pGridInfo[Z] = new GridBins_t();
        pGridInfo[R]->initialize(  Rverts );
        pGridInfo[Z]->initialize(  Zverts );

        Grid_t grid( 2, pGridInfo );
        CHECK_EQUAL( 2, grid.getDimension() );

        CHECK_EQUAL(  -1, grid.getAxialIndex( -100.5 ) );
        CHECK_EQUAL(  -1, grid.getAxialIndex( -10.5 ) );
        CHECK_EQUAL(   0, grid.getAxialIndex( -9.5 ) );
        CHECK_EQUAL(   1, grid.getAxialIndex( 9.5) );
        CHECK_EQUAL(   2, grid.getAxialIndex( 10.5 ) );
        CHECK_EQUAL(   2, grid.getAxialIndex( 100.5 ) );

        delete pGridInfo[R];
        delete pGridInfo[Z];
    }

    TEST( isIndexOutside_R ) {
        std::vector<double> Rverts = { 1.0, 2.0, 3.0 };
        std::vector<double> Zverts = { -10,  0,  10 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[R] = new GridBins_t();
        pGridInfo[Z] = new GridBins_t();
        pGridInfo[R]->initialize(  Rverts );
        pGridInfo[Z]->initialize(  Zverts );

        Grid_t grid( 2, pGridInfo );
        CHECK_EQUAL( 2, grid.getDimension() );

        CHECK_EQUAL(   false, grid.isIndexOutside(R, 0 ) );
        CHECK_EQUAL(   false, grid.isIndexOutside(R, 1 ) );
        CHECK_EQUAL(   false, grid.isIndexOutside(R, 2 ) );
        CHECK_EQUAL(    true, grid.isIndexOutside(R, 3 ) );

        delete pGridInfo[R];
        delete pGridInfo[Z];
    }

    TEST( isIndexOutside_Z ) {
        std::vector<double> Rverts = { 1.0, 2.0, 3.0 };
        std::vector<double> Zverts = { -10,  0,  10 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[R] = new GridBins_t();
        pGridInfo[Z] = new GridBins_t();
        pGridInfo[R]->initialize(  Rverts );
        pGridInfo[Z]->initialize(  Zverts );

        Grid_t grid( 2, pGridInfo );
        CHECK_EQUAL( 2, grid.getDimension() );

        CHECK_EQUAL(    true, grid.isIndexOutside(Z, -1 ) );
        CHECK_EQUAL(   false, grid.isIndexOutside(Z,  0 ) );
        CHECK_EQUAL(   false, grid.isIndexOutside(Z,  1 ) );
        CHECK_EQUAL(    true, grid.isIndexOutside(Z,  2 ) );

        delete pGridInfo[R];
        delete pGridInfo[Z];
    }

    TEST( getIndex ) {
        std::vector<double> Rverts = { 1.0, 2.0, 3.0 };
        std::vector<double> Zverts = { -10,  0,  10 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[R] = new GridBins_t();
        pGridInfo[Z] = new GridBins_t();
        pGridInfo[R]->initialize(  Rverts );
        pGridInfo[Z]->initialize(  Zverts );

        Grid_t grid( 2, pGridInfo );
        CHECK_EQUAL( 2, grid.getDimension() );

        Position_t pos1(  0.5,  0.0, -9.5 );
        Position_t pos2(  1.5,  0.0, -9.5 );
        Position_t pos3(  2.5,  0.0, -9.5 );
        Position_t pos4(  3.5,  0.0, -9.5 );

        CHECK_EQUAL(   0, grid.getIndex( pos1 ) );
        CHECK_EQUAL(   1, grid.getIndex( pos2 ) );
        CHECK_EQUAL(   2, grid.getIndex( pos3 ) );
        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, grid.getIndex( pos4 ) );

        pos1 = Position_t(  0.5,  0.0, 9.5 );
        pos2 = Position_t(  1.5,  0.0, 9.5 );
        pos3 = Position_t(  2.5,  0.0, 9.5 );
        pos4 = Position_t(  3.5,  0.0, 9.5 );

        CHECK_EQUAL(   3, grid.getIndex( pos1 ) );
        CHECK_EQUAL(   4, grid.getIndex( pos2 ) );
        CHECK_EQUAL(   5, grid.getIndex( pos3 ) );
        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, grid.getIndex( pos4 ) );

        pos1 = Position_t(  0.0,  0.5, 9.5 );
        pos2 = Position_t(  0.0,  1.5, 9.5 );
        pos3 = Position_t(  0.0,  2.5, 9.5 );
        pos4 = Position_t(  0.0,  3.5, 9.5 );

        CHECK_EQUAL(   3, grid.getIndex( pos1 ) );
        CHECK_EQUAL(   4, grid.getIndex( pos2 ) );
        CHECK_EQUAL(   5, grid.getIndex( pos3 ) );
        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, grid.getIndex( pos4 ) );

        pos1 = Position_t(  0.0,  0.5, 10.5 );
        pos2 = Position_t(  0.0,  1.5, 10.5 );
        pos3 = Position_t(  0.0,  2.5, 10.5 );
        pos4 = Position_t(  0.0,  3.5, 10.5 );

        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, grid.getIndex( pos1 ) );
        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, grid.getIndex( pos2 ) );
        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, grid.getIndex( pos3 ) );
        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, grid.getIndex( pos4 ) );

        delete pGridInfo[R];
        delete pGridInfo[Z];
    }

    TEST( isOutside_posRadius ) {
        std::vector<double> Rverts = { 1.0, 2.0, 3.0 };
        std::vector<double> Zverts = { -10,  0,  10 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[R] = new GridBins_t();
        pGridInfo[Z] = new GridBins_t();
        pGridInfo[R]->initialize(  Rverts );
        pGridInfo[Z]->initialize(  Zverts );

        Grid_t grid( 2, pGridInfo );
        CHECK_EQUAL( 2, grid.getDimension() );

        int indices[] = {3,0,0};
        CHECK_EQUAL( true, grid.isOutside( indices ) );

        delete pGridInfo[R];
        delete pGridInfo[Z];
    }

    TEST( isOutside_Radius_false ) {
        std::vector<double> Rverts = { 1.0, 2.0, 3.0 };
        std::vector<double> Zverts = { -10,  0,  10 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[R] = new GridBins_t();
        pGridInfo[Z] = new GridBins_t();
        pGridInfo[R]->initialize(  Rverts );
        pGridInfo[Z]->initialize(  Zverts );

        Grid_t grid( 2, pGridInfo );
        CHECK_EQUAL( 2, grid.getDimension() );

        int indices[] = {2,0,0};
        CHECK_EQUAL( false, grid.isOutside( indices ) );

        indices[0] = 0;
        CHECK_EQUAL( false, grid.isOutside( indices ) );

        delete pGridInfo[R];
        delete pGridInfo[Z];
    }

    TEST( isOutside_negZ ) {
        std::vector<double> Rverts = { 1.0, 2.0, 3.0 };
        std::vector<double> Zverts = { -10,  0,  10 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[R] = new GridBins_t();
        pGridInfo[Z] = new GridBins_t();
        pGridInfo[R]->initialize(  Rverts );
        pGridInfo[Z]->initialize(  Zverts );

        Grid_t grid( 2, pGridInfo );
        CHECK_EQUAL( 2, grid.getDimension() );

        int indices[] = {0,-1,0};
        CHECK_EQUAL( true, grid.isOutside( indices ) );

        delete pGridInfo[R];
        delete pGridInfo[Z];
    }

    TEST( isOutside_posZ ) {
        std::vector<double> Rverts = { 1.0, 2.0, 3.0 };
        std::vector<double> Zverts = { -10,  0,  10 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[R] = new GridBins_t();
        pGridInfo[Z] = new GridBins_t();
        pGridInfo[R]->initialize(  Rverts );
        pGridInfo[Z]->initialize(  Zverts );

        Grid_t grid( 2, pGridInfo );
        CHECK_EQUAL( 2, grid.getDimension() );

        int indices[] = {0,2,0};
        CHECK_EQUAL( true, grid.isOutside( indices ) );

        delete pGridInfo[R];
        delete pGridInfo[Z];
    }

    TEST( isOutside_Z_false ) {
        std::vector<double> Rverts = { 1.0, 2.0, 3.0 };
        std::vector<double> Zverts = { -10,  0,  10 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[R] = new GridBins_t();
        pGridInfo[Z] = new GridBins_t();
        pGridInfo[R]->initialize(  Rverts );
        pGridInfo[Z]->initialize(  Zverts );

        Grid_t grid( 2, pGridInfo );
        CHECK_EQUAL( 2, grid.getDimension() );

        int indices[] = {0,1,0};
        CHECK_EQUAL( false, grid.isOutside( indices ) );

        delete pGridInfo[R];
        delete pGridInfo[Z];
    }

    TEST( calcIJK ) {
        std::vector<double> Rverts = { 1.0, 2.0, 3.0 };
        std::vector<double> Zverts = { -10,  0,  10 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[R] = new GridBins_t();
        pGridInfo[Z] = new GridBins_t();
        pGridInfo[R]->initialize(  Rverts );
        pGridInfo[Z]->initialize(  Zverts );

        Grid_t grid( 2, pGridInfo );
        CHECK_EQUAL( 2, grid.getDimension() );

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

        indices = grid.calcIJK( 3 );
        CHECK_EQUAL( 0, indices.x );
        CHECK_EQUAL( 1, indices.y );
        CHECK_EQUAL( 0, indices.z );

        indices = grid.calcIJK( 4 );
        CHECK_EQUAL( 1, indices.x );
        CHECK_EQUAL( 1, indices.y );
        CHECK_EQUAL( 0, indices.z );

        indices = grid.calcIJK( 5 );
        CHECK_EQUAL( 2, indices.x );
        CHECK_EQUAL( 1, indices.y );
        CHECK_EQUAL( 0, indices.z );

        delete pGridInfo[R];
        delete pGridInfo[Z];
    }


    TEST( getVolume ) {
        std::vector<double> Rverts = { 1.0, 2.0, 3.0 };
        std::vector<double> Zverts = { -10,  0,  10 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[R] = new GridBins_t();
        pGridInfo[Z] = new GridBins_t();
        pGridInfo[R]->initialize(  Rverts );
        pGridInfo[Z]->initialize(  Zverts );

        Grid_t grid( 2, pGridInfo );
        CHECK_EQUAL( 2, grid.getDimension() );

        CHECK_CLOSE( 10.0*(1.0)*MonteRay::pi, grid.getVolume(0), 1e-5 );
        CHECK_CLOSE( 10.0*(4.0-1.0)*MonteRay::pi, grid.getVolume(1), 1e-5 );

        delete pGridInfo[R];
        delete pGridInfo[Z];
    }

}

} // end namespace
