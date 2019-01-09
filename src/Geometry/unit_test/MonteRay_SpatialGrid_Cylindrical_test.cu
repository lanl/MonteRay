#include <UnitTest++.h>

#include "MonteRay_SpatialGrid.hh"
#include "MonteRayDefinitions.hh"
#include "MonteRayConstants.hh"
#include "MonteRay_ReadLnk3dnt.hh"
#include "MonteRayCopyMemory.t.hh"
#include "MonteRay_GridSystemInterface.hh"

#include <stdexcept>
#include <fstream>
#include <vector>

#include "MonteRayVector3D.hh"


namespace MonteRay_SpatialGrid_Cylindrical_test{

using namespace MonteRay;

SUITE( MonteRay_SpatialGrid_Cylindrical_tests ) {
    typedef MonteRay_SpatialGrid SpatialGrid;
    typedef MonteRay_SpatialGrid Grid_t;
    typedef MonteRay_SpatialGrid::Position_t Position_t;

    TEST( set_Radial_Vertices ){
        std::vector<double> Rvertices = { 1.0, 2.0, 3.0, 5.0 };

        Grid_t grid;
        grid.setGrid( MonteRay_SpatialGrid::CYLR_R, Rvertices);

        CHECK_EQUAL(   3, grid.getNumGridBins(MonteRay_SpatialGrid::CYLR_R) );

        // set Z vertices, coordinate system and dimension to pass initialization
        std::vector<double> Zvertices = { 1.0, 2.0, 3.0, 5.0 };
        grid.setGrid( MonteRay_SpatialGrid::CYLR_Z, Zvertices);
        grid.setDimension(2);
        grid.setCoordinateSystem( TransportMeshTypeEnum::Cylindrical );
        grid.initialize();

        CHECK_EQUAL(   4, grid.getNumGridBins(MonteRay_SpatialGrid::CYLR_R) );
        CHECK_CLOSE( 1.0, grid.getMinVertex(MonteRay_SpatialGrid::CYLR_R), 1e-6 );
        CHECK_CLOSE( 5.0, grid.getMaxVertex(MonteRay_SpatialGrid::CYLR_R), 1e-6 );
    }

    TEST( set_rz_Vertices ){
        Grid_t grid;
        grid.setCoordinateSystem( TransportMeshTypeEnum::Cylindrical );

        std::vector<double> Rvertices = { 1.0, 2.0, 3.0, 5.0 };
        std::vector<double> Zvertices = { -30, -2, -1, 0, 1, 2, 30 };

        grid.setGrid( SpatialGrid::CYLR_R, Rvertices);
        grid.setGrid( SpatialGrid::CYLR_Z, Zvertices);

        CHECK_EQUAL( 3, grid.getNumGridBins(SpatialGrid::CYLR_R) );
        CHECK_CLOSE( 1.0, grid.getMinVertex(SpatialGrid::CYLR_R), 1e-11 );
        CHECK_CLOSE( 5.0, grid.getMaxVertex(SpatialGrid::CYLR_R), 1e-11 );

        CHECK_EQUAL( 6, grid.getNumGridBins(SpatialGrid::CYLR_Z) );
        CHECK_CLOSE( -30.0, grid.getMinVertex(SpatialGrid::CYLR_Z), 1e-11 );
        CHECK_CLOSE( 30.0, grid.getMaxVertex(SpatialGrid::CYLR_Z), 1e-11 );
    }


    TEST( cylindrical_modifies_grid_data_with_0_entry ){
        Grid_t grid;
        grid.setDimension( 2 );
        grid.setCoordinateSystem( TransportMeshTypeEnum::Cylindrical );

        std::vector<double> Rvertices = { 0.0, 1.0, 2.0, 3.0, 5.0 };
        std::vector<double> Zvertices = { -30, -2, -1, 0, 1, 2, 30 };

        grid.setGrid( SpatialGrid::CYLR_R, Rvertices);
        grid.setGrid( SpatialGrid::CYLR_Z, Zvertices);

        CHECK_EQUAL( 4, grid.getNumGridBins(SpatialGrid::CYLR_R) );
        CHECK_CLOSE( 0.0, grid.getVertex( SpatialGrid::CYLR_R, 0), 1e-11 );

        grid.initialize();

        CHECK_EQUAL( 4, grid.getNumGridBins(SpatialGrid::CYLR_R) );
        CHECK_CLOSE( 1.0, grid.getVertex( SpatialGrid::CYLR_R, 0), 1e-11 );
    }

    TEST( cylindrical_modifies_grid_data_without_0_entry ){
        Grid_t grid;
        grid.setDimension( 2 );
        grid.setCoordinateSystem( TransportMeshTypeEnum::Cylindrical );

        std::vector<double> Rvertices = { 1.0, 2.0, 3.0, 5.0 };
        std::vector<double> Zvertices = { -30, -2, -1, 0, 1, 2, 30 };

        grid.setGrid( SpatialGrid::CYLR_R, Rvertices);
        grid.setGrid( SpatialGrid::CYLR_Z, Zvertices);

        CHECK_EQUAL( 3, grid.getNumGridBins(SpatialGrid::CYLR_R) );
        CHECK_CLOSE( 1.0, grid.getVertex( SpatialGrid::CYLR_R, 0), 1e-11 );

        grid.initialize();

        CHECK_EQUAL( 4, grid.getNumGridBins(SpatialGrid::CYLR_R) );
        CHECK_CLOSE( 1.0, grid.getVertex( SpatialGrid::CYLR_R, 0), 1e-11 );
    }


    TEST( initialize_check_vertices_set ){
        Grid_t grid;
        grid.setDimension( 2 );
        grid.setCoordinateSystem( TransportMeshTypeEnum::Cylindrical );

        std::vector<double> Rvertices = { 0.0, 1.0, 2.0, 3.0, 5.0 };
        std::vector<double> Zvertices = { -30, -2, -1, 0, 1, 2, 30 };

        grid.setGrid( SpatialGrid::CYLR_R, Rvertices);

        CHECK_THROW( grid.initialize(), std::runtime_error);
    }

    TEST( getIndexByPos ){
        Grid_t grid;
        grid.setDimension( 2 );
        grid.setCoordinateSystem( TransportMeshTypeEnum::Cylindrical );

        std::vector<double> Rvertices = { 1.0, 2.0, 3.0 };
        std::vector<double> Zvertices = { -10,  0,  10 };

        grid.setGrid( SpatialGrid::CYLR_R, Rvertices);
        grid.setGrid( SpatialGrid::CYLR_Z, Zvertices);

        grid.initialize();

        Position_t pos1(  0.5,  0.0, -9.5 );
        Position_t pos2(  1.5,  0.0, -9.5 );
        Position_t pos3(  2.5,  0.0, -9.5 );
        Position_t pos4(  3.5,  0.0, -9.5 );

        CHECK_EQUAL(   0, grid.getIndex( pos1 ) );
        CHECK_EQUAL(   1, grid.getIndex( pos2 ) );
        CHECK_EQUAL(   2, grid.getIndex( pos3 ) );
        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, grid.getIndex( pos4 ) );

        pos1 = Position_t(  0.5,  0.0, 9.5 );
        pos2 = Position_t(  1.5,  0.0, 9.5 );
        pos3 = Position_t(  2.5,  0.0, 9.5 );
        pos4 = Position_t(  3.5,  0.0, 9.5 );

        CHECK_EQUAL(   3, grid.getIndex( pos1 ) );
        CHECK_EQUAL(   4, grid.getIndex( pos2 ) );
        CHECK_EQUAL(   5, grid.getIndex( pos3 ) );
        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, grid.getIndex( pos4 ) );

        pos1 = Position_t(  0.0,  0.5, 9.5 );
        pos2 = Position_t(  0.0,  1.5, 9.5 );
        pos3 = Position_t(  0.0,  2.5, 9.5 );
        pos4 = Position_t(  0.0,  3.5, 9.5 );

        CHECK_EQUAL(   3, grid.getIndex( pos1 ) );
        CHECK_EQUAL(   4, grid.getIndex( pos2 ) );
        CHECK_EQUAL(   5, grid.getIndex( pos3 ) );
        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, grid.getIndex( pos4 ) );

        pos1 = Position_t(  0.0,  0.5, 10.5 );
        pos2 = Position_t(  0.0,  1.5, 10.5 );
        pos3 = Position_t(  0.0,  2.5, 10.5 );
        pos4 = Position_t(  0.0,  3.5, 10.5 );

        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, grid.getIndex( pos1 ) );
        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, grid.getIndex( pos2 ) );
        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, grid.getIndex( pos3 ) );
        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, grid.getIndex( pos4 ) );
    }

    TEST( getVolume_byIndex ){
        Grid_t grid;
        grid.setDimension( 2 );
        grid.setCoordinateSystem( TransportMeshTypeEnum::Cylindrical );

        std::vector<double> Rvertices = { 1.0, 2.0, 3.0 };
        std::vector<double> Zvertices = { -10,  0,  1.0, 3.0, 10 };

        grid.setGrid( SpatialGrid::CYLR_R, Rvertices);
        grid.setGrid( SpatialGrid::CYLR_Z, Zvertices);

        grid.initialize();

        double pi =  MonteRay::pi;
        CHECK_CLOSE( pi*std::pow(1.0,2)*10.0, grid.getVolume(0), 1e-6);
        CHECK_CLOSE( pi*(std::pow(2.0,2)-std::pow(1.0,2))*10.0, grid.getVolume(1), 1e-5);
        CHECK_CLOSE( pi*(std::pow(3.0,2)-std::pow(2.0,2))*10.0, grid.getVolume(2), 1e-5);

        CHECK_CLOSE( pi*std::pow(1.0,2)*1.0, grid.getVolume(3), 1e-6);
        CHECK_CLOSE( pi*(std::pow(2.0,2)-std::pow(1.0,2))*1.0, grid.getVolume(4), 1e-6);
        CHECK_CLOSE( pi*(std::pow(3.0,2)-std::pow(2.0,2))*1.0, grid.getVolume(5), 1e-6);

        CHECK_CLOSE( pi*std::pow(1.0,2)*2.0, grid.getVolume(6), 1e-6);
        CHECK_CLOSE( pi*(std::pow(2.0,2)-std::pow(1.0,2))*2.0, grid.getVolume(7), 1e-6);
        CHECK_CLOSE( pi*(std::pow(3.0,2)-std::pow(2.0,2))*2.0, grid.getVolume(8), 1e-6);

        CHECK_CLOSE( pi*std::pow(1.0,2)*7.0, grid.getVolume(9), 1e-6);
        CHECK_CLOSE( pi*(std::pow(2.0,2)-std::pow(1.0,2))*7.0, grid.getVolume(10), 1e-5);
        CHECK_CLOSE( pi*(std::pow(3.0,2)-std::pow(2.0,2))*7.0, grid.getVolume(11), 1e-5);
    }

    class particle {
    public:
        particle(void){};
        SpatialGrid::Position_t pos;
        SpatialGrid::Position_t dir;
        CUDA_CALLABLE_MEMBER SpatialGrid::Position_t getPosition(void) const { return pos; }
        CUDA_CALLABLE_MEMBER SpatialGrid::Position_t getDirection(void) const { return dir; }
    };

    TEST( getIndex_particle ){
        Grid_t grid;
        grid.setDimension( 2 );
        grid.setCoordinateSystem( TransportMeshTypeEnum::Cylindrical );

        std::vector<double> Rvertices = { 1.0, 2.0, 3.0 };
        std::vector<double> Zvertices = { -10,  0,  10 };

        grid.setGrid( SpatialGrid::CYLR_R, Rvertices);
        grid.setGrid( SpatialGrid::CYLR_Z, Zvertices);

        grid.initialize();
        particle p;

        Position_t pos1(  0.5,  0.0, -9.5 );
        Position_t pos2(  1.5,  0.0, -9.5 );
        Position_t pos3(  2.5,  0.0, -9.5 );
        Position_t pos4(  3.5,  0.0, -9.5 );

        p.pos = pos1;
        CHECK_EQUAL(   0, grid.getIndex( p ) );
        p.pos = pos2;
        CHECK_EQUAL(   1, grid.getIndex( p ) );
        p.pos = pos3;
        CHECK_EQUAL(   2, grid.getIndex( p ) );
        p.pos = pos4;
        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, grid.getIndex( p ) );

        pos1 = Position_t(  0.5,  0.0, 9.5 );
        pos2 = Position_t(  1.5,  0.0, 9.5 );
        pos3 = Position_t(  2.5,  0.0, 9.5 );
        pos4 = Position_t(  3.5,  0.0, 9.5 );

        p.pos = pos1;
        CHECK_EQUAL(   3, grid.getIndex( p ) );
        p.pos = pos2;
        CHECK_EQUAL(   4, grid.getIndex( p ) );
        p.pos = pos3;
        CHECK_EQUAL(   5, grid.getIndex( p ) );
        p.pos = pos4;
        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, grid.getIndex( p ) );

        pos1 = Position_t(  0.0,  0.5, 9.5 );
        pos2 = Position_t(  0.0,  1.5, 9.5 );
        pos3 = Position_t(  0.0,  2.5, 9.5 );
        pos4 = Position_t(  0.0,  3.5, 9.5 );

        p.pos = pos1;
        CHECK_EQUAL(   3, grid.getIndex( p ) );
        p.pos = pos2;
        CHECK_EQUAL(   4, grid.getIndex( p ) );
        p.pos = pos3;
        CHECK_EQUAL(   5, grid.getIndex( p ) );
        p.pos = pos4;
        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, grid.getIndex( p ) );

        pos1 = Position_t(  0.0,  0.5, 10.5 );
        pos2 = Position_t(  0.0,  1.5, 10.5 );
        pos3 = Position_t(  0.0,  2.5, 10.5 );
        pos4 = Position_t(  0.0,  3.5, 10.5 );

        p.pos = pos1;
        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, grid.getIndex( p ) );
        p.pos = pos2;
        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, grid.getIndex( p ) );
        p.pos = pos3;
        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, grid.getIndex( p ) );
        p.pos = pos4;
        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, grid.getIndex( p ) );

    }


    TEST( getIndex_particle_afterReadFromSerialization ){
        Grid_t grid;
        grid.setDimension( 2 );
        grid.setCoordinateSystem( TransportMeshTypeEnum::Cylindrical );

        std::vector<double> Rvertices = { 1.0, 2.0, 3.0 };
        std::vector<double> Zvertices = { -10,  0,  10 };

        grid.setGrid( SpatialGrid::CYLR_R, Rvertices);
        grid.setGrid( SpatialGrid::CYLR_Z, Zvertices);

        grid.initialize();
        grid.write( "spatialgrid_cylindrical_test_1.bin" );

        {
            Grid_t newGrid;
            newGrid.read( "spatialgrid_cylindrical_test_1.bin" );

            particle p;

            Position_t pos1(  0.5,  0.0, -9.5 );
            Position_t pos2(  1.5,  0.0, -9.5 );
            Position_t pos3(  2.5,  0.0, -9.5 );
            Position_t pos4(  3.5,  0.0, -9.5 );

            p.pos = pos1;
            CHECK_EQUAL(   0, newGrid.getIndex( p ) );
            p.pos = pos2;
            CHECK_EQUAL(   1, newGrid.getIndex( p ) );
            p.pos = pos3;
            CHECK_EQUAL(   2, newGrid.getIndex( p ) );
            p.pos = pos4;
            CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, newGrid.getIndex( p ) );

            pos1 = Position_t(  0.5,  0.0, 9.5 );
            pos2 = Position_t(  1.5,  0.0, 9.5 );
            pos3 = Position_t(  2.5,  0.0, 9.5 );
            pos4 = Position_t(  3.5,  0.0, 9.5 );

            p.pos = pos1;
            CHECK_EQUAL(   3, newGrid.getIndex( p ) );
            p.pos = pos2;
            CHECK_EQUAL(   4, newGrid.getIndex( p ) );
            p.pos = pos3;
            CHECK_EQUAL(   5, newGrid.getIndex( p ) );
            p.pos = pos4;
            CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, newGrid.getIndex( p ) );

            pos1 = Position_t(  0.0,  0.5, 9.5 );
            pos2 = Position_t(  0.0,  1.5, 9.5 );
            pos3 = Position_t(  0.0,  2.5, 9.5 );
            pos4 = Position_t(  0.0,  3.5, 9.5 );

            p.pos = pos1;
            CHECK_EQUAL(   3, newGrid.getIndex( p ) );
            p.pos = pos2;
            CHECK_EQUAL(   4, newGrid.getIndex( p ) );
            p.pos = pos3;
            CHECK_EQUAL(   5, newGrid.getIndex( p ) );
            p.pos = pos4;
            CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, newGrid.getIndex( p ) );

            pos1 = Position_t(  0.0,  0.5, 10.5 );
            pos2 = Position_t(  0.0,  1.5, 10.5 );
            pos3 = Position_t(  0.0,  2.5, 10.5 );
            pos4 = Position_t(  0.0,  3.5, 10.5 );

            p.pos = pos1;
            CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, newGrid.getIndex( p ) );
            p.pos = pos2;
            CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, newGrid.getIndex( p ) );
            p.pos = pos3;
            CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, newGrid.getIndex( p ) );
            p.pos = pos4;
            CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, newGrid.getIndex( p ) );
        }

    }

    TEST( rayTrace_in_ZDir_outside_to_outside ) {
        Grid_t grid;
        grid.setDimension( 2 );
        grid.setCoordinateSystem( TransportMeshTypeEnum::Cylindrical );

        std::vector<double> Rvertices = { 1.0, 2.0, 3.0, 5.0 };
        std::vector<double> Zvertices = { -6, -3, -1, 1, 3, 6 };

        grid.setGrid( SpatialGrid::CYLR_R, Rvertices);
        grid.setGrid( SpatialGrid::CYLR_Z, Zvertices);

        grid.initialize();

        Position_t position ( 0.0, 0.0, -7.5 );
        Position_t direction(   0,   0,    1 );
        double distance = 100.0;

        rayTraceList_t distances;
        grid.rayTrace(distances, position, direction, distance);

         CHECK_EQUAL(   5,  distances.size() );
         CHECK_EQUAL(   0,  distances.id(0) );
         CHECK_CLOSE( 3.0,  distances.dist(0), 1e-11 );
         CHECK_EQUAL(   4,  distances.id(1) );
         CHECK_CLOSE( 2.0,  distances.dist(1), 1e-11 );
         CHECK_EQUAL(   8,  distances.id(2) );
         CHECK_CLOSE( 2.0,  distances.dist(2), 1e-11 );
         CHECK_EQUAL(  12,  distances.id(3) );
         CHECK_CLOSE( 2.0,  distances.dist(3), 1e-11 );
         CHECK_EQUAL(  16,  distances.id(4) );
         CHECK_CLOSE( 3.0,  distances.dist(4), 1e-11 );
    }

    TEST( rayTrace_in_ZDir_outside_to_outside_along_radial_vertex ) {
        Grid_t grid;
        grid.setDimension( 2 );
        grid.setCoordinateSystem( TransportMeshTypeEnum::Cylindrical );

        std::vector<double> Rvertices = { 1.0, 2.0, 3.0, 5.0 };
        std::vector<double> Zvertices = { -6, -3, -1, 1, 3, 6 };

        grid.setGrid( SpatialGrid::CYLR_R, Rvertices);
        grid.setGrid( SpatialGrid::CYLR_Z, Zvertices);

        grid.initialize();

        Position_t position ( 1.0, 0.0, -7.5 );
        Position_t direction(   0,   0,    1 );
        double distance = 100.0;

        rayTraceList_t distances;
        grid.rayTrace(distances, position, direction, distance);

        CHECK_EQUAL(   5,  distances.size() );
        CHECK_EQUAL(   1,  distances.id(0) );
        CHECK_CLOSE( 3.0,  distances.dist(0), 1e-11 );
        CHECK_EQUAL(   5,  distances.id(1) );
        CHECK_CLOSE( 2.0,  distances.dist(1), 1e-11 );
        CHECK_EQUAL(   9,  distances.id(2) );
        CHECK_CLOSE( 2.0,  distances.dist(2), 1e-11 );
        CHECK_EQUAL(  13,  distances.id(3) );
        CHECK_CLOSE( 2.0,  distances.dist(3), 1e-11 );
        CHECK_EQUAL(  17,  distances.id(4) );
        CHECK_CLOSE( 3.0,  distances.dist(4), 1e-11 );
    }

    TEST( rayTrace_in_RDir_outside_to_outside ) {
        Grid_t grid;
        grid.setDimension( 2 );
        grid.setCoordinateSystem( TransportMeshTypeEnum::Cylindrical );

        std::vector<double> Rvertices = { 1.0, 2.0, 3.0, 5.0 };
        std::vector<double> Zvertices = { -6, -3, -1, 1, 3, 6 };

        grid.setGrid( SpatialGrid::CYLR_R, Rvertices);
        grid.setGrid( SpatialGrid::CYLR_Z, Zvertices);

        grid.initialize();

        Position_t position ( -6.0, 0.0,  -5.0 );
        Position_t direction(   1,   0,    0 );
        double distance = 100.0;

        rayTraceList_t distances;
        grid.rayTrace(distances, position, direction, distance);

        CHECK_EQUAL(   7,  distances.size() );
        CHECK_EQUAL(   3,  distances.id(0) );
        CHECK_CLOSE( 2.0,  distances.dist(0), 1e-11 );
        CHECK_EQUAL(   2,  distances.id(1) );
        CHECK_CLOSE( 1.0,  distances.dist(1), 1e-11 );
        CHECK_EQUAL(   1,  distances.id(2) );
        CHECK_CLOSE( 1.0,  distances.dist(2), 1e-11 );
        CHECK_EQUAL(   0,  distances.id(3) );
        CHECK_CLOSE( 2.0,  distances.dist(3), 1e-11 );
        CHECK_EQUAL(   1,  distances.id(4) );
        CHECK_CLOSE( 1.0,  distances.dist(4), 1e-11 );
        CHECK_EQUAL(   2,  distances.id(5) );
        CHECK_CLOSE( 1.0,  distances.dist(5), 1e-11 );
        CHECK_EQUAL(   3,  distances.id(6) );
        CHECK_CLOSE( 2.0,  distances.dist(6), 1e-11 );
    }

    TEST( rayTrace_in_RZDir_outside_to_outside_at_45degrees_thru_a_corner ) {
        // !!! DO NOT CHANGE WITHOUT DRAWING A PICTURE

        Grid_t grid;
        grid.setDimension( 2 );
        grid.setCoordinateSystem( TransportMeshTypeEnum::Cylindrical );

        std::vector<double> Rvertices = { 1.0, 2.0, 3.0, 5.0 };
        std::vector<double> Zvertices = { -6, -3, -1, 1, 3, 6 };

        grid.setGrid( SpatialGrid::CYLR_R, Rvertices);
        grid.setGrid( SpatialGrid::CYLR_Z, Zvertices);

        grid.initialize();

        Position_t position ( -6.0, 0.0,  -7.0 );
        Position_t direction(   1,   0,    1 );
        direction.normalize();
        double distance = 100.0;

        rayTraceList_t distances;
        grid.rayTrace(distances, position, direction, distance);

        CHECK_EQUAL(  11,  distances.size() );

        // expectedIndex[2] can be 1 or 6
#if RAY_DOUBLEPRECISION < 1
        std::vector<unsigned> expectedIndex = { 3, 2, 6, 5, 4, 8, 9, 10, 14, 15, 19 };
#else
        std::vector<unsigned> expectedIndex = { 3, 2, 1, 5, 4, 8, 9, 10, 14, 15, 19 };
#endif
        double s2 = std::sqrt(2.0);

        std::vector<double> expectedDistance = {2.0*s2, s2, 0.0, s2, s2, s2, s2, 0.0, s2, s2, s2, s2 };

        for( auto i=0; i<distances.size(); ++i ) {
            CHECK_EQUAL( expectedIndex   [i], distances.id(i) );
            CHECK_CLOSE( expectedDistance[i], distances.dist(i), 1e-5 );
        }
    }

    TEST( rayTrace_in_RZDir_inside_to_outside_at_45degrees_thru_a_corner ) {
        // !!! DO NOT CHANGE WITHOUT DRAWING A PICTURE

        Grid_t grid;
        grid.setDimension( 2 );
        grid.setCoordinateSystem( TransportMeshTypeEnum::Cylindrical );

        std::vector<double> Rvertices = { 1.0, 2.0, 3.0, 5.0 };
        std::vector<double> Zvertices = { -6, -3, -1, 1, 3, 6 };

        grid.setGrid( SpatialGrid::CYLR_R, Rvertices);
        grid.setGrid( SpatialGrid::CYLR_Z, Zvertices);

        grid.initialize();

        Position_t position ( -2.0, 0.0,  -3.0 );
        Position_t direction(   -1,   0,    -1 );
        direction.normalize();
        double distance = 100.0;

        rayTraceList_t distances;
        grid.rayTrace(distances, position, direction, distance);

        CHECK_EQUAL(   3,            distances.size() );
        CHECK_EQUAL(   6,            distances.id(0)  );
        CHECK_CLOSE( 0.0,            distances.dist(0), 1e-6 );
        CHECK_EQUAL(   2,            distances.id(1)  );
        CHECK_CLOSE( 1.0*sqrt(2.0),  distances.dist(1), 1e-6 );
        CHECK_EQUAL(   3,            distances.id(2)  );
        CHECK_CLOSE( 2.0*sqrt(2.0),  distances.dist(2), 1e-6 );
    }

    TEST( read_lnk3dnt ) {
        MonteRay_ReadLnk3dnt readerObject( "lnk3dnt/zeus2.lnk3dnt" );
        Grid_t grid(readerObject);

        CHECK_EQUAL( TransportMeshTypeEnum::Cylindrical, grid.getCoordinateSystem() );
        CHECK_EQUAL( 28, grid.getNumGridBins(MonteRay_SpatialGrid::CYLR_R) );
        CHECK_EQUAL( 34, grid.getNumGridBins(MonteRay_SpatialGrid::CYLR_Z) );
    }
}

} /* end namespace */
