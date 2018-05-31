#include <UnitTest++.h>

#include "MonteRay_SpatialGrid.hh"
#include "MonteRayDefinitions.hh"
#include "MonteRayConstants.hh"

#include <stdexcept>
#include <fstream>
#include <vector>

#include "MonteRayVector3D.hh"


namespace MonteRay_SpatialGrid_Cartesian_test{

using namespace MonteRay;

SUITE( MonteRay_SpatialGrid_Spherical_tests ) {
    typedef MonteRay_SpatialGrid Grid_t;

    TEST( set_Vertices ){
//        CHECK(false);
        std::vector<gpuRayFloat_t> vertices= {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10};

        Grid_t grid;

        grid.setGrid( MonteRay_SpatialGrid::SPH_R, vertices);
        CHECK_EQUAL(  10, grid.getNumGridBins(MonteRay_SpatialGrid::SPH_R) );
        CHECK_CLOSE( 0.0, grid.getMinVertex(MonteRay_SpatialGrid::SPH_R), 1e-11 );
        CHECK_CLOSE(10.0, grid.getMaxVertex(MonteRay_SpatialGrid::SPH_R), 1e-11 );
    }

    TEST( write_test_Vertices ){
        Grid_t grid;
        grid.setCoordinateSystem( TransportMeshTypeEnum::Spherical );
        grid.setDimension( 1 );

        std::vector<gpuRayFloat_t> vertices= {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10};
        grid.setGrid( MonteRay_SpatialGrid::SPH_R, vertices);
        bool exception=false;
        try{
        	grid.setGrid( 1, vertices);
        }
        catch( ... ) {
        	exception=true;
        }
        CHECK_EQUAL(true, exception );
        exception=false;
        try{
        	grid.setGrid( 2, vertices);
        }
        catch( ... ) {
        	exception=true;
        }
        CHECK_EQUAL(true, exception );


        grid.write( "spatialgrid_cartesian_test_vertices_1.bin" );

        {
            // read class state from archive
            Grid_t newGrid;
            newGrid.read( "spatialgrid_cartesian_test_vertices_1.bin" );

            CHECK_EQUAL( 10, newGrid.getNumGridBins(MonteRay_SpatialGrid::SPH_R) );
        }
    }

    TEST( isInitialized ){
        Grid_t grid;
        grid.setDimension( 1 );
        grid.setCoordinateSystem( TransportMeshTypeEnum::Spherical );

        std::vector<gpuRayFloat_t> Rvertices = { 0, 1, 10 };

        grid.setGrid( MonteRay_SpatialGrid::SPH_R, Rvertices);

        CHECK_EQUAL( false, grid.isInitialized() );
        grid.initialize();
        CHECK_EQUAL( true, grid.isInitialized() );
    }

    TEST( initialize_check_vertices_set ){
        Grid_t grid;
        grid.setDimension( 1 );
        grid.setCoordinateSystem( TransportMeshTypeEnum::Spherical );

        bool exception=false;
        try{
            grid.initialize();
        }
        catch( ... ) {
        	exception=true;
        }
        CHECK_EQUAL(true, exception );
    }

    TEST( getIndexByPos ){
         Grid_t grid;
         grid.setDimension( 1 );
         grid.setCoordinateSystem( TransportMeshTypeEnum::Spherical );

         std::vector<gpuRayFloat_t> Rvertices = { 0, 1, 10 };

         grid.setGrid( MonteRay_SpatialGrid::SPH_R, Rvertices);

         grid.initialize();

         MonteRay_SpatialGrid::Position_t pos1( 0.5, 0.5, 0.5 );
         MonteRay_SpatialGrid::Position_t pos2( 5.0, 5.0, 5.0 );

         CHECK_EQUAL(   0, grid.getIndex( pos1 ) );
         CHECK_EQUAL(   1, grid.getIndex( pos2 ) );
     }

    TEST( getVolume_byIndex ){
        Grid_t grid;
        grid.setDimension( 1 );
        grid.setCoordinateSystem( TransportMeshTypeEnum::Spherical );

        std::vector<gpuRayFloat_t> Rvertices = { 0, 1, 10 };

        grid.setGrid( MonteRay_SpatialGrid::SPH_R, Rvertices);

        grid.initialize();

        CHECK_CLOSE( 4.0*pi/3.0, grid.getVolume(0), 1e-11);
        CHECK_CLOSE(  (4.0*pi/3.0)*(1000.0 - 1.0), grid.getVolume(1), 1e-11);
    }

    class particle {
    public:
    	CUDA_CALLABLE_MEMBER particle(void){};

        MonteRay_SpatialGrid::Position_t pos;
        MonteRay_SpatialGrid::Position_t dir;

        CUDA_CALLABLE_MEMBER
        MonteRay_SpatialGrid::Position_t getPosition(void) const { return pos; }

        CUDA_CALLABLE_MEMBER
        MonteRay_SpatialGrid::Position_t getDirection(void) const { return dir; }
    };

    TEST( getIndex_particle ){
        Grid_t grid;
        grid.setDimension( 1 );
        grid.setCoordinateSystem( TransportMeshTypeEnum::Spherical );

        std::vector<gpuRayFloat_t> Rvertices = { 0, 1, 10 };

        grid.setGrid( MonteRay_SpatialGrid::SPH_R, Rvertices);

        grid.initialize();

        particle p;

        MonteRay_SpatialGrid::Position_t pos1( 0.5, 0.5, 0.5 );
        MonteRay_SpatialGrid::Position_t pos2( 5.0, 5.0, 5.0 );

        p.pos = pos1;
        CHECK_EQUAL(   0, grid.getIndex( p ) );
        p.pos = pos2;
        CHECK_EQUAL(   1, grid.getIndex( p ) );
    }

    TEST( getIndex_particle_afterReadFromSerialization ){
        Grid_t grid;
        grid.setDimension( 1 );
        grid.setCoordinateSystem( TransportMeshTypeEnum::Spherical );

        std::vector<gpuRayFloat_t> Rvertices = {1.0, 2.0, 3.0};

        grid.setGrid( MonteRay_SpatialGrid::SPH_R, Rvertices);

        grid.initialize();

    	grid.write( "spatialgrid_spherical_test_2.bin" );

        {
            // read class state from archive
            Grid_t newGrid;
    		newGrid.read( "spatialgrid_spherical_test_2.bin" );

            particle p;

            MonteRay_SpatialGrid::Position_t pos1(  0.5,  0.0,  0.0 );
            MonteRay_SpatialGrid::Position_t pos2(  1.5,  0.0,  0.0 );
            MonteRay_SpatialGrid::Position_t pos3(  2.5,  0.0,  0.0 );
            MonteRay_SpatialGrid::Position_t pos4(  3.5,  0.0,  0.0 );

            p.pos = pos1;
            CHECK_EQUAL(   0, newGrid.getIndex( p ) );
            p.pos = pos2;
            CHECK_EQUAL(   1, newGrid.getIndex( p ) );
            p.pos = pos3;
            CHECK_EQUAL(   2, newGrid.getIndex( p ) );
            p.pos = pos4;
            CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, newGrid.getIndex( p ) );


            pos1 = MonteRay_SpatialGrid::Position_t(  0.0,  0.5, 0.0 );
            pos2 = MonteRay_SpatialGrid::Position_t(  0.0,  1.5, 0.0 );
            pos3 = MonteRay_SpatialGrid::Position_t(  0.0,  2.5, 0.0 );
            pos4 = MonteRay_SpatialGrid::Position_t(  0.0,  3.5, 0.0 );

            p.pos = pos1;
            CHECK_EQUAL(   0, newGrid.getIndex( p ) );
            p.pos = pos2;
            CHECK_EQUAL(   1, newGrid.getIndex( p ) );
            p.pos = pos3;
            CHECK_EQUAL(   2, newGrid.getIndex( p ) );
            p.pos = pos4;
            CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, newGrid.getIndex( p ) );

        }
    }

    TEST( rayTrace_in_XDir_outside_to_outside ) {
        Grid_t grid;
        grid.setDimension( 1 );
        grid.setCoordinateSystem( TransportMeshTypeEnum::Spherical );

        std::vector<gpuRayFloat_t> Rvertices = {1.0, 2.0, 3.0, 5.0};

        grid.setGrid( MonteRay_SpatialGrid::SPH_R, Rvertices);

        grid.initialize();

        MonteRay_SpatialGrid::Position_t position ( -6.0,  0.0,  0.0 );
        MonteRay_SpatialGrid::Position_t direction(   1,   0,    0 );
        gpuRayFloat_t distance = 100.0;

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
    TEST( rayTrace_in_XZDir_outside_to_outside_at_45degrees ) {
        Grid_t grid;
        grid.setDimension( 1 );
        grid.setCoordinateSystem( TransportMeshTypeEnum::Spherical );

        std::vector<gpuRayFloat_t> Rvertices = {1.0, 2.0, 3.0, 5.0};

        grid.setGrid( MonteRay_SpatialGrid::SPH_R, Rvertices);

        grid.initialize();

        MonteRay_SpatialGrid::Position_t position ( -6.0, 0.0,  -6.0 );
        MonteRay_SpatialGrid::Position_t direction(   1,   0,    1 );
        direction.normalize();
        gpuRayFloat_t distance = 100.0;

        rayTraceList_t distances;
        grid.rayTrace(distances, position, direction, distance);

        CHECK_EQUAL(   7,  distances.size() );
        CHECK_EQUAL(   3,  distances.id(0) );
        CHECK_CLOSE( 2.0,  distances.dist(0), 1e-5 );
        CHECK_EQUAL(   2,  distances.id(1) );
        CHECK_CLOSE( 1.0,  distances.dist(1), 1e-5 );
        CHECK_EQUAL(   1,  distances.id(2) );
        CHECK_CLOSE( 1.0,  distances.dist(2), 1e-5 );
        CHECK_EQUAL(   0,  distances.id(3) );
        CHECK_CLOSE( 2.0,  distances.dist(3), 1e-5 );
        CHECK_EQUAL(   1,  distances.id(4) );
        CHECK_CLOSE( 1.0,  distances.dist(4), 1e-5 );
        CHECK_EQUAL(   2,  distances.id(5) );
        CHECK_CLOSE( 1.0,  distances.dist(5), 1e-5 );
        CHECK_EQUAL(   3,  distances.id(6) );
        CHECK_CLOSE( 2.0,  distances.dist(6), 1e-5 );
    }
    TEST( rayTrace_in_XZDir_inside_to_outside_at_45degrees ) {
        Grid_t grid;
        grid.setDimension( 1 );
        grid.setCoordinateSystem( TransportMeshTypeEnum::Spherical );

        std::vector<gpuRayFloat_t> Rvertices = {1.0, 2.0, 3.0, 5.0};

        grid.setGrid( MonteRay_SpatialGrid::SPH_R, Rvertices);

        grid.initialize();

        MonteRay_SpatialGrid::Position_t position ( -2.0, 0.0,  -2.0 );
        MonteRay_SpatialGrid::Position_t direction(   -1,   0,    -1 );
        direction.normalize();
        gpuRayFloat_t distance = 100.0;

        rayTraceList_t distances;
        grid.rayTrace(distances, position, direction, distance);

        CHECK_EQUAL(   2,  distances.size() );
        CHECK_EQUAL(   2,  distances.id(0) );
        CHECK_CLOSE( 0.171572875254,  distances.dist(0), 1e-6 );
        CHECK_EQUAL(   3,  distances.id(1) );
        CHECK_CLOSE( 2.0,  distances.dist(1), 1e-6 );
    }

}

} /* end namespace */
