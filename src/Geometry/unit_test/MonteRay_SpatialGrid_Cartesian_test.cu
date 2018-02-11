#include <UnitTest++.h>

#include "MonteRay_SpatialGrid.hh"
#include "MonteRayDefinitions.hh"

#include <stdexcept>
#include <fstream>
#include <vector>

#include "MonteRayVector3D.hh"


namespace MonteRay_SpatialGrid_Cartesian_test{

using namespace MonteRay;

SUITE( MonteRay_SpatialGrid_Cartesian_tests ) {
    typedef MonteRay_SpatialGrid Grid_t;

    TEST( set_Vertices ){
        //CHECK(false);
        std::vector<gpuFloatType_t> vertices= {
        		    -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
                      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10};

        Grid_t grid;

        grid.setGrid( MonteRay_SpatialGrid::CART_X, vertices);
        CHECK_EQUAL(   20, grid.getNumGridBins(MonteRay_SpatialGrid::CART_X) );
        CHECK_CLOSE(-10.0, grid.getMinVertex(MonteRay_SpatialGrid::CART_X), 1e-11 );
        CHECK_CLOSE( 10.0, grid.getMaxVertex(MonteRay_SpatialGrid::CART_X), 1e-11 );
    }

    TEST( write_test_Vertices ){
        Grid_t grid;
        grid.setCoordinateSystem( TransportMeshTypeEnum::Cartesian );
        grid.setDimension( 3 );

        std::vector<gpuFloatType_t> vertices= {
                		    -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
                              0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10};
        grid.setGrid( MonteRay_SpatialGrid::CART_X, vertices);
        grid.setGrid( MonteRay_SpatialGrid::CART_Y, vertices);
        grid.setGrid( MonteRay_SpatialGrid::CART_Z, vertices);

        grid.write( "spatialgrid_cartesian_test_vertices_1.bin" );

        {
            // read class state from archive
            Grid_t newGrid;
            newGrid.read( "spatialgrid_cartesian_test_vertices_1.bin" );

            CHECK_EQUAL(    20, newGrid.getNumGridBins(MonteRay_SpatialGrid::CART_X) );
            CHECK_CLOSE( -10.0, newGrid.getMinVertex(MonteRay_SpatialGrid::CART_X), 1e-11 );
            CHECK_CLOSE(  10.0, newGrid.getMaxVertex(MonteRay_SpatialGrid::CART_X), 1e-11 );
        }
    }

    TEST( set_xyz_Vertices ){
        Grid_t grid;
        grid.setCoordinateSystem( TransportMeshTypeEnum::Cartesian );

        std::vector<gpuFloatType_t> Xvertices = { -10, -1, 1, 10 };
        std::vector<gpuFloatType_t> Yvertices = { -20, -1, 0, 1, 10 };
        std::vector<gpuFloatType_t> Zvertices = { -30, -2, -1, 0, 1, 2, 30 };

        grid.setGrid( MonteRay_SpatialGrid::CART_X, Xvertices);
        grid.setGrid( MonteRay_SpatialGrid::CART_Y, Yvertices);
        grid.setGrid( MonteRay_SpatialGrid::CART_Z, Zvertices);
        CHECK_EQUAL( 3, grid.getNumGridBins(MonteRay_SpatialGrid::CART_X) );
        CHECK_CLOSE( -10.0, grid.getMinVertex(MonteRay_SpatialGrid::CART_X), 1e-11 );
        CHECK_CLOSE( 10.0, grid.getMaxVertex(MonteRay_SpatialGrid::CART_X), 1e-11 );

        CHECK_EQUAL( 4, grid.getNumGridBins(MonteRay_SpatialGrid::CART_Y) );
        CHECK_CLOSE( -20.0, grid.getMinVertex(MonteRay_SpatialGrid::CART_Y), 1e-11 );
        CHECK_CLOSE( 10.0, grid.getMaxVertex(MonteRay_SpatialGrid::CART_Y), 1e-11 );

        CHECK_EQUAL( 6, grid.getNumGridBins(MonteRay_SpatialGrid::CART_Z) );
        CHECK_CLOSE( -30.0, grid.getMinVertex(MonteRay_SpatialGrid::CART_Z), 1e-11 );
        CHECK_CLOSE( 30.0, grid.getMaxVertex(MonteRay_SpatialGrid::CART_Z), 1e-11 );
    }

    TEST( isInitialized ){
        Grid_t grid;
        grid.setDimension( 3 );
        grid.setCoordinateSystem( TransportMeshTypeEnum::Cartesian );

        std::vector<gpuFloatType_t> Xvertices = { -10, -1, 1, 10 };
        std::vector<gpuFloatType_t> Yvertices = { -20, -1, 0, 1, 10 };
        std::vector<gpuFloatType_t> Zvertices = { -30, -2, -1, 0, 1, 2, 30 };

        grid.setGrid( MonteRay_SpatialGrid::CART_X, Xvertices);
        grid.setGrid( MonteRay_SpatialGrid::CART_Y, Yvertices);
        grid.setGrid( MonteRay_SpatialGrid::CART_Z, Zvertices);

        CHECK_EQUAL( false, grid.isInitialized() );
        grid.initialize();
        CHECK_EQUAL( true, grid.isInitialized() );
    }

    TEST( initialize_check_vertices_set ){
        Grid_t grid;
        grid.setDimension( 3 );
        grid.setCoordinateSystem( TransportMeshTypeEnum::Cartesian );

        std::vector<gpuFloatType_t> Xvertices = { -10, -1, 1, 10 };
        std::vector<gpuFloatType_t> Yvertices = { -20, -1, 0, 1, 10 };
        std::vector<gpuFloatType_t> Zvertices = { -30, -2, -1, 0, 1, 2, 30 };

        grid.setGrid( MonteRay_SpatialGrid::CART_X, Xvertices);
        grid.setGrid( MonteRay_SpatialGrid::CART_Y, Yvertices);
        //grid.setGrid( SpatialGrid::CART_Z, Zvertices);
        // removed Z vertices

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
         grid.setDimension( 3 );
         grid.setCoordinateSystem( TransportMeshTypeEnum::Cartesian );

         std::vector<gpuFloatType_t> Xvertices = { -10, -1, 1, 10 };
         std::vector<gpuFloatType_t> Yvertices = { -10, -1, 1, 10 };
         std::vector<gpuFloatType_t> Zvertices = { -10, -1, 1, 10 };

         grid.setGrid( MonteRay_SpatialGrid::CART_X, Xvertices);
         grid.setGrid( MonteRay_SpatialGrid::CART_Y, Yvertices);
         grid.setGrid( MonteRay_SpatialGrid::CART_Z, Zvertices);

         grid.initialize();

         MonteRay_SpatialGrid::Position_t pos1( -9.5, -9.5, -9.5 );
         MonteRay_SpatialGrid::Position_t pos2(  0.0, -9.5, -9.5 );
         MonteRay_SpatialGrid::Position_t pos3( -9.5,  0.0, -9.5 );
         MonteRay_SpatialGrid::Position_t pos4( -9.5, -9.5,  0.0 );
         MonteRay_SpatialGrid::Position_t pos5(  2.0,  2.0,  2.0 );

         CHECK_EQUAL(   0, grid.getIndex( pos1 ) );
         CHECK_EQUAL(   1, grid.getIndex( pos2 ) );
         CHECK_EQUAL(   3, grid.getIndex( pos3 ) );
         CHECK_EQUAL(   9, grid.getIndex( pos4 ) );
         CHECK_EQUAL(  26, grid.getIndex( pos5 ) );
     }

    TEST( getVolume_byIndex ){
        Grid_t grid;
        grid.setDimension( 3 );
        grid.setCoordinateSystem( TransportMeshTypeEnum::Cartesian );

        std::vector<gpuFloatType_t> Xvertices = { -10, -1, 1, 10 };
        std::vector<gpuFloatType_t> Yvertices = {  -2, -1, 1,  2 };
        std::vector<gpuFloatType_t> Zvertices = { -10, -1, 1, 20 };

        grid.setGrid( MonteRay_SpatialGrid::CART_X, Xvertices);
        grid.setGrid( MonteRay_SpatialGrid::CART_Y, Yvertices);
        grid.setGrid( MonteRay_SpatialGrid::CART_Z, Zvertices);

        grid.initialize();

        CHECK_CLOSE( 9.0*1.0*9.0, grid.getVolume(0), 1e-11);
        CHECK_CLOSE( 2.0*1.0*9.0, grid.getVolume(1), 1e-11);
        CHECK_CLOSE( 9.0*2.0*9.0, grid.getVolume(3), 1e-11);
        CHECK_CLOSE( 9.0*1.0*9.0, grid.getVolume(8), 1e-11);
        CHECK_CLOSE( 9.0*1.0*2.0, grid.getVolume(9), 1e-11);
        CHECK_CLOSE( 9.0*1.0*19.0, grid.getVolume(26), 1e-11);
    }

    class particle {
    public:
        particle(void){};
        MonteRay_SpatialGrid::Position_t pos;
        MonteRay_SpatialGrid::Position_t dir;
        MonteRay_SpatialGrid::Position_t getPosition(void) const { return pos; }
        MonteRay_SpatialGrid::Position_t getDirection(void) const { return dir; }
    };

    TEST( getIndex_particle ){
        Grid_t grid;
        grid.setDimension( 3 );
        grid.setCoordinateSystem( TransportMeshTypeEnum::Cartesian );

        std::vector<gpuFloatType_t> Xvertices = { -10, -1, 1, 10 };
        std::vector<gpuFloatType_t> Yvertices = { -10, -1, 1, 10 };
        std::vector<gpuFloatType_t> Zvertices = { -10, -1, 1, 10 };

        grid.setGrid( MonteRay_SpatialGrid::CART_X, Xvertices);
        grid.setGrid( MonteRay_SpatialGrid::CART_Y, Yvertices);
        grid.setGrid( MonteRay_SpatialGrid::CART_Z, Zvertices);

        grid.initialize();

        particle p;

        MonteRay_SpatialGrid::Position_t pos1( -9.5, -9.5, -9.5 );
        MonteRay_SpatialGrid::Position_t pos2(  0.0, -9.5, -9.5 );
        MonteRay_SpatialGrid::Position_t pos3( -9.5,  0.0, -9.5 );
        MonteRay_SpatialGrid::Position_t pos4( -9.5, -9.5,  0.0 );
        MonteRay_SpatialGrid::Position_t pos5(  2.0,  2.0,  2.0 );

        p.pos = pos1;
        CHECK_EQUAL(   0, grid.getIndex( p ) );
        p.pos = pos2;
        CHECK_EQUAL(   1, grid.getIndex( p ) );
        p.pos = pos3;
        CHECK_EQUAL(   3, grid.getIndex( p ) );
        p.pos = pos4;
        CHECK_EQUAL(   9, grid.getIndex( p ) );
        p.pos = pos5;
        CHECK_EQUAL(  26, grid.getIndex( p ) );
    }

    TEST( getIndex_particle_afterReadFromSerialization ){
    	Grid_t grid;
    	grid.setDimension( 3 );
    	grid.setCoordinateSystem( TransportMeshTypeEnum::Cartesian );

    	std::vector<gpuFloatType_t> Xvertices = { -10, -1, 1, 10 };
    	std::vector<gpuFloatType_t> Yvertices = { -10, -1, 1, 10 };
    	std::vector<gpuFloatType_t> Zvertices = { -10, -1, 1, 10 };

    	grid.setGrid( MonteRay_SpatialGrid::CART_X, Xvertices);
    	grid.setGrid( MonteRay_SpatialGrid::CART_Y, Yvertices);
    	grid.setGrid( MonteRay_SpatialGrid::CART_Z, Zvertices);

    	grid.initialize();

    	grid.write( "spatialgrid_cartesian_test_2.bin" );

    	{
    		Grid_t newGrid;
    		newGrid.read( "spatialgrid_cartesian_test_2.bin" );

    		particle p;

    		MonteRay_SpatialGrid::Position_t pos1( -9.5, -9.5, -9.5 );
    		MonteRay_SpatialGrid::Position_t pos2(  0.0, -9.5, -9.5 );
    		MonteRay_SpatialGrid::Position_t pos3( -9.5,  0.0, -9.5 );
    		MonteRay_SpatialGrid::Position_t pos4( -9.5, -9.5,  0.0 );
    		MonteRay_SpatialGrid::Position_t pos5(  2.0,  2.0,  2.0 );

    		p.pos = pos1;
    		CHECK_EQUAL(   0, newGrid.getIndex( p ) );
    		p.pos = pos2;
    		CHECK_EQUAL(   1, newGrid.getIndex( p ) );
    		p.pos = pos3;
    		CHECK_EQUAL(   3, newGrid.getIndex( p ) );
    		p.pos = pos4;
    		CHECK_EQUAL(   9, newGrid.getIndex( p ) );
    		p.pos = pos5;
    		CHECK_EQUAL(  26, newGrid.getIndex( p ) );
    	}

    }

    TEST( rayTrace_1D_external_to_internal_posX_pos_and_dir ) {
         Grid_t grid;
         grid.setDimension( 3 );
         grid.setCoordinateSystem( TransportMeshTypeEnum::Cartesian );

         std::vector<gpuFloatType_t> Xvertices = { -1, 0, 1 };
         std::vector<gpuFloatType_t> Yvertices = { -1, 0, 1 };
         std::vector<gpuFloatType_t> Zvertices = { -1, 0, 1 };

         grid.setGrid( MonteRay_SpatialGrid::CART_X, Xvertices);
         grid.setGrid( MonteRay_SpatialGrid::CART_Y, Yvertices);
         grid.setGrid( MonteRay_SpatialGrid::CART_Z, Zvertices);

         grid.initialize();

         Grid_t::Position_t position (  -1.5, -0.5, -0.5 );
         Grid_t::Position_t direction(    1,   0,    0 );
         direction.normalize();
         gpuFloatType_t distance = 2.0;

         rayTraceList_t distances = grid.rayTrace( position, direction, distance);

         CHECK_EQUAL( 2, distances.size() );
         CHECK_EQUAL( 0, distances.id(0) );
         CHECK_CLOSE( 1.0, distances.dist(0), 1e-11 );
         CHECK_EQUAL( 1, distances.id(1) );
         CHECK_CLOSE( 0.5, distances.dist(1), 1e-11 );
     }

    TEST( rayTrace_1D_external_to_internal_posX_particle ) {
        Grid_t grid;
        grid.setDimension( 3 );
        grid.setCoordinateSystem( TransportMeshTypeEnum::Cartesian );

        std::vector<gpuFloatType_t> Xvertices = { -1, 0, 1 };
        std::vector<gpuFloatType_t> Yvertices = { -1, 0, 1 };
        std::vector<gpuFloatType_t> Zvertices = { -1, 0, 1 };

        grid.setGrid( MonteRay_SpatialGrid::CART_X, Xvertices);
        grid.setGrid( MonteRay_SpatialGrid::CART_Y, Yvertices);
        grid.setGrid( MonteRay_SpatialGrid::CART_Z, Zvertices);

        grid.initialize();

        Grid_t::Position_t position (  -1.5, -0.5, -0.5 );
        Grid_t::Position_t direction(    1,   0,    0 );
        direction.normalize();
        gpuFloatType_t distance = 2.0;

        particle p;
        p.pos = position;
        p.dir = direction;

        rayTraceList_t distances = grid.rayTrace( p, distance);

        CHECK_EQUAL( 2, distances.size() );
        CHECK_EQUAL( 0, distances.id(0) );
        CHECK_CLOSE( 1.0, distances.dist(0), 1e-11 );
        CHECK_EQUAL( 1, distances.id(1) );
        CHECK_CLOSE( 0.5, distances.dist(1), 1e-11 );
    }

//    TEST( rayTrace_setRotation_1D_external_to_internal_posX_particle ) {
//        Grid_t grid;
//        grid.setDimension( 3 );
//        grid.setCoordinateSystem( TransportMeshTypeEnum::Cartesian );
//
//        std::vector<double> Xvertices, Yvertices, Zvertices;
//        Xvertices += -1, 0, 1;
//        Yvertices += -1, 0, 1;
//        Zvertices += -1, 0, 1;
//
//        grid.setGrid( SpatialGrid::CART_X, Xvertices);
//        grid.setGrid( SpatialGrid::CART_Y, Yvertices);
//        grid.setGrid( SpatialGrid::CART_Z, Zvertices);
//
//        Transformation transform1;
//        transform1.rotateAroundAxis( Transformation::Z_AXIS, 90.0 );
//        grid.setTransformation( transform1 );
//
//        grid.initialize();
//
//        Grid_t::Position_t position (  -1.5, -0.5, -0.5 );
//        Grid_t::Position_t direction(    1,   0,    0 );
//        direction.normalize();
//        double distance = 2.0;
//
//        particle p;
//        p.pos = position;
//        p.dir = direction;
//
//        Grid_t::rayTraceList_t distances = grid.rayTrace( p, distance);
//
//        CHECK_EQUAL( 2, distances.size() );
//        CHECK_CLOSE( 2, distances[0].first, 1e-11 );
//        CHECK_CLOSE( 1, distances[0].second, 1e-11 );
//        CHECK_CLOSE( 0, distances[1].first, 1e-11 );
//        CHECK_CLOSE( 0.5, distances[1].second, 1e-11 );
//    }

}

} /* end namespace */
