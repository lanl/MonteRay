#include <UnitTest++.h>

#include "MonteRay_SpatialGrid_GPU_helper.hh"

namespace MonteRay_SpatialGrid_Cartesian_GPU_tester {

using namespace MonteRay_SpatialGrid_helper;

SUITE( MonteRay_SpatialGrid_Cartesian_GPU_Tests ) {

   	TEST( setup ) {
   		gpuReset();
   	}

   	TEST_FIXTURE(SpatialGridGPUTester, set_Vertices ){
        //CHECK(false);
        std::vector<gpuRayFloat_t> vertices= {
        		    -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
                      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10};

        std::vector<gpuRayFloat_t> verticesZ= {
        		-10,  -1,  0,  1, 10};

        setCoordinateSystem( TransportMeshTypeEnum::Cartesian );
		setDimension( 3 );
        setGrid( MonteRay_SpatialGrid::CART_X, vertices);
        setGrid( MonteRay_SpatialGrid::CART_Y, vertices);
        setGrid( MonteRay_SpatialGrid::CART_Z, verticesZ);
        initialize();
        copyToGPU();

        CHECK_EQUAL(   20, getNumGridBins(MonteRay_SpatialGrid::CART_X) );
        CHECK_EQUAL(   20, getNumGridBins(MonteRay_SpatialGrid::CART_Y) );
        CHECK_EQUAL(   4, getNumGridBins(MonteRay_SpatialGrid::CART_Z) );
        CHECK_CLOSE(-10.0, getMinVertex(MonteRay_SpatialGrid::CART_X), 1e-6 );
        CHECK_CLOSE( 10.0, getMaxVertex(MonteRay_SpatialGrid::CART_X), 1e-6 );
    }

   	TEST_FIXTURE(SpatialGridGPUTester, read_test_Vertices_access_on_GPU ){
        Grid_t grid;
        grid.setCoordinateSystem( TransportMeshTypeEnum::Cartesian );
        grid.setDimension( 3 );

        std::vector<gpuRayFloat_t> vertices= {
                		    -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
                              0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10};
        grid.setGrid( MonteRay_SpatialGrid::CART_X, vertices);
        grid.setGrid( MonteRay_SpatialGrid::CART_Y, vertices);
        grid.setGrid( MonteRay_SpatialGrid::CART_Z, vertices);


        grid.initialize();
        CHECK_EQUAL( 21, grid.getNumVertices(0) );
        CHECK_EQUAL( 0, grid.getNumVerticesSq(0) );

        grid.write( "spatialgrid_cartesian_test_vertices_gpu_1.bin" );

        {
            // read class state from archive
            read( "spatialgrid_cartesian_test_vertices_gpu_1.bin" );
            CHECK_EQUAL( 21, pGridInfo->getNumVertices(0) );
            CHECK_EQUAL( 0, pGridInfo->getNumVerticesSq(0) );
            CHECK_EQUAL( 20, pGridInfo->getNumGridBins( MonteRay_SpatialGrid::CART_X ) );
            CHECK_EQUAL( 20, pGridInfo->getNumGridBins( MonteRay_SpatialGrid::CART_Y ) );
            copyToGPU();

            CHECK_EQUAL(    20, getNumGridBins(MonteRay_SpatialGrid::CART_X) );
            CHECK_CLOSE( -10.0,   getMinVertex(MonteRay_SpatialGrid::CART_X), 1e-6 );
            CHECK_CLOSE(  10.0,   getMaxVertex(MonteRay_SpatialGrid::CART_X), 1e-6 );
        }
    }

   	TEST_FIXTURE(SpatialGridGPUTester, set_xyz_Vertices ){
        setCoordinateSystem( TransportMeshTypeEnum::Cartesian );
        setDimension( 3 );

        std::vector<gpuRayFloat_t> Xvertices = { -10, -1, 1, 10 };
        std::vector<gpuRayFloat_t> Yvertices = { -20, -1, 0, 1, 10 };
        std::vector<gpuRayFloat_t> Zvertices = { -30, -2, -1, 0, 1, 2, 30 };

        setGrid( MonteRay_SpatialGrid::CART_X, Xvertices);
        setGrid( MonteRay_SpatialGrid::CART_Y, Yvertices);
        setGrid( MonteRay_SpatialGrid::CART_Z, Zvertices);
        initialize();
        copyToGPU();

        CHECK_EQUAL( 3, getNumGridBins(MonteRay_SpatialGrid::CART_X) );
        CHECK_CLOSE( -10.0, getMinVertex(MonteRay_SpatialGrid::CART_X), 1e-6 );
        CHECK_CLOSE( 10.0, getMaxVertex(MonteRay_SpatialGrid::CART_X), 1e-6 );

        CHECK_EQUAL( 4, getNumGridBins(MonteRay_SpatialGrid::CART_Y) );
        CHECK_CLOSE( -20.0, getMinVertex(MonteRay_SpatialGrid::CART_Y), 1e-6 );
        CHECK_CLOSE( 10.0, getMaxVertex(MonteRay_SpatialGrid::CART_Y), 1e-6 );

        CHECK_EQUAL( 6, getNumGridBins(MonteRay_SpatialGrid::CART_Z) );
        CHECK_CLOSE( -30.0, getMinVertex(MonteRay_SpatialGrid::CART_Z), 1e-6 );
        CHECK_CLOSE( 30.0, getMaxVertex(MonteRay_SpatialGrid::CART_Z), 1e-6 );
    }

   	TEST_FIXTURE(SpatialGridGPUTester,  isInitialized ){
        setDimension( 3 );
        setCoordinateSystem( TransportMeshTypeEnum::Cartesian );

        std::vector<gpuRayFloat_t> Xvertices = { -10, -1, 1, 10 };
        std::vector<gpuRayFloat_t> Yvertices = { -20, -1, 0, 1, 10 };
        std::vector<gpuRayFloat_t> Zvertices = { -30, -2, -1, 0, 1, 2, 30 };

        setGrid( MonteRay_SpatialGrid::CART_X, Xvertices);
        setGrid( MonteRay_SpatialGrid::CART_Y, Yvertices);
        setGrid( MonteRay_SpatialGrid::CART_Z, Zvertices);
        initialize();
        copyToGPU();

        CHECK_EQUAL( true, isInitialized() );
    }

   	TEST_FIXTURE(SpatialGridGPUTester, getIndexByPos ){
         setDimension( 3 );
         setCoordinateSystem( TransportMeshTypeEnum::Cartesian );

         std::vector<gpuRayFloat_t> Xvertices = { -10, -1, 1, 10 };
         std::vector<gpuRayFloat_t> Yvertices = { -10, -1, 1, 10 };
         std::vector<gpuRayFloat_t> Zvertices = { -10, -1, 1, 10 };

         setGrid( MonteRay_SpatialGrid::CART_X, Xvertices);
         setGrid( MonteRay_SpatialGrid::CART_Y, Yvertices);
         setGrid( MonteRay_SpatialGrid::CART_Z, Zvertices);

         initialize();
         copyToGPU();

         MonteRay_SpatialGrid::Position_t pos1( -9.5, -9.5, -9.5 );
         MonteRay_SpatialGrid::Position_t pos2(  0.0, -9.5, -9.5 );
         MonteRay_SpatialGrid::Position_t pos3( -9.5,  0.0, -9.5 );
         MonteRay_SpatialGrid::Position_t pos4( -9.5, -9.5,  0.0 );
         MonteRay_SpatialGrid::Position_t pos5(  2.0,  2.0,  2.0 );

         CHECK_EQUAL(   0, getIndex( pos1 ) );
         CHECK_EQUAL(   1, getIndex( pos2 ) );
         CHECK_EQUAL(   3, getIndex( pos3 ) );
         CHECK_EQUAL(   9, getIndex( pos4 ) );
         CHECK_EQUAL(  26, getIndex( pos5 ) );
     }

   	TEST_FIXTURE(SpatialGridGPUTester, getVolume_byIndex ){
        setDimension( 3 );
        setCoordinateSystem( TransportMeshTypeEnum::Cartesian );

        std::vector<gpuRayFloat_t> Xvertices = { -10, -1, 1, 10 };
        std::vector<gpuRayFloat_t> Yvertices = {  -2, -1, 1,  2 };
        std::vector<gpuRayFloat_t> Zvertices = { -10, -1, 1, 20 };

        setGrid( MonteRay_SpatialGrid::CART_X, Xvertices);
        setGrid( MonteRay_SpatialGrid::CART_Y, Yvertices);
        setGrid( MonteRay_SpatialGrid::CART_Z, Zvertices);

        initialize();
        copyToGPU();

        CHECK_CLOSE( 9.0*1.0*9.0, getVolume(0), 1e-6);
        CHECK_CLOSE( 2.0*1.0*9.0, getVolume(1), 1e-6);
        CHECK_CLOSE( 9.0*2.0*9.0, getVolume(3), 1e-6);
        CHECK_CLOSE( 9.0*1.0*9.0, getVolume(8), 1e-6);
        CHECK_CLOSE( 9.0*1.0*2.0, getVolume(9), 1e-6);
        CHECK_CLOSE( 9.0*1.0*19.0, getVolume(26), 1e-6);
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

    TEST_FIXTURE(SpatialGridGPUTester, getIndex_particle ){
        setDimension( 3 );
        setCoordinateSystem( TransportMeshTypeEnum::Cartesian );

        std::vector<gpuRayFloat_t> Xvertices = { -10, -1, 1, 10 };
        std::vector<gpuRayFloat_t> Yvertices = { -10, -1, 1, 10 };
        std::vector<gpuRayFloat_t> Zvertices = { -10, -1, 1, 10 };

        setGrid( MonteRay_SpatialGrid::CART_X, Xvertices);
        setGrid( MonteRay_SpatialGrid::CART_Y, Yvertices);
        setGrid( MonteRay_SpatialGrid::CART_Z, Zvertices);

        initialize();
        copyToGPU();

        particle p;

        MonteRay_SpatialGrid::Position_t pos1( -9.5, -9.5, -9.5 );
        MonteRay_SpatialGrid::Position_t pos2(  0.0, -9.5, -9.5 );
        MonteRay_SpatialGrid::Position_t pos3( -9.5,  0.0, -9.5 );
        MonteRay_SpatialGrid::Position_t pos4( -9.5, -9.5,  0.0 );
        MonteRay_SpatialGrid::Position_t pos5(  2.0,  2.0,  2.0 );

        p.pos = pos1;
        CHECK_EQUAL(   0, getIndex( p ) );
        p.pos = pos2;
        CHECK_EQUAL(   1, getIndex( p ) );
        p.pos = pos3;
        CHECK_EQUAL(   3, getIndex( p ) );
        p.pos = pos4;
        CHECK_EQUAL(   9, getIndex( p ) );
        p.pos = pos5;
        CHECK_EQUAL(  26, getIndex( p ) );
    }

    TEST_FIXTURE(SpatialGridGPUTester, getIndex_particle_afterReadFromSerialization ){
    	Grid_t grid;
    	grid.setDimension( 3 );
    	grid.setCoordinateSystem( TransportMeshTypeEnum::Cartesian );

    	std::vector<gpuRayFloat_t> Xvertices = { -10, -1, 1, 10 };
    	std::vector<gpuRayFloat_t> Yvertices = { -10, -1, 1, 10 };
    	std::vector<gpuRayFloat_t> Zvertices = { -10, -1, 1, 10 };

    	grid.setGrid( MonteRay_SpatialGrid::CART_X, Xvertices);
    	grid.setGrid( MonteRay_SpatialGrid::CART_Y, Yvertices);
    	grid.setGrid( MonteRay_SpatialGrid::CART_Z, Zvertices);

    	grid.initialize();

    	grid.write( "spatialgrid_cartesian_test_2.bin" );

    	{;
    		read( "spatialgrid_cartesian_test_2.bin" );
    		copyToGPU();

    		particle p;

    		MonteRay_SpatialGrid::Position_t pos1( -9.5, -9.5, -9.5 );
    		MonteRay_SpatialGrid::Position_t pos2(  0.0, -9.5, -9.5 );
    		MonteRay_SpatialGrid::Position_t pos3( -9.5,  0.0, -9.5 );
    		MonteRay_SpatialGrid::Position_t pos4( -9.5, -9.5,  0.0 );
    		MonteRay_SpatialGrid::Position_t pos5(  2.0,  2.0,  2.0 );

    		p.pos = pos1;
    		CHECK_EQUAL(   0, getIndex( p ) );
    		p.pos = pos2;
    		CHECK_EQUAL(   1, getIndex( p ) );
    		p.pos = pos3;
    		CHECK_EQUAL(   3, getIndex( p ) );
    		p.pos = pos4;
    		CHECK_EQUAL(   9, getIndex( p ) );
    		p.pos = pos5;
    		CHECK_EQUAL(  26, getIndex( p ) );
    	}

    }

#if true
    TEST_FIXTURE(SpatialGridGPUTester,  rayTrace_1D_external_to_internal_posX_pos_and_dir ) {
         setDimension( 3 );
         setCoordinateSystem( TransportMeshTypeEnum::Cartesian );

         std::vector<gpuRayFloat_t> Xvertices = { -1, 0, 1 };
         std::vector<gpuRayFloat_t> Yvertices = { -1, 0, 1 };
         std::vector<gpuRayFloat_t> Zvertices = { -1, 0, 1 };

         setGrid( MonteRay_SpatialGrid::CART_X, Xvertices);
         setGrid( MonteRay_SpatialGrid::CART_Y, Yvertices);
         setGrid( MonteRay_SpatialGrid::CART_Z, Zvertices);

         initialize();
         copyToGPU();

         CHECK_EQUAL( 2, getNumGridBins(0) );
         CHECK_EQUAL( 2, getNumGridBins(1) );
         CHECK_EQUAL( 2, getNumGridBins(2) );

         Grid_t::Position_t position (  -1.5, -0.5, -0.5 );
         Grid_t::Position_t direction(    1,   0,    0 );
         direction.normalize();
         gpuRayFloat_t distance = 2.0;

         rayTraceList_t distances = rayTrace( position, direction, distance);

         CHECK_EQUAL( 2, distances.size() );
         CHECK_EQUAL( 0, distances.id(0) );
         CHECK_CLOSE( 1.0, distances.dist(0), 1e-6 );
         CHECK_EQUAL( 1, distances.id(1) );
         CHECK_CLOSE( 0.5, distances.dist(1), 1e-6 );
     }
#endif

#if true

    TEST_FIXTURE(SpatialGridGPUTester, rayTrace_1D_external_to_internal_posX_particle ) {
        setDimension( 3 );
        setCoordinateSystem( TransportMeshTypeEnum::Cartesian );

        std::vector<gpuRayFloat_t> Xvertices = { -1, 0, 1 };
        std::vector<gpuRayFloat_t> Yvertices = { -1, 0, 1 };
        std::vector<gpuRayFloat_t> Zvertices = { -1, 0, 1 };

        setGrid( MonteRay_SpatialGrid::CART_X, Xvertices);
        setGrid( MonteRay_SpatialGrid::CART_Y, Yvertices);
        setGrid( MonteRay_SpatialGrid::CART_Z, Zvertices);

        initialize();
        copyToGPU();

        Grid_t::Position_t position (  -1.5, -0.5, -0.5 );
        Grid_t::Position_t direction(    1,   0,    0 );
        direction.normalize();
        gpuRayFloat_t distance = 2.0;

        particle p;
        p.pos = position;
        p.dir = direction;

        rayTraceList_t distances = rayTrace( p, distance);

        CHECK_EQUAL( 2, distances.size() );
        CHECK_EQUAL( 0, distances.id(0) );
        CHECK_CLOSE( 1.0, distances.dist(0), 1e-6 );
        CHECK_EQUAL( 1, distances.id(1) );
        CHECK_CLOSE( 0.5, distances.dist(1), 1e-6 );
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
#endif

  	TEST( cleanup ) {
   		gpuReset();
   	}

}

} // end namespace
