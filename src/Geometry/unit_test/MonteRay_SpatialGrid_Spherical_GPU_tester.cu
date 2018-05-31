#include <UnitTest++.h>

#include "MonteRay_SpatialGrid_GPU_helper.hh"
#include "MonteRayConstants.hh"

namespace MonteRay_SpatialGrid_Spherical_GPU_tester {

using namespace MonteRay_SpatialGrid_helper;

SUITE( MonteRay_SpatialGrid_Spherical_GPU_Tests ) {

   	TEST( setup ) {
   		gpuReset();
   	}

   	TEST_FIXTURE(SpatialGridGPUTester, set_Vertices ){
        //CHECK(false);
        std::vector<gpuRayFloat_t> vertices= {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10};
        setCoordinateSystem( TransportMeshTypeEnum::Spherical );
		setDimension( 1 );
        setGrid( MonteRay_SpatialGrid::SPH_R, vertices);
        initialize();
        copyToGPU();

        CHECK_EQUAL(  10, getNumGridBins(MonteRay_SpatialGrid::SPH_R) );
        CHECK_CLOSE( 1.0, getMinVertex(MonteRay_SpatialGrid::SPH_R), 1e-11 ); // 0.0 is removed from vertices during initialize()
        CHECK_CLOSE(10.0, getMaxVertex(MonteRay_SpatialGrid::SPH_R), 1e-11 );
    }

   	TEST_FIXTURE(SpatialGridGPUTester, read_test_Vertices_access_on_GPU ){
        Grid_t grid;
        grid.setCoordinateSystem( TransportMeshTypeEnum::Spherical );
        grid.setDimension( 1 );

        std::vector<gpuRayFloat_t> vertices= {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10};
        grid.setGrid( MonteRay_SpatialGrid::SPH_R, vertices);

        grid.initialize();
        CHECK_EQUAL( 10, grid.getNumVertices(0) );
        CHECK_EQUAL( 10, grid.getNumVerticesSq(0) );

        grid.write( "spatialgrid_spherical_test_vertices_gpu_1.bin" );

        {
            // read class state from archive
            read( "spatialgrid_spherical_test_vertices_gpu_1.bin" );
            CHECK_EQUAL( 10, pGridInfo->getNumVertices(0) );
            CHECK_EQUAL( 10, pGridInfo->getNumVerticesSq(0) );
            CHECK_EQUAL( 10, pGridInfo->getNumGridBins( MonteRay_SpatialGrid::SPH_R ) );
            CHECK_EQUAL( 10, pGridInfo->getNumGridBins( MonteRay_SpatialGrid::SPH_R ) );
            copyToGPU();

            CHECK_EQUAL(   10, getNumGridBins(MonteRay_SpatialGrid::SPH_R) );
            CHECK_CLOSE(  1.0,   getMinVertex(MonteRay_SpatialGrid::SPH_R), 1e-11 );
            CHECK_CLOSE( 10.0,   getMaxVertex(MonteRay_SpatialGrid::SPH_R), 1e-11 );
        }
    }

   	TEST_FIXTURE(SpatialGridGPUTester,  isInitialized ){
        setDimension( 1 );
        setCoordinateSystem( TransportMeshTypeEnum::Spherical );

        std::vector<gpuRayFloat_t> Rvertices = { 0, 1, 10 };

        setGrid( MonteRay_SpatialGrid::SPH_R, Rvertices);
        initialize();
        copyToGPU();

        CHECK_EQUAL( true, isInitialized() );
    }

   	TEST_FIXTURE(SpatialGridGPUTester, getIndexByPos ){
         setDimension( 1 );
         setCoordinateSystem( TransportMeshTypeEnum::Spherical );

         std::vector<gpuRayFloat_t> Rvertices = { 0, 1, 10 };

         setGrid( MonteRay_SpatialGrid::SPH_R, Rvertices);

         initialize();
         copyToGPU();

         MonteRay_SpatialGrid::Position_t pos1( 0.5, 0.5, 0.5 );
         MonteRay_SpatialGrid::Position_t pos2( 5.0, 5.0, 5.0 );

         CHECK_EQUAL(   0, getIndex( pos1 ) );
         CHECK_EQUAL(   1, getIndex( pos2 ) );
     }

   	TEST_FIXTURE(SpatialGridGPUTester, getVolume_byIndex ){
        setDimension( 1 );
        setCoordinateSystem( TransportMeshTypeEnum::Spherical );

        std::vector<gpuRayFloat_t> Rvertices = { 0, 1, 10 };

        setGrid( MonteRay_SpatialGrid::SPH_R, Rvertices);

        initialize();
        copyToGPU();

        CHECK_CLOSE( 4.0*pi/3.0, getVolume(0), 1e-11);
        CHECK_CLOSE(  (4.0*pi/3.0)*(1000.0 - 1.0), getVolume(1), 1e-11);
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
        setDimension( 1 );
        setCoordinateSystem( TransportMeshTypeEnum::Spherical );

        std::vector<gpuRayFloat_t> Rvertices = { 0, 1, 10 };

        setGrid( MonteRay_SpatialGrid::SPH_R, Rvertices);

        initialize();
        copyToGPU();

        particle p;

        MonteRay_SpatialGrid::Position_t pos1( 0.5, 0.5, 0.5 );
        MonteRay_SpatialGrid::Position_t pos2( 5.0, 5.0, 5.0 );

        p.pos = pos1;
        CHECK_EQUAL(   0, getIndex( p ) );
        p.pos = pos2;
        CHECK_EQUAL(   1, getIndex( p ) );
    }

    TEST_FIXTURE(SpatialGridGPUTester, getIndex_particle_afterReadFromSerialization ){
        Grid_t grid;
        grid.setDimension( 1 );
        grid.setCoordinateSystem( TransportMeshTypeEnum::Spherical );

        std::vector<gpuRayFloat_t> Rvertices = {1.0, 2.0, 3.0};

        grid.setGrid( MonteRay_SpatialGrid::SPH_R, Rvertices);

        grid.initialize();

    	grid.write( "spatialgrid_spherical_test_gpu_2.bin" );

    	{;
    		read( "spatialgrid_spherical_test_gpu_2.bin" );
    		copyToGPU();

    		particle p;

            MonteRay_SpatialGrid::Position_t pos1(  0.5,  0.0,  0.0 );
            MonteRay_SpatialGrid::Position_t pos2(  1.5,  0.0,  0.0 );
            MonteRay_SpatialGrid::Position_t pos3(  2.5,  0.0,  0.0 );
            MonteRay_SpatialGrid::Position_t pos4(  3.5,  0.0,  0.0 );

            p.pos = pos1;
            CHECK_EQUAL(   0, getIndex( p ) );
            p.pos = pos2;
            CHECK_EQUAL(   1, getIndex( p ) );
            p.pos = pos3;
            CHECK_EQUAL(   2, getIndex( p ) );
            p.pos = pos4;
            CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, getIndex( p ) );


            pos1 = MonteRay_SpatialGrid::Position_t(  0.0,  0.5, 0.0 );
            pos2 = MonteRay_SpatialGrid::Position_t(  0.0,  1.5, 0.0 );
            pos3 = MonteRay_SpatialGrid::Position_t(  0.0,  2.5, 0.0 );
            pos4 = MonteRay_SpatialGrid::Position_t(  0.0,  3.5, 0.0 );

            p.pos = pos1;
            CHECK_EQUAL(   0, getIndex( p ) );
            p.pos = pos2;
            CHECK_EQUAL(   1, getIndex( p ) );
            p.pos = pos3;
            CHECK_EQUAL(   2, getIndex( p ) );
            p.pos = pos4;
            CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, getIndex( p ) );
    	}

    }

    TEST_FIXTURE(SpatialGridGPUTester, rayTrace_in_XDir_outside_to_outside ) {
        setDimension( 1 );
        setCoordinateSystem( TransportMeshTypeEnum::Spherical );

        std::vector<gpuRayFloat_t> Rvertices = {1.0, 2.0, 3.0, 5.0};

        setGrid( MonteRay_SpatialGrid::SPH_R, Rvertices);

        initialize();
        copyToGPU();

        MonteRay_SpatialGrid::Position_t position ( -6.0,  0.0,  0.0 );
        MonteRay_SpatialGrid::Position_t direction(   1,   0,    0 );
        gpuRayFloat_t distance = 100.0;

        rayTraceList_t distances = rayTrace( position, direction, distance );

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
    TEST_FIXTURE(SpatialGridGPUTester, rayTrace_in_XZDir_outside_to_outside_at_45degrees ) {
        setDimension( 1 );
        setCoordinateSystem( TransportMeshTypeEnum::Spherical );

        std::vector<gpuRayFloat_t> Rvertices = {1.0, 2.0, 3.0, 5.0};

        setGrid( MonteRay_SpatialGrid::SPH_R, Rvertices);

        initialize();
        copyToGPU();

        MonteRay_SpatialGrid::Position_t position ( -6.0, 0.0,  -6.0 );
        MonteRay_SpatialGrid::Position_t direction(   1,   0,    1 );
        direction.normalize();
        gpuRayFloat_t distance = 100.0;

        rayTraceList_t distances = rayTrace( position, direction, distance);

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
    TEST_FIXTURE(SpatialGridGPUTester, rayTrace_in_XZDir_inside_to_outside_at_45degrees ) {
        setDimension( 1 );
        setCoordinateSystem( TransportMeshTypeEnum::Spherical );

        std::vector<gpuRayFloat_t> Rvertices = {1.0, 2.0, 3.0, 5.0};

        setGrid( MonteRay_SpatialGrid::SPH_R, Rvertices);

        initialize();
        copyToGPU();

        MonteRay_SpatialGrid::Position_t position ( -2.0, 0.0,  -2.0 );
        MonteRay_SpatialGrid::Position_t direction(   -1,   0,    -1 );
        direction.normalize();
        gpuRayFloat_t distance = 100.0;

        rayTraceList_t distances = rayTrace( position, direction, distance);

        CHECK_EQUAL(   2,  distances.size() );
        CHECK_EQUAL(   2,  distances.id(0) );
        CHECK_CLOSE( 0.171572875254,  distances.dist(0), 1e-6 );
        CHECK_EQUAL(   3,  distances.id(1) );
        CHECK_CLOSE( 2.0,  distances.dist(1), 1e-6 );
    }

  	TEST( cleanup ) {
   		gpuReset();
   	}

}

} // end namespace
