#include <UnitTest++.h>

#include "MonteRay_SpatialGrid_GPU_helper.hh"
#include "MonteRayConstants.hh"

#include <stdexcept>
#include <fstream>
#include <vector>

#include "MonteRayVector3D.hh"

namespace MonteRay_SpatialGrid_Cylindrical_GPU_test{

using namespace MonteRay;
using namespace MonteRay_SpatialGrid_helper;

SUITE( MonteRay_SpatialGrid_Cylindrical_GPU_tests ) {
#ifdef __CUDACC__

    typedef MonteRay_SpatialGrid SpatialGrid;
    typedef MonteRay_SpatialGrid Grid_t;
    typedef MonteRay_SpatialGrid::Position_t Position_t;

    TEST( setup ) {
        //gpuReset();
    }

    template<typename VERTICES_T>
    void testSetup(SpatialGridGPUTester* tester, const VERTICES_T& Rverts, const VERTICES_T& Zverts) {
        tester->setCoordinateSystem( TransportMeshTypeEnum::Cylindrical );
        tester->setDimension( 2 );
        tester->setGrid( MonteRay_SpatialGrid::CYLR_R, Rverts);
        tester->setGrid( MonteRay_SpatialGrid::CYLR_Z, Zverts);
        tester->initialize();
        tester->copyToGPU();
    }

    TEST_FIXTURE(SpatialGridGPUTester, set_Radial_Vertices ){
        std::vector<gpuRayFloat_t> Rvertices = { 1.0, 2.0, 3.0, 5.0 };
        std::vector<gpuRayFloat_t> Zvertices = { 1.0, 2.0, 3.0, 5.0 };
        testSetup(this, Rvertices, Zvertices);

        CHECK_EQUAL(   4, getNumGridBins(MonteRay_SpatialGrid::CYLR_R)  );
        CHECK_CLOSE( 1.0, getMinVertex(MonteRay_SpatialGrid::CYLR_R), 1e-6 );
        CHECK_CLOSE( 5.0, getMaxVertex(MonteRay_SpatialGrid::CYLR_R), 1e-6 );
    }

    TEST_FIXTURE(SpatialGridGPUTester, set_rz_Vertices ){
        std::vector<gpuRayFloat_t> Rvertices = { 1.0, 2.0, 3.0, 5.0 };
        std::vector<gpuRayFloat_t> Zvertices = { -30, -2, -1, 0, 1, 2, 30 };
        testSetup(this, Rvertices, Zvertices);

        CHECK_EQUAL( 4, getNumGridBins(SpatialGrid::CYLR_R) );
        CHECK_CLOSE( 1.0, getMinVertex(SpatialGrid::CYLR_R), 1e-11 );
        CHECK_CLOSE( 5.0, getMaxVertex(SpatialGrid::CYLR_R), 1e-11 );

        CHECK_EQUAL( 6, getNumGridBins(SpatialGrid::CYLR_Z) );
        CHECK_CLOSE( -30.0, getMinVertex(SpatialGrid::CYLR_Z), 1e-11 );
        CHECK_CLOSE( 30.0, getMaxVertex(SpatialGrid::CYLR_Z), 1e-11 );
    }

    TEST_FIXTURE(SpatialGridGPUTester, cylindrical_modifies_grid_data_with_0_entry ){
        std::vector<gpuRayFloat_t> Rvertices = { 0.0, 1.0, 2.0, 3.0, 5.0 };
        std::vector<gpuRayFloat_t> Zvertices = { -30, -2, -1, 0, 1, 2, 30 };
        testSetup(this, Rvertices, Zvertices);

        CHECK_EQUAL( 4, getNumGridBins(SpatialGrid::CYLR_R) );
        CHECK_CLOSE( 1.0, getVertex( SpatialGrid::CYLR_R, 0), 1e-11 );
    }


    TEST_FIXTURE(SpatialGridGPUTester, cylindrical_modifies_grid_data_without_0_entry ){
        std::vector<gpuRayFloat_t> Rvertices = { 1.0, 2.0, 3.0, 5.0 };
        std::vector<gpuRayFloat_t> Zvertices = { -30, -2, -1, 0, 1, 2, 30 };
        testSetup(this, Rvertices, Zvertices);

        CHECK_EQUAL( 4, getNumGridBins(SpatialGrid::CYLR_R) );
        CHECK_CLOSE( 1.0, getVertex( SpatialGrid::CYLR_R, 0), 1e-11 );
    }

    TEST_FIXTURE(SpatialGridGPUTester, getIndexByPos ){
        std::vector<gpuRayFloat_t> Rvertices = { 1.0, 2.0, 3.0 };
        std::vector<gpuRayFloat_t> Zvertices = { -10,  0,  10 };
        testSetup(this, Rvertices, Zvertices);

        Position_t pos1(  0.5,  0.0, -9.5 );
        Position_t pos2(  1.5,  0.0, -9.5 );
        Position_t pos3(  2.5,  0.0, -9.5 );
        Position_t pos4(  3.5,  0.0, -9.5 );

        CHECK_EQUAL(   0, getIndex( pos1 ) );
        CHECK_EQUAL(   1, getIndex( pos2 ) );
        CHECK_EQUAL(   2, getIndex( pos3 ) );
        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, getIndex( pos4 ) );

        pos1 = Position_t(  0.5,  0.0, 9.5 );
        pos2 = Position_t(  1.5,  0.0, 9.5 );
        pos3 = Position_t(  2.5,  0.0, 9.5 );
        pos4 = Position_t(  3.5,  0.0, 9.5 );

        CHECK_EQUAL(   3, getIndex( pos1 ) );
        CHECK_EQUAL(   4, getIndex( pos2 ) );
        CHECK_EQUAL(   5, getIndex( pos3 ) );
        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, getIndex( pos4 ) );

        pos1 = Position_t(  0.0,  0.5, 9.5 );
        pos2 = Position_t(  0.0,  1.5, 9.5 );
        pos3 = Position_t(  0.0,  2.5, 9.5 );
        pos4 = Position_t(  0.0,  3.5, 9.5 );

        CHECK_EQUAL(   3, getIndex( pos1 ) );
        CHECK_EQUAL(   4, getIndex( pos2 ) );
        CHECK_EQUAL(   5, getIndex( pos3 ) );
        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, getIndex( pos4 ) );

        pos1 = Position_t(  0.0,  0.5, 10.5 );
        pos2 = Position_t(  0.0,  1.5, 10.5 );
        pos3 = Position_t(  0.0,  2.5, 10.5 );
        pos4 = Position_t(  0.0,  3.5, 10.5 );

        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, getIndex( pos1 ) );
        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, getIndex( pos2 ) );
        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, getIndex( pos3 ) );
        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, getIndex( pos4 ) );
    }

    TEST_FIXTURE(SpatialGridGPUTester, getVolume_byIndex ){
        std::vector<gpuRayFloat_t> Rvertices = { 1.0, 2.0, 3.0 };
        std::vector<gpuRayFloat_t> Zvertices = { -10,  0,  1.0, 3.0, 10 };
        testSetup(this, Rvertices, Zvertices);

        gpuRayFloat_t pi =  MonteRay::pi;
        CHECK_CLOSE( pi*std::pow(1.0,2)*10.0, getVolume(0), 1e-6);
        CHECK_CLOSE( pi*(std::pow(2.0,2)-std::pow(1.0,2))*10.0, getVolume(1), 1e-5);
        CHECK_CLOSE( pi*(std::pow(3.0,2)-std::pow(2.0,2))*10.0, getVolume(2), 1e-5);

        CHECK_CLOSE( pi*std::pow(1.0,2)*1.0, getVolume(3), 1e-6);
        CHECK_CLOSE( pi*(std::pow(2.0,2)-std::pow(1.0,2))*1.0, getVolume(4), 1e-6);
        CHECK_CLOSE( pi*(std::pow(3.0,2)-std::pow(2.0,2))*1.0, getVolume(5), 1e-6);

        CHECK_CLOSE( pi*std::pow(1.0,2)*2.0, getVolume(6), 1e-6);
        CHECK_CLOSE( pi*(std::pow(2.0,2)-std::pow(1.0,2))*2.0, getVolume(7), 1e-6);
        CHECK_CLOSE( pi*(std::pow(3.0,2)-std::pow(2.0,2))*2.0, getVolume(8), 1e-6);

        CHECK_CLOSE( pi*std::pow(1.0,2)*7.0, getVolume(9), 1e-6);
        CHECK_CLOSE( pi*(std::pow(2.0,2)-std::pow(1.0,2))*7.0, getVolume(10), 1e-5);
        CHECK_CLOSE( pi*(std::pow(3.0,2)-std::pow(2.0,2))*7.0, getVolume(11), 1e-5);
    }

    class particle {
    public:
        particle(void){};
        SpatialGrid::Position_t pos;
        SpatialGrid::Position_t dir;
        CUDA_CALLABLE_MEMBER SpatialGrid::Position_t getPosition(void) const { return pos; }
        CUDA_CALLABLE_MEMBER SpatialGrid::Position_t getDirection(void) const { return dir; }
    };

    TEST_FIXTURE(SpatialGridGPUTester, getIndex_particle ){
        std::vector<gpuRayFloat_t> Rvertices = { 1.0, 2.0, 3.0 };
        std::vector<gpuRayFloat_t> Zvertices = { -10,  0,  10 };
        testSetup(this, Rvertices, Zvertices);

        particle p;

        Position_t pos1(  0.5,  0.0, -9.5 );
        Position_t pos2(  1.5,  0.0, -9.5 );
        Position_t pos3(  2.5,  0.0, -9.5 );
        Position_t pos4(  3.5,  0.0, -9.5 );

        p.pos = pos1;
        CHECK_EQUAL(   0, getIndex( p ) );
        p.pos = pos2;
        CHECK_EQUAL(   1, getIndex( p ) );
        p.pos = pos3;
        CHECK_EQUAL(   2, getIndex( p ) );
        p.pos = pos4;
        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, getIndex( p ) );

        pos1 = Position_t(  0.5,  0.0, 9.5 );
        pos2 = Position_t(  1.5,  0.0, 9.5 );
        pos3 = Position_t(  2.5,  0.0, 9.5 );
        pos4 = Position_t(  3.5,  0.0, 9.5 );

        p.pos = pos1;
        CHECK_EQUAL(   3, getIndex( p ) );
        p.pos = pos2;
        CHECK_EQUAL(   4, getIndex( p ) );
        p.pos = pos3;
        CHECK_EQUAL(   5, getIndex( p ) );
        p.pos = pos4;
        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, getIndex( p ) );

        pos1 = Position_t(  0.0,  0.5, 9.5 );
        pos2 = Position_t(  0.0,  1.5, 9.5 );
        pos3 = Position_t(  0.0,  2.5, 9.5 );
        pos4 = Position_t(  0.0,  3.5, 9.5 );

        p.pos = pos1;
        CHECK_EQUAL(   3, getIndex( p ) );
        p.pos = pos2;
        CHECK_EQUAL(   4, getIndex( p ) );
        p.pos = pos3;
        CHECK_EQUAL(   5, getIndex( p ) );
        p.pos = pos4;
        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, getIndex( p ) );

        pos1 = Position_t(  0.0,  0.5, 10.5 );
        pos2 = Position_t(  0.0,  1.5, 10.5 );
        pos3 = Position_t(  0.0,  2.5, 10.5 );
        pos4 = Position_t(  0.0,  3.5, 10.5 );

        p.pos = pos1;
        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, getIndex( p ) );
        p.pos = pos2;
        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, getIndex( p ) );
        p.pos = pos3;
        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, getIndex( p ) );
        p.pos = pos4;
        CHECK_EQUAL( SpatialGrid::OUTSIDE_MESH, getIndex( p ) );

    }

    TEST_FIXTURE(SpatialGridGPUTester, getIndex_particle_afterReadFromSerialization ){
        std::vector<gpuRayFloat_t> Rvertices = { 1.0, 2.0, 3.0 };
        std::vector<gpuRayFloat_t> Zvertices = { -10,  0,  10 };
        testSetup(this, Rvertices, Zvertices);

        write( "spatialgrid_cylindrical_test_1.bin" );

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

    TEST_FIXTURE(SpatialGridGPUTester, rayTrace_in_ZDir_outside_to_outside ) {
        std::vector<gpuRayFloat_t> Rvertices = { 1.0, 2.0, 3.0, 5.0 };
        std::vector<gpuRayFloat_t> Zvertices = { -6, -3, -1, 1, 3, 6 };
        testSetup(this, Rvertices, Zvertices);

        Position_t position ( 0.0, 0.0, -7.5 );
        Position_t direction(   0,   0,    1 );
        gpuRayFloat_t distance = 100.0;

        rayTraceList_t distances = rayTrace( position, direction, distance );

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

    TEST_FIXTURE(SpatialGridGPUTester, rayTrace_in_ZDir_outside_to_outside_along_radial_vertex ) {
        std::vector<gpuRayFloat_t> Rvertices = { 1.0, 2.0, 3.0, 5.0 };
        std::vector<gpuRayFloat_t> Zvertices = { -6, -3, -1, 1, 3, 6 };
        testSetup(this, Rvertices, Zvertices);

        Position_t position ( 1.0, 0.0, -7.5 );
        Position_t direction(   0,   0,    1 );
        gpuRayFloat_t distance = 100.0;

        rayTraceList_t distances = rayTrace( position, direction, distance );

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

    TEST_FIXTURE(SpatialGridGPUTester, rayTrace_in_RDir_outside_to_outside ) {
        std::vector<gpuRayFloat_t> Rvertices = { 1.0, 2.0, 3.0, 5.0 };
        std::vector<gpuRayFloat_t> Zvertices = { -6, -3, -1, 1, 3, 6 };
        testSetup(this, Rvertices, Zvertices);

        Position_t position ( -6.0, 0.0,  -5.0 );
        Position_t direction(   1,   0,    0 );
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

    TEST_FIXTURE(SpatialGridGPUTester, rayTrace_in_RZDir_outside_to_outside_at_45degrees_thru_a_corner ) {
        // !!! DO NOT CHANGE WITHOUT DRAWING A PICTURE
        std::vector<gpuRayFloat_t> Rvertices = { 1.0, 2.0, 3.0, 5.0 };
        std::vector<gpuRayFloat_t> Zvertices = { -6, -3, -1, 1, 3, 6 };
        testSetup(this, Rvertices, Zvertices);

        Position_t position ( -6.0, 0.0,  -7.0 );
        Position_t direction(   1,   0,    1 );
        direction.normalize();
        gpuRayFloat_t distance = 100.0;

        rayTraceList_t distances = rayTrace( position, direction, distance );

        CHECK_EQUAL(  11,  distances.size() );

        // expectedIndex[2] can be 1 or 6
        std::vector<unsigned> expectedIndex = { 3, 2, 6, 5, 4, 8, 9, 10, 14, 15, 19 };
        gpuRayFloat_t s2 = std::sqrt(2.0);

        std::vector<gpuRayFloat_t> expectedDistance = {2.0f*s2, s2, 0.0f, s2, s2, s2, s2, 0.0f, s2, s2, s2, s2 };

        for( auto i=0; i<distances.size(); ++i ) {
            CHECK_EQUAL( expectedIndex   [i], distances.id(i) );
            CHECK_CLOSE( expectedDistance[i], distances.dist(i), 1e-5 );
        }
    }

    TEST_FIXTURE(SpatialGridGPUTester, rayTrace_in_RZDir_inside_to_outside_at_45degrees_thru_a_corner ) {
        // !!! DO NOT CHANGE WITHOUT DRAWING A PICTURE
        std::vector<gpuRayFloat_t> Rvertices = { 1.0, 2.0, 3.0, 5.0 };
        std::vector<gpuRayFloat_t> Zvertices = { -6, -3, -1, 1, 3, 6 };
        testSetup(this, Rvertices, Zvertices);

        Position_t position ( -2.0, 0.0,  -3.0 );
        Position_t direction(   -1,   0,    -1 );
        direction.normalize();
        gpuRayFloat_t distance = 100.0;

        rayTraceList_t distances = rayTrace( position, direction, distance );

        CHECK_EQUAL(   3,            distances.size() );
        CHECK_EQUAL(   6,            distances.id(0)  );
        CHECK_CLOSE( 0.0,            distances.dist(0), 1e-6 );
        CHECK_EQUAL(   2,            distances.id(1)  );
        CHECK_CLOSE( 1.0*sqrt(2.0),  distances.dist(1), 1e-6 );
        CHECK_EQUAL(   3,            distances.id(2)  );
        CHECK_CLOSE( 2.0*sqrt(2.0),  distances.dist(2), 1e-6 );
    }

#endif /* CUDACC */
}

} /* end namespace */
