#include <UnitTest++.h>

#include <memory>
#include <vector>
#include <array>

#include "MonteRay_CylindricalGrid.hh"
#include "MonteRayConstants.hh"
#include "UnitTestHelper.hh"

using namespace MonteRay;

namespace MonteRay_CylindricalGrid_tester{

SUITE( MonteRay_CylindricalGrid_tests ) {
    using CylindricalGrid = MonteRay_CylindricalGrid;
    using GridBins = MonteRay_GridBins;
    using Position = Vector3D<gpuRayFloat_t>;
    using Direction = Vector3D<gpuRayFloat_t>;

    enum cartesian_coord {x=0,y=1,z=2};
    enum coord {R=0,CZ=1,Theta=2,DIM=3};


    TEST(GettersAndConvertFromCartesian) {

      std::unique_ptr<CylindricalGrid> pCyl;
      std::vector<double> Rverts = { 0.0, 1.5, 2.0 };
      std::vector<double> Zverts = { -10, -5, 0, 5, 10 };
      pCyl = std::make_unique<CylindricalGrid>(2, GridBins{Rverts, GridBins::RADIAL}, GridBins{Zverts});
      auto pos = pCyl->convertFromCartesian( Vector3D<gpuRayFloat_t>( 1.0, 1.0, 5.0) );

      CHECK_EQUAL( 2, pCyl->getDimension() );
      CHECK_EQUAL(2, pCyl->getNumBins(0));
      CHECK_EQUAL(4, pCyl->getNumBins(1));

      pos = pCyl->convertFromCartesian( Vector3D<gpuRayFloat_t>( 1.0, 1.0, 5.0) );
      CHECK_CLOSE( std::sqrt(2.0), pos[0], 1e-5);
      CHECK_CLOSE( 5.0, pos[1], 1e-11);
      CHECK_CLOSE( 0.0, pos[2], 1e-11);
    }

    class CylindricalGridTester{
      public:
        std::unique_ptr<CylindricalGrid> pCyl;
        CylindricalGridTester(){
        std::vector<double> Rverts = { 1.0, 2.0, 3.0 };
        std::vector<double> Zverts = { -10,  0,  10 };
        
          pCyl = std::make_unique<CylindricalGrid>(2, GridBins{Rverts, GridBins::RADIAL}, GridBins{Zverts});
        }
    };

    TEST_FIXTURE(CylindricalGridTester, getRadialIndexFromR ) {

        CHECK_EQUAL(   0, pCyl->getRadialIndexFromR( 0.5 ) );
        CHECK_EQUAL(   1, pCyl->getRadialIndexFromR( 1.5 ) );
        CHECK_EQUAL(   2, pCyl->getRadialIndexFromR( 2.5 ) );
        CHECK_EQUAL(   3, pCyl->getRadialIndexFromR( 3.5 ) );
        CHECK_EQUAL(   3, pCyl->getRadialIndexFromR( 30.5 ) );

    }

    TEST_FIXTURE(CylindricalGridTester, getRadialIndexFromRSq ) {
        CHECK_EQUAL(   0, pCyl->getRadialIndexFromRSq( 0.5*0.5 ) );
        CHECK_EQUAL(   1, pCyl->getRadialIndexFromRSq( 1.5*1.5 ) );
        CHECK_EQUAL(   2, pCyl->getRadialIndexFromRSq( 2.5*2.5 ) );
        CHECK_EQUAL(   3, pCyl->getRadialIndexFromRSq( 3.5*3.4 ) );
        CHECK_EQUAL(   3, pCyl->getRadialIndexFromRSq( 30.5*30.5 ) );
    }

    TEST_FIXTURE(CylindricalGridTester, getAxialIndex ) {

        CHECK_EQUAL(  -1, pCyl->getAxialIndex( -100.5 ) );
        CHECK_EQUAL(  -1, pCyl->getAxialIndex( -10.5 ) );
        CHECK_EQUAL(   0, pCyl->getAxialIndex( -9.5 ) );
        CHECK_EQUAL(   1, pCyl->getAxialIndex( 9.5) );
        CHECK_EQUAL(   2, pCyl->getAxialIndex( 10.5 ) );
        CHECK_EQUAL(   2, pCyl->getAxialIndex( 100.5 ) );
    }

    TEST_FIXTURE(CylindricalGridTester, isIndexOutside_R ) {

        CHECK_EQUAL(   false, pCyl->isIndexOutside(R, 0 ) );
        CHECK_EQUAL(   false, pCyl->isIndexOutside(R, 1 ) );
        CHECK_EQUAL(   false, pCyl->isIndexOutside(R, 2 ) );
        CHECK_EQUAL(    true, pCyl->isIndexOutside(R, 3 ) );

    }

    TEST_FIXTURE(CylindricalGridTester, isIndexOutside_CZ ) {

        CHECK_EQUAL(    true, pCyl->isIndexOutside(CZ, -1 ) );
        CHECK_EQUAL(   false, pCyl->isIndexOutside(CZ,  0 ) );
        CHECK_EQUAL(   false, pCyl->isIndexOutside(CZ,  1 ) );
        CHECK_EQUAL(    true, pCyl->isIndexOutside(CZ,  2 ) );
    }

    TEST_FIXTURE(CylindricalGridTester, getIndex ) {

        Position pos1(  0.5,  0.0, -9.5 );
        Position pos2(  1.5,  0.0, -9.5 );
        Position pos3(  2.5,  0.0, -9.5 );
        Position pos4(  3.5,  0.0, -9.5 );

        CHECK_EQUAL(   0, pCyl->getIndex( pos1 ) );
        CHECK_EQUAL(   1, pCyl->getIndex( pos2 ) );
        CHECK_EQUAL(   2, pCyl->getIndex( pos3 ) );
        CHECK_EQUAL( std::numeric_limits<unsigned>::max(), pCyl->getIndex( pos4 ) );

        pos1 = Position(  0.5,  0.0, 9.5 );
        pos2 = Position(  1.5,  0.0, 9.5 );
        pos3 = Position(  2.5,  0.0, 9.5 );
        pos4 = Position(  3.5,  0.0, 9.5 );

        CHECK_EQUAL(   3, pCyl->getIndex( pos1 ) );
        CHECK_EQUAL(   4, pCyl->getIndex( pos2 ) );
        CHECK_EQUAL(   5, pCyl->getIndex( pos3 ) );
        CHECK_EQUAL( std::numeric_limits<unsigned>::max(), pCyl->getIndex( pos4 ) );

        pos1 = Position(  0.0,  0.5, 9.5 );
        pos2 = Position(  0.0,  1.5, 9.5 );
        pos3 = Position(  0.0,  2.5, 9.5 );
        pos4 = Position(  0.0,  3.5, 9.5 );

        CHECK_EQUAL(   3, pCyl->getIndex( pos1 ) );
        CHECK_EQUAL(   4, pCyl->getIndex( pos2 ) );
        CHECK_EQUAL(   5, pCyl->getIndex( pos3 ) );
        CHECK_EQUAL( std::numeric_limits<unsigned>::max(), pCyl->getIndex( pos4 ) );

        pos1 = Position(  0.0,  0.5, 10.5 );
        pos2 = Position(  0.0,  1.5, 10.5 );
        pos3 = Position(  0.0,  2.5, 10.5 );
        pos4 = Position(  0.0,  3.5, 10.5 );

        CHECK_EQUAL( std::numeric_limits<unsigned>::max(), pCyl->getIndex( pos1 ) );
        CHECK_EQUAL( std::numeric_limits<unsigned>::max(), pCyl->getIndex( pos2 ) );
        CHECK_EQUAL( std::numeric_limits<unsigned>::max(), pCyl->getIndex( pos3 ) );
        CHECK_EQUAL( std::numeric_limits<unsigned>::max(), pCyl->getIndex( pos4 ) );

    }

    TEST_FIXTURE(CylindricalGridTester, isOutside_posRadius ) {
        int indices[] = {3,0,0};
        CHECK_EQUAL( true, pCyl->isOutside( indices ) );
    }

    TEST_FIXTURE(CylindricalGridTester, isOutside_Radius_false ) {
        int indices[] = {2,0,0};
        CHECK_EQUAL( false, pCyl->isOutside( indices ) );

        indices[0] = 0;
        CHECK_EQUAL( false, pCyl->isOutside( indices ) );
    }

    TEST_FIXTURE(CylindricalGridTester, isOutside_negZ ) {
        int indices[] = {0,-1,0};
        CHECK_EQUAL( true, pCyl->isOutside( indices ) );
    }

    TEST_FIXTURE(CylindricalGridTester, isOutside_posZ ) {
        int indices[] = {0,2,0};
        CHECK_EQUAL( true, pCyl->isOutside( indices ) );
    }

    TEST_FIXTURE(CylindricalGridTester, isOutside_Z_false ) {
        int indices[] = {0,1,0};
        CHECK_EQUAL( false, pCyl->isOutside( indices ) );
    }

    TEST_FIXTURE(CylindricalGridTester, calcIJK ) {
        uint3 indices;

        indices = pCyl->calcIJK( 0 );
        CHECK_EQUAL( 0, indices.x );
        CHECK_EQUAL( 0, indices.y );
        CHECK_EQUAL( 0, indices.z );

        indices = pCyl->calcIJK( 1 );
        CHECK_EQUAL( 1, indices.x );
        CHECK_EQUAL( 0, indices.y );
        CHECK_EQUAL( 0, indices.z );

        indices = pCyl->calcIJK( 2 );
        CHECK_EQUAL( 2, indices.x );
        CHECK_EQUAL( 0, indices.y );
        CHECK_EQUAL( 0, indices.z );

        indices = pCyl->calcIJK( 3 );
        CHECK_EQUAL( 0, indices.x );
        CHECK_EQUAL( 1, indices.y );
        CHECK_EQUAL( 0, indices.z );

        indices = pCyl->calcIJK( 4 );
        CHECK_EQUAL( 1, indices.x );
        CHECK_EQUAL( 1, indices.y );
        CHECK_EQUAL( 0, indices.z );

        indices = pCyl->calcIJK( 5 );
        CHECK_EQUAL( 2, indices.x );
        CHECK_EQUAL( 1, indices.y );
        CHECK_EQUAL( 0, indices.z );
    }

    TEST_FIXTURE(CylindricalGridTester, getVolume ) {
        CHECK_CLOSE( 10.0*(1.0)*MonteRay::pi, pCyl->getVolume(0), 1e-5 );
        CHECK_CLOSE( 10.0*(4.0-1.0)*MonteRay::pi, pCyl->getVolume(1), 1e-5 );
    }

#ifdef __CUDACC__
    __global__ void kernelCylindricalGridOnGPU(bool* testVal, const CylindricalGrid* const pCyl){
      *testVal = true;
    // TEST_FIXTURE(CylindricalGridTester, getRadialIndexFromRSq ) {

        GPU_CHECK_EQUAL(   0, pCyl->getRadialIndexFromRSq( 0.5*0.5 ) );
        GPU_CHECK_EQUAL(   1, pCyl->getRadialIndexFromRSq( 1.5*1.5 ) );
        GPU_CHECK_EQUAL(   2, pCyl->getRadialIndexFromRSq( 2.5*2.5 ) );
        GPU_CHECK_EQUAL(   3, pCyl->getRadialIndexFromRSq( 3.5*3.4 ) );
        GPU_CHECK_EQUAL(   3, pCyl->getRadialIndexFromRSq( 30.5*30.5 ) );

    // TEST_FIXTURE(CylindricalGridTester, getAxialIndex ) {

        GPU_CHECK_EQUAL(  -1, pCyl->getAxialIndex( -100.5 ) );
        GPU_CHECK_EQUAL(  -1, pCyl->getAxialIndex( -10.5 ) );
        GPU_CHECK_EQUAL(   0, pCyl->getAxialIndex( -9.5 ) );
        GPU_CHECK_EQUAL(   1, pCyl->getAxialIndex( 9.5) );
        GPU_CHECK_EQUAL(   2, pCyl->getAxialIndex( 10.5 ) );
        GPU_CHECK_EQUAL(   2, pCyl->getAxialIndex( 100.5 ) );

    // TEST_FIXTURE(CylindricalGridTester, isIndexOutside_R ) {

        GPU_CHECK_EQUAL(   false, pCyl->isIndexOutside(R, 0 ) );
        GPU_CHECK_EQUAL(   false, pCyl->isIndexOutside(R, 1 ) );
        GPU_CHECK_EQUAL(   false, pCyl->isIndexOutside(R, 2 ) );
        GPU_CHECK_EQUAL(    true, pCyl->isIndexOutside(R, 3 ) );

    // TEST_FIXTURE(CylindricalGridTester, isIndexOutside_CZ ) {

        GPU_CHECK_EQUAL(    true, pCyl->isIndexOutside(CZ, -1 ) );
        GPU_CHECK_EQUAL(   false, pCyl->isIndexOutside(CZ,  0 ) );
        GPU_CHECK_EQUAL(   false, pCyl->isIndexOutside(CZ,  1 ) );
        GPU_CHECK_EQUAL(    true, pCyl->isIndexOutside(CZ,  2 ) );

    // TEST_FIXTURE(CylindricalGridTester, getIndex ) {

        Position pos1(  0.5,  0.0, -9.5 );
        Position pos2(  1.5,  0.0, -9.5 );
        Position pos3(  2.5,  0.0, -9.5 );
        Position pos4(  3.5,  0.0, -9.5 );

        GPU_CHECK_EQUAL(   0, pCyl->getIndex( pos1 ) );
        GPU_CHECK_EQUAL(   1, pCyl->getIndex( pos2 ) );
        GPU_CHECK_EQUAL(   2, pCyl->getIndex( pos3 ) );
        GPU_CHECK_EQUAL( std::numeric_limits<unsigned>::max(), pCyl->getIndex( pos4 ) );

        pos1 = Position(  0.5,  0.0, 9.5 );
        pos2 = Position(  1.5,  0.0, 9.5 );
        pos3 = Position(  2.5,  0.0, 9.5 );
        pos4 = Position(  3.5,  0.0, 9.5 );

        GPU_CHECK_EQUAL(   3, pCyl->getIndex( pos1 ) );
        GPU_CHECK_EQUAL(   4, pCyl->getIndex( pos2 ) );
        GPU_CHECK_EQUAL(   5, pCyl->getIndex( pos3 ) );
        GPU_CHECK_EQUAL( std::numeric_limits<unsigned>::max(), pCyl->getIndex( pos4 ) );

        pos1 = Position(  0.0,  0.5, 9.5 );
        pos2 = Position(  0.0,  1.5, 9.5 );
        pos3 = Position(  0.0,  2.5, 9.5 );
        pos4 = Position(  0.0,  3.5, 9.5 );

        GPU_CHECK_EQUAL(   3, pCyl->getIndex( pos1 ) );
        GPU_CHECK_EQUAL(   4, pCyl->getIndex( pos2 ) );
        GPU_CHECK_EQUAL(   5, pCyl->getIndex( pos3 ) );
        GPU_CHECK_EQUAL( std::numeric_limits<unsigned>::max(), pCyl->getIndex( pos4 ) );

        pos1 = Position(  0.0,  0.5, 10.5 );
        pos2 = Position(  0.0,  1.5, 10.5 );
        pos3 = Position(  0.0,  2.5, 10.5 );
        pos4 = Position(  0.0,  3.5, 10.5 );

        GPU_CHECK_EQUAL( std::numeric_limits<unsigned>::max(), pCyl->getIndex( pos1 ) );
        GPU_CHECK_EQUAL( std::numeric_limits<unsigned>::max(), pCyl->getIndex( pos2 ) );
        GPU_CHECK_EQUAL( std::numeric_limits<unsigned>::max(), pCyl->getIndex( pos3 ) );
        GPU_CHECK_EQUAL( std::numeric_limits<unsigned>::max(), pCyl->getIndex( pos4 ) );

    // TEST_FIXTURE(CylindricalGridTester, isOutside_posRadius ) {
    {
        int indices[] = {3,0,0};
        GPU_CHECK_EQUAL( true, pCyl->isOutside( indices ) );
    }

    // TEST_FIXTURE(CylindricalGridTester, isOutside_Radius_false ) {
    {
        int indices[] = {2,0,0};
        GPU_CHECK_EQUAL( false, pCyl->isOutside( indices ) );

        indices[0] = 0;
        GPU_CHECK_EQUAL( false, pCyl->isOutside( indices ) );
    }

   // TEST_FIXTURE(CylindricalGridTester, isOutside_negZ ) {
    {
        int indices[] = {0,-1,0};
        GPU_CHECK_EQUAL( true, pCyl->isOutside( indices ) );
    }

    // TEST_FIXTURE(CylindricalGridTester, isOutside_posZ ) {
    {
        int indices[] = {0,2,0};
        GPU_CHECK_EQUAL( true, pCyl->isOutside( indices ) );
    }

    // TEST_FIXTURE(CylindricalGridTester, isOutside_Z_false ) {
    {
        int indices[] = {0,1,0};
        GPU_CHECK_EQUAL( false, pCyl->isOutside( indices ) );
    }

    // TEST_FIXTURE(CylindricalGridTester, calcIJK ) {
        uint3 indices;

        indices = pCyl->calcIJK( 0 );
        GPU_CHECK_EQUAL( 0, indices.x );
        GPU_CHECK_EQUAL( 0, indices.y );
        GPU_CHECK_EQUAL( 0, indices.z );

        indices = pCyl->calcIJK( 1 );
        GPU_CHECK_EQUAL( 1, indices.x );
        GPU_CHECK_EQUAL( 0, indices.y );
        GPU_CHECK_EQUAL( 0, indices.z );

        indices = pCyl->calcIJK( 2 );
        GPU_CHECK_EQUAL( 2, indices.x );
        GPU_CHECK_EQUAL( 0, indices.y );
        GPU_CHECK_EQUAL( 0, indices.z );

        indices = pCyl->calcIJK( 3 );
        GPU_CHECK_EQUAL( 0, indices.x );
        GPU_CHECK_EQUAL( 1, indices.y );
        GPU_CHECK_EQUAL( 0, indices.z );

        indices = pCyl->calcIJK( 4 );
        GPU_CHECK_EQUAL( 1, indices.x );
        GPU_CHECK_EQUAL( 1, indices.y );
        GPU_CHECK_EQUAL( 0, indices.z );

        indices = pCyl->calcIJK( 5 );
        GPU_CHECK_EQUAL( 2, indices.x );
        GPU_CHECK_EQUAL( 1, indices.y );
        GPU_CHECK_EQUAL( 0, indices.z );


    // TEST_FIXTURE(CylindricalGridTester, getVolume ) {
        GPU_CHECK_CLOSE( 10.0*(1.0)*MonteRay::pi, pCyl->getVolume(0), 1e-5 );
        GPU_CHECK_CLOSE( 10.0*(4.0-1.0)*MonteRay::pi, pCyl->getVolume(1), 1e-5 );
    }
    
    TEST_FIXTURE(CylindricalGridTester, CylindricalGridOnGPU ) {
      bool* testVal;
      cudaMallocManaged(&testVal, sizeof(bool));
      kernelCylindricalGridOnGPU<<<1, 1>>>(testVal, pCyl.get());
      cudaDeviceSynchronize();
      CHECK(*testVal);
      cudaFree(testVal);
    }
#endif

  class CylindricalGridTesterTwo{
    public:
      std::unique_ptr<CylindricalGrid> pCyl;
      gpuRayFloat_t two = 2.0;
      gpuRayFloat_t one = 1.0;

      CylindricalGridTesterTwo(){
      std::vector<gpuRayFloat_t> Rverts = { 1.0, 2.0 };
      std::vector<gpuRayFloat_t> Zverts = { -1, 0, 1};
      
        pCyl = std::make_unique<CylindricalGrid>(2, GridBins{Rverts, GridBins::RADIAL}, GridBins{Zverts});
        CHECK_EQUAL(4, pCyl->size());
      }
  };

  TEST_FIXTURE( CylindricalGridTesterTwo, ConvertToCellReferenceFrame ) {
    Position pos(0.5, 0.0, 0.5);
    Direction dir(1.0,   0.0,    0.0);
    gpuRayFloat_t speed = 1.0;
    Direction velocity(1.0, 2.0, 3.0);
    auto newDirAndSpeed = pCyl->convertToCellReferenceFrame(velocity, pos, dir, speed);
    CHECK_CLOSE(2.0, newDirAndSpeed.speed(), 1e-6);
    CHECK_CLOSE(0.0, newDirAndSpeed.direction()[0], 1e-6);
    CHECK_CLOSE(-1.0, newDirAndSpeed.direction()[2], 1e-6);
  }

  TEST_FIXTURE( CylindricalGridTesterTwo, CalcIndices ){
    Position pos( 0.5, 0.0, 0.5 );
    auto indices = pCyl->calcIndices(pos);
    CHECK_EQUAL(0, indices[0]);
    CHECK_EQUAL(1, indices[1]);
  }

  TEST_FIXTURE( CylindricalGridTesterTwo, getMinRadialDistAndDir){
    Position pos = { -1.0, -1.0, 0.5};
    Direction dir = Direction{1.0, 1.0, 0.0}.normalize();
    auto indices = pCyl->calcIndices(pos);
    auto distAndDir = pCyl->getMinRadialDistAndDir(pos, dir, indices[R]);
    CHECK_CLOSE(sqrt(2) - 1.0, distAndDir.distance(), 1e-6);
    CHECK_EQUAL(R, distAndDir.dimension());
    CHECK_EQUAL(false, distAndDir.isPositiveDir());
  }

  TEST_FIXTURE( CylindricalGridTesterTwo, GetMinDistToSurface){
    Position pos( 0.5, 0.0,  0.5 );
    Position dir(1, 0, 0);
    auto indices = pCyl->calcIndices(pos);
    auto distAndDir = pCyl->getMinDistToSurface(pos, dir, indices.data());
    CHECK_CLOSE(0.5, distAndDir.distance(), 1e-6);
    CHECK_EQUAL(R, distAndDir.dimension());
    CHECK_EQUAL(true, distAndDir.isPositiveDir());

    pos = { 0.0, 0.5,  0.5 };
    dir = {0, -1, 0};
    distAndDir = pCyl->getMinDistToSurface(pos, dir, indices.data());
    CHECK_CLOSE(1.5, distAndDir.distance(), 1e-6);
    CHECK_EQUAL(R, distAndDir.dimension());
    CHECK_EQUAL(true, distAndDir.isPositiveDir());

    pos = { 0.0, 0.0,  0.5 };
    dir = {0, 0, 1};
    distAndDir = pCyl->getMinDistToSurface(pos, dir, indices.data());
    CHECK_CLOSE(0.5, distAndDir.distance(), 1e-6);
    CHECK_EQUAL(CZ, distAndDir.dimension());
    CHECK_EQUAL(true, distAndDir.isPositiveDir());

    pos = { 0.0, 0.0,  0.5 };
    dir = {0, 0, -1};
    distAndDir = pCyl->getMinDistToSurface(pos, dir, indices.data());
    CHECK_CLOSE(0.5, distAndDir.distance(), 1e-6);
    CHECK_EQUAL(CZ, distAndDir.dimension());
    CHECK_EQUAL(false, distAndDir.isPositiveDir());

    pos = { 1.1, 1.1,  0.5 };
    dir = {-1, 0, 0};
    indices = pCyl->calcIndices(pos);
    distAndDir = pCyl->getMinDistToSurface(pos, dir, indices.data());
    CHECK_CLOSE(1.1 + Math::sqrt(4.0 - 1.1*1.1), distAndDir.distance(), 1e-6);
    CHECK_EQUAL(R, distAndDir.dimension());
    CHECK_EQUAL(true, distAndDir.isPositiveDir());
  }

  TEST_FIXTURE( CylindricalGridTesterTwo, isMovingInward){
    Position pos = { -1.5, -1.5, -0.75 };
    Direction dir = Direction{-Math::sqrt(two), -Math::sqrt(two), one}.normalize();
    CHECK(not pCyl->isMovingInward(pos, dir));

    dir = Direction{Math::sqrt(two), Math::sqrt(two), one}.normalize();
    CHECK(pCyl->isMovingInward(pos, dir));
  }

  TEST_FIXTURE( CylindricalGridTesterTwo, DistanceToInsideOfMesh){
    Position pos = { -1.5, -1.5, -0.75 };
    Direction dir = Direction{-1.0, -1.0, 0.0};
    auto distanceToInsideOfMesh = pCyl->getDistanceToInsideOfMesh(pos, dir);
    CHECK_EQUAL(std::numeric_limits<gpuRayFloat_t>::infinity(), distanceToInsideOfMesh);

    dir = Direction{Math::sqrt(two), Math::sqrt(two), one}.normalize();
    distanceToInsideOfMesh = pCyl->getDistanceToInsideOfMesh(pos, dir);
    auto answer = (Math::sqrt(2*1.5*1.5) - 2.0)/Math::sqrt(dir[0]*dir[0] + dir[1]*dir[1]);
    CHECK_CLOSE(answer, distanceToInsideOfMesh, 1e-6);

    pos = { -1.5, -1.5, -1.5 };
    distanceToInsideOfMesh = pCyl->getDistanceToInsideOfMesh(pos, dir);
    answer = 0.5/dir[2];
    CHECK_CLOSE(answer, distanceToInsideOfMesh, 1e-6);

    dir = Direction{Math::sqrt(two), Math::sqrt(two), -one}.normalize();
    distanceToInsideOfMesh = pCyl->getDistanceToInsideOfMesh(pos, dir);
    CHECK_EQUAL(std::numeric_limits<gpuRayFloat_t>::infinity(), distanceToInsideOfMesh);
  }

#ifdef __CUDACC__
  __global__ void kernelCylindricalGridCellByCellTests(bool* testVal, const CylindricalGrid* const pCyl){
    *testVal = true;
    gpuRayFloat_t one = 2.0;
    gpuRayFloat_t two = 2.0;

    //TEST_FIXTURE( CylindricalGridTesterTwo, ConvertToCellReferenceFrame ) {
    {
      Position pos(0.5, 0.0, 0.5);
      Direction dir(1.0,   0.0,    0.0);
      gpuRayFloat_t speed = 1.0;
      Direction velocity(1.0, 2.0, 3.0);
      auto newDirAndSpeed = pCyl->convertToCellReferenceFrame(velocity, pos, dir, speed);
      GPU_CHECK_CLOSE(2.0, newDirAndSpeed.speed(), 1e-6);
      GPU_CHECK_CLOSE(0.0, newDirAndSpeed.direction()[0], 1e-6);
      GPU_CHECK_CLOSE(-1.0, newDirAndSpeed.direction()[2], 1e-6);
    }

    //TEST_FIXTURE( CylindricalGridTesterTwo, CalcIndices ){
    {
      Position pos( 0.5, 0.0, 0.5 );
      auto indices = pCyl->calcIndices(pos);
      GPU_CHECK_EQUAL(0, indices[0]);
      GPU_CHECK_EQUAL(1, indices[1]);
    }

    //TEST_FIXTURE( CylindricalGridTesterTwo, getMinRadialDistAndDir){
    {
      Position pos = { -1.0, -1.0, 0.5};
      Direction dir = Direction{1.0, 1.0, 0.0}.normalize();
      auto indices = pCyl->calcIndices(pos);
      auto distAndDir = pCyl->getMinRadialDistAndDir(pos, dir, indices[R]);
      GPU_CHECK_CLOSE(sqrt(2) - 1.0, distAndDir.distance(), 1e-6);
      GPU_CHECK_EQUAL(R, distAndDir.dimension());
      GPU_CHECK_EQUAL(false, distAndDir.isPositiveDir());
    }

    //TEST_FIXTURE( CylindricalGridTesterTwo, GetMinDistToSurface){
    {
      Position pos( 0.5, 0.0,  0.5 );
      Position dir(1, 0, 0);
      auto indices = pCyl->calcIndices(pos);
      auto distAndDir = pCyl->getMinDistToSurface(pos, dir, indices.data());
      GPU_CHECK_CLOSE(0.5, distAndDir.distance(), 1e-6);
      GPU_CHECK_EQUAL(R, distAndDir.dimension());
      GPU_CHECK_EQUAL(true, distAndDir.isPositiveDir());

      pos = { 0.0, 0.5,  0.5 };
      dir = {0, -1, 0};
      distAndDir = pCyl->getMinDistToSurface(pos, dir, indices.data());
      GPU_CHECK_CLOSE(1.5, distAndDir.distance(), 1e-6);
      GPU_CHECK_EQUAL(R, distAndDir.dimension());
      GPU_CHECK_EQUAL(true, distAndDir.isPositiveDir());

      pos = { 0.0, 0.0,  0.5 };
      dir = {0, 0, 1};
      distAndDir = pCyl->getMinDistToSurface(pos, dir, indices.data());
      GPU_CHECK_CLOSE(0.5, distAndDir.distance(), 1e-6);
      GPU_CHECK_EQUAL(CZ, distAndDir.dimension());
      GPU_CHECK_EQUAL(true, distAndDir.isPositiveDir());

      pos = { 0.0, 0.0,  0.5 };
      dir = {0, 0, -1};
      distAndDir = pCyl->getMinDistToSurface(pos, dir, indices.data());
      GPU_CHECK_CLOSE(0.5, distAndDir.distance(), 1e-6);
      GPU_CHECK_EQUAL(CZ, distAndDir.dimension());
      GPU_CHECK_EQUAL(false, distAndDir.isPositiveDir());

      pos = { 1.1, 1.1,  0.5 };
      dir = {-1, 0, 0};
      indices = pCyl->calcIndices(pos);
      distAndDir = pCyl->getMinDistToSurface(pos, dir, indices.data());
      GPU_CHECK_CLOSE(1.1 + Math::sqrt(4.0 - 1.1*1.1), distAndDir.distance(), 1e-6);
      GPU_CHECK_EQUAL(R, distAndDir.dimension());
      GPU_CHECK_EQUAL(true, distAndDir.isPositiveDir());
    }

    //TEST_FIXTURE( CylindricalGridTesterTwo, isMovingInward){
    {
      Position pos = { -1.5, -1.5, -0.75 };
      Direction dir = Direction{-Math::sqrt(two), -Math::sqrt(two), one}.normalize();
      GPU_CHECK(not pCyl->isMovingInward(pos, dir));

      dir = Direction{Math::sqrt(two), Math::sqrt(two), one}.normalize();
      GPU_CHECK(pCyl->isMovingInward(pos, dir));
    }

    //TEST_FIXTURE( CylindricalGridTesterTwo, DistanceToInsideOfMesh){
    {
      Position pos = { -1.5, -1.5, -0.75 };
      Direction dir = Direction{-1.0, -1.0, 0.0};
      auto distanceToInsideOfMesh = pCyl->getDistanceToInsideOfMesh(pos, dir);
      GPU_CHECK_EQUAL(std::numeric_limits<gpuRayFloat_t>::infinity(), distanceToInsideOfMesh);

      dir = Direction{Math::sqrt(two), Math::sqrt(two), one}.normalize();
      distanceToInsideOfMesh = pCyl->getDistanceToInsideOfMesh(pos, dir);
      auto answer = (Math::sqrt(2*1.5*1.5) - 2.0)/Math::sqrt(dir[0]*dir[0] + dir[1]*dir[1]);
      GPU_CHECK_CLOSE(answer, distanceToInsideOfMesh, 1e-6);

      pos = { -1.5, -1.5, -1.5 };
      distanceToInsideOfMesh = pCyl->getDistanceToInsideOfMesh(pos, dir);
      answer = 0.5/dir[2];
      GPU_CHECK_CLOSE(answer, distanceToInsideOfMesh, 1e-6);

      dir = Direction{Math::sqrt(two), Math::sqrt(two), -one}.normalize();
      distanceToInsideOfMesh = pCyl->getDistanceToInsideOfMesh(pos, dir);
      GPU_CHECK_EQUAL(std::numeric_limits<gpuRayFloat_t>::infinity(), distanceToInsideOfMesh);
    }
  }

  TEST_FIXTURE( CylindricalGridTesterTwo, CellByCellTestsOnGPU){
      bool* testVal;
      cudaMallocManaged(&testVal, sizeof(bool));
      kernelCylindricalGridCellByCellTests<<<1, 1>>>(testVal, pCyl.get());
      cudaDeviceSynchronize();
      CHECK(*testVal);
      cudaFree(testVal);
  }

#endif

}

} // end namespace
