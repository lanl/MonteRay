#include <UnitTest++.h>

#include <memory>
#include <vector>
#include <array>

#include "MonteRay_CartesianGrid.hh"
#include "MonteRayVector3D.hh"
#include "GPU_Utilities/UnitTestHelper.hh"

using namespace MonteRay;

namespace MonteRay_CartesianGrid_tester{

SUITE( MonteRay_CartesianGrid_basic_tests ) {
    using CartesianGrid = MonteRay_CartesianGrid;
    using Grid_t = MonteRay_CartesianGrid;
    using GridBins = MonteRay_GridBins;

    typedef MonteRay::Vector3D<gpuRayFloat_t> Position_t;

    class CartesianGridTester{
    public:
      std::unique_ptr<CartesianGrid> pCart;
      const int DIM = 3;
      enum coord {X,Y,Z};
      CartesianGridTester(){
          std::vector<gpuRayFloat_t> vertices = { -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
                  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10 };
      pCart = std::make_unique<CartesianGrid>(3, GridBins{vertices}, GridBins{vertices}, GridBins{vertices});
      }
    };

    TEST_FIXTURE(CartesianGridTester, getNumBins ) {
        CHECK_EQUAL( 20, pCart->getNumBins(0) );
        CHECK_EQUAL( 20, pCart->getNumBins(1) );
        CHECK_EQUAL( 20, pCart->getNumBins(2) );
    }

    TEST_FIXTURE(CartesianGridTester, getIndex ) {

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

    TEST_FIXTURE(CartesianGridTester, getDimIndex_negX ) {
        CHECK_EQUAL( -1, pCart->getDimIndex( 0, -10.5 ) );
    }
    TEST_FIXTURE(CartesianGridTester, getDimIndex_posX ) {
        CHECK_EQUAL( 20, pCart->getDimIndex( 0, 10.5 ) );
    }
    TEST_FIXTURE(CartesianGridTester, getDimIndex_inside_negSide_X ) {
        CHECK_EQUAL( 0, pCart->getDimIndex( 0, -9.5 ) );
    }
    TEST_FIXTURE(CartesianGridTester, getDimIndex_inside_posSide_X ) {
        CHECK_EQUAL( 19, pCart->getDimIndex( 0, 9.5 ) );
    }

    TEST_FIXTURE(CartesianGridTester, getDimIndex_negY ) {
        CHECK_EQUAL( -1, pCart->getDimIndex( 1, -10.5 ) );
    }
    TEST_FIXTURE(CartesianGridTester, getDimIndex_posY ) {
        CHECK_EQUAL( 20, pCart->getDimIndex( 1, 10.5 ) );
    }
    TEST_FIXTURE(CartesianGridTester, getDimIndex_inside_negSide_Y ) {
        CHECK_EQUAL( 0, pCart->getDimIndex( 1, -9.5 ) );
    }
    TEST_FIXTURE(CartesianGridTester, getDimIndex_inside_posSide_Y ) {
        CHECK_EQUAL( 19, pCart->getDimIndex( 1, 9.5 ) );
    }

    TEST_FIXTURE(CartesianGridTester, getDimIndex_negZ ) {
        CHECK_EQUAL( -1, pCart->getDimIndex( 2, -10.5 ) );
    }
    TEST_FIXTURE(CartesianGridTester, getDimIndex_posZ ) {
        CHECK_EQUAL( 20, pCart->getDimIndex( 2, 10.5 ) );
    }
    TEST_FIXTURE(CartesianGridTester, getDimIndex_inside_negSide_Z ) {
        CHECK_EQUAL( 0, pCart->getDimIndex( 2, -9.5 ) );
    }
    TEST_FIXTURE(CartesianGridTester, getDimIndex_inside_posSide_Z ) {
        CHECK_EQUAL( 19, pCart->getDimIndex( 2, 9.5 ) );
    }

    TEST_FIXTURE(CartesianGridTester, PositionOutOfBoundsToGrid ) {

        Position_t posNegX( -10.5, -9.5, -9.5 );
        Position_t posPosX(  10.5, -9.5, -9.5 );

        CHECK_EQUAL( std::numeric_limits<unsigned>::max(), pCart->getIndex( posNegX ) );
        CHECK_EQUAL( std::numeric_limits<unsigned>::max(), pCart->getIndex( posPosX ) );

        Position_t posNegY( -9.5, -10.5, -9.5 );
        Position_t posPosY( -9.5,  10.5, -9.5 );

        CHECK_EQUAL( std::numeric_limits<unsigned>::max(), pCart->getIndex( posNegY ) );
        CHECK_EQUAL( std::numeric_limits<unsigned>::max(), pCart->getIndex( posPosY ) );

        Position_t posNegZ( -9.5, -9.5, -10.5 );
        Position_t posPosZ( -9.5, -9.5,  10.5 );

        CHECK_EQUAL( std::numeric_limits<unsigned>::max(), pCart->getIndex( posNegZ ) );
        CHECK_EQUAL( std::numeric_limits<unsigned>::max(), pCart->getIndex( posPosZ ) );
    }


    TEST_FIXTURE(CartesianGridTester, PositionOnTheBoundsToGrid_WeDefineOutside ) {
        Position_t posNegX( -10.5, -9.5, -9.5 );
        Position_t posPosX(  10.5, -9.5, -9.5 );

        CHECK_EQUAL( std::numeric_limits<unsigned>::max(), pCart->getIndex( posNegX ) );
        CHECK_EQUAL( std::numeric_limits<unsigned>::max(), pCart->getIndex( posPosX ) );

        Position_t posNegY( -9.5, -10.5, -9.5 );
        Position_t posPosY( -9.5,  10.5, -9.5 );

        CHECK_EQUAL( std::numeric_limits<unsigned>::max(), pCart->getIndex( posNegY ) );
        CHECK_EQUAL( std::numeric_limits<unsigned>::max(), pCart->getIndex( posPosY ) );

        Position_t posNegZ( -9.5, -9.5, -10.5 );
        Position_t posPosZ( -9.5, -9.5,  10.5 );

        CHECK_EQUAL( std::numeric_limits<unsigned>::max(), pCart->getIndex( posNegZ ) );
        CHECK_EQUAL( std::numeric_limits<unsigned>::max(), pCart->getIndex( posPosZ ) );
    }

    TEST_FIXTURE(CartesianGridTester, isIndexOutside_negX ) {
        CHECK_EQUAL( true, pCart->isIndexOutside(0, -1) );
    }

    TEST_FIXTURE(CartesianGridTester, isIndexOutside_posX ) {
        CHECK_EQUAL( true, pCart->isIndexOutside(0, 20) );
    }
    TEST_FIXTURE(CartesianGridTester, isIndexOutside_false_negEnd ) {
        CHECK_EQUAL( false, pCart->isIndexOutside(0, 0) );
    }
    TEST_FIXTURE(CartesianGridTester, isIndexOutside_false_posEnd ) {
        CHECK_EQUAL( false, pCart->isIndexOutside(0, 19) );
    }

    TEST_FIXTURE(CartesianGridTester, isOutside_negX ) {
        int indices[] = {-1,0,0};
        CHECK_EQUAL( true, pCart->isOutside( indices ) );
    }
    TEST_FIXTURE(CartesianGridTester, isOutside_posX ) {
        int indices[] = {20,0,0};
        CHECK_EQUAL( true, pCart->isOutside( indices ) );
    }

    TEST_FIXTURE(CartesianGridTester, isOutside_negY ) {
        int indices[] = {0,-1,0};
        CHECK_EQUAL( true, pCart->isOutside( indices ) );
    }
    TEST_FIXTURE(CartesianGridTester, isOutside_posY ) {
        int indices[] = {0,20,0};
        CHECK_EQUAL( true, pCart->isOutside( indices ) );
    }
    TEST_FIXTURE(CartesianGridTester, isOutside_negZ ) {
        int indices[] = {0,0,-1};
        CHECK_EQUAL( true, pCart->isOutside( indices ) );
    }
    TEST_FIXTURE(CartesianGridTester, isOutside_posZ ) {
        int indices[] = {0,0,20};
        CHECK_EQUAL( true, pCart->isOutside( indices ) );
    }
    TEST_FIXTURE(CartesianGridTester, isOutside_false1 ) {
        int indices[] = {19,0,0};
        CHECK_EQUAL( false, pCart->isOutside( indices ) );
    }
    TEST_FIXTURE(CartesianGridTester, isOutside_false2 ) {
        int indices[] = {0,0,0};
        CHECK_EQUAL( false, pCart->isOutside( indices ) );
    }

    TEST(getVolume ) {
        std::vector<gpuRayFloat_t> vertices = {-3, -1, 0};

        CartesianGrid cart(3,
          std::array<GridBins, 3>{
            GridBins{vertices},
            GridBins{vertices},
            GridBins{vertices} 
          }
        );

        CHECK_CLOSE( 8.0, cart.getVolume(0), 1e-11 );
        CHECK_CLOSE( 4.0, cart.getVolume(1), 1e-11 );
        CHECK_CLOSE( 4.0, cart.getVolume(2), 1e-11 );
        CHECK_CLOSE( 2.0, cart.getVolume(3), 1e-11 );
        CHECK_CLOSE( 4.0, cart.getVolume(4), 1e-11 );
        CHECK_CLOSE( 2.0, cart.getVolume(5), 1e-11 );
        CHECK_CLOSE( 2.0, cart.getVolume(6), 1e-11 );
        CHECK_CLOSE( 1.0, cart.getVolume(7), 1e-11 );
    }


#ifdef __CUDAC__
  __global__ kernelTestCartesianGridOnGPU(bool* testVal, CartesianGrid* pCart){
    *testVal = true;

    *testVal = *testVal && ( 20, pCart->getNumBins(0) );
    *testVal = *testVal && ( 20, pCart->getNumBins(1) );
    *testVal = *testVal && ( 20, pCart->getNumBins(2) );

    Position_t pos1( -9.5, -9.5, -9.5 );
    Position_t pos2( -8.5, -9.5, -9.5 );
    Position_t pos3( -9.5, -8.5, -9.5 );
    Position_t pos4( -9.5, -9.5, -8.5 );
    Position_t pos5( -9.5, -7.5, -9.5 );

    *testVal = *testVal && (   0, pCart->getIndex( pos1 ) );
    *testVal = *testVal && (   1, pCart->getIndex( pos2 ) );
    *testVal = *testVal && (  20, pCart->getIndex( pos3 ) );
    *testVal = *testVal && (  40, pCart->getIndex( pos5 ) );
    *testVal = *testVal && ( 400, pCart->getIndex( pos4 ) );

    *testVal = *testVal && ( 19, pCart->getDimIndex( 2, 9.5 ) );
    *testVal = *testVal && ( 0, pCart->getDimIndex( 2, -9.5 ) );
    *testVal = *testVal && ( 20, pCart->getDimIndex( 2, 10.5 ) );
    *testVal = *testVal && ( -1, pCart->getDimIndex( 2, -10.5 ) );
    *testVal = *testVal && ( 19, pCart->getDimIndex( 1, 9.5 ) );
    *testVal = *testVal && ( 0, pCart->getDimIndex( 1, -9.5 ) );
    *testVal = *testVal && ( 20, pCart->getDimIndex( 1, 10.5 ) );
    *testVal = *testVal && ( -1, pCart->getDimIndex( 1, -10.5 ) );
    *testVal = *testVal && ( 19, pCart->getDimIndex( 0, 9.5 ) );
    *testVal = *testVal && ( 0, pCart->getDimIndex( 0, -9.5 ) );
    *testVal = *testVal && ( 20, pCart->getDimIndex( 0, 10.5 ) );
    *testVal = *testVal && ( -1, pCart->getDimIndex( 0, -10.5 ) );
    *testVal = *testVal && ( 0, pCart->getDimIndex( 2, -9.5 ) );

    Position_t posNegX( -10.5, -9.5, -9.5 );
    Position_t posPosX(  10.5, -9.5, -9.5 );

    *testVal = *testVal && ( std::numeric_limits<unsigned>::max(), pCart->getIndex( posNegX ) );
    *testVal = *testVal && ( std::numeric_limits<unsigned>::max(), pCart->getIndex( posPosX ) );

    Position_t posNegY( -9.5, -10.5, -9.5 );
    Position_t posPosY( -9.5,  10.5, -9.5 );

    *testVal = *testVal && ( std::numeric_limits<unsigned>::max(), pCart->getIndex( posNegY ) );
    *testVal = *testVal && ( std::numeric_limits<unsigned>::max(), pCart->getIndex( posPosY ) );

    Position_t posNegZ( -9.5, -9.5, -10.5 );
    Position_t posPosZ( -9.5, -9.5,  10.5 );

    *testVal = *testVal && ( std::numeric_limits<unsigned>::max(), pCart->getIndex( posNegZ ) );
    *testVal = *testVal && ( std::numeric_limits<unsigned>::max(), pCart->getIndex( posPosZ ) );
    Position_t posNegX( -10.5, -9.5, -9.5 );
    Position_t posPosX(  10.5, -9.5, -9.5 );

    *testVal = *testVal && ( std::numeric_limits<unsigned>::max(), pCart->getIndex( posNegX ) );
    *testVal = *testVal && ( std::numeric_limits<unsigned>::max(), pCart->getIndex( posPosX ) );

    Position_t posNegY( -9.5, -10.5, -9.5 );
    Position_t posPosY( -9.5,  10.5, -9.5 );

    *testVal = *testVal && ( std::numeric_limits<unsigned>::max(), pCart->getIndex( posNegY ) );
    *testVal = *testVal && ( std::numeric_limits<unsigned>::max(), pCart->getIndex( posPosY ) );

    Position_t posNegZ( -9.5, -9.5, -10.5 );
    Position_t posPosZ( -9.5, -9.5,  10.5 );

    *testVal = *testVal && ( std::numeric_limits<unsigned>::max(), pCart->getIndex( posNegZ ) );
    *testVal = *testVal && ( std::numeric_limits<unsigned>::max(), pCart->getIndex( posPosZ ) );

    
    int indices[] = {19,0,0};
    *testVal = *testVal && ( false, pCart->isOutside( indices ) );
    int indices[] = {0,0,20};
    *testVal = *testVal && ( true, pCart->isOutside( indices ) );
    int indices[] = {0,20,0};
    *testVal = *testVal && ( true, pCart->isOutside( indices ) );
    int indices[] = {0,-1,0};
    *testVal = *testVal && ( true, pCart->isOutside( indices ) );
    int indices[] = {20,0,0};
    *testVal = *testVal && ( true, pCart->isOutside( indices ) );
    int indices[] = {-1,0,0};
    *testVal = *testVal && ( true, pCart->isOutside( indices ) );
    int indices[] = {0,0,-1};
    *testVal = *testVal && ( true, pCart->isOutside( indices ) );
    int indices[] = {0,0,0};
    *testVal = *testVal && ( false, pCart->isOutside( indices ) );

    *testVal = *testVal && ( false, pCart->isIndexOutside(0, 19) );
    *testVal = *testVal && ( false, pCart->isIndexOutside(0, 0) );
    *testVal = *testVal && ( true, pCart->isIndexOutside(0, 20) );
    *testVal = *testVal && ( true, pCart->isIndexOutside(0, -1) );

    CHECK_CLOSE( 1.0, pCart->getVolume(4), 1e-11 );
  }

  TEST_FIXTURE(CartesianGridTester, CartesianGridOnGPU){
    bool* testVal;
    cudaMallocManaged(&testVal, sizeof(bool));

    kernelTestCartesianGridOnGPU<<<1, 1>>>(testVal, pCart.get());
    cudaDeviceSynchronize();
    CHECK(*testVal);

  }
#endif

  using Position_t = Vector3D<gpuRayFloat_t>;
  using Direction_t = Vector3D<gpuRayFloat_t>;
  using CartesianGrid = MonteRay_CartesianGrid;

  class CartesianGridTesterTwo{
    public:
    std::unique_ptr<CartesianGrid> pCart;

    CartesianGridTesterTwo(){
      std::vector<gpuRayFloat_t> vertices{-1, 0, 1};
      pCart = std::make_unique<CartesianGrid>(3, GridBins{vertices}, GridBins{vertices}, GridBins{vertices});
    }
  };

  TEST_FIXTURE( CartesianGridTesterTwo, ConvertToCellReferenceFrame ) {
    Position_t pos(0.0,   0.0,    0.0);
    Direction_t dir(1.0,   0.0,    0.0);
    gpuRayFloat_t speed = 4.0;
    Direction_t velocity{3.0, -2.0, 3.0};
    auto newDirAndSpeed = pCart->convertToCellReferenceFrame(velocity, pos, dir, speed);
    auto newSpeed = std::sqrt(1.0 + 2.0*2.0 + 3.0*3.0);
    CHECK_CLOSE(newSpeed, newDirAndSpeed.speed(), 1e-6);
    CHECK_CLOSE(1.0/newSpeed, newDirAndSpeed.direction()[0], 1e-6);
    CHECK_CLOSE(-3.0/newSpeed, newDirAndSpeed.direction()[2], 1e-6);
  }

  TEST_FIXTURE( CartesianGridTesterTwo, CalcIndices ){
    Position_t pos( -0.5, -1.5,  0.5 );
    auto indices = pCart->calcIndices(pos);
    CHECK_EQUAL(0, indices[0]);
    CHECK_EQUAL(-1, indices[1]);
    CHECK_EQUAL(1, indices[2]);
  }

  TEST_FIXTURE( CartesianGridTesterTwo, GetMinDistToSurface){
    Position_t pos( -0.5, 0.2,  0.3 );
    Position_t dir(1, 0, 0);
    auto indices = pCart->calcIndices(pos);
    auto distAndDir = pCart->getMinDistToSurface(pos, dir, indices.data());
    CHECK_CLOSE(0.5, distAndDir.distance(), 1e-6);
    CHECK_EQUAL(0, distAndDir.dimension());

    dir = {0, -1, 0};
    distAndDir = pCart->getMinDistToSurface(pos, dir, indices.data());
    CHECK_CLOSE(0.2, distAndDir.distance(), 1e-6);
    CHECK_EQUAL(1, distAndDir.dimension());

    dir = {0, 0, 1};
    distAndDir = pCart->getMinDistToSurface(pos, dir, indices.data());
    CHECK_CLOSE(0.7, distAndDir.distance(), 1e-6);
    CHECK_EQUAL(2, distAndDir.dimension());
  }

#ifdef __CUDACC__

  __global__ void kernelTestCartesianGrid(bool* testVal, const CartesianGrid* pCart) {
    *testVal = true;
    {
      Position_t pos(0.0,   0.0,    0.0);
      Direction_t dir(1.0,   0.0,    0.0);
      gpuRayFloat_t speed = 4.0;
      Direction_t velocity{3.0, -2.0, 3.0};
      auto newDirAndSpeed = pCart->convertToCellReferenceFrame(velocity, pos, dir, speed);
      auto newSpeed = std::sqrt(1.0 + 2.0*2.0 + 3.0*3.0);
      GPU_CHECK_CLOSE(newSpeed, newDirAndSpeed.speed(), 1e-6);
      GPU_CHECK_CLOSE(1.0/newSpeed, newDirAndSpeed.direction()[0], 1e-6);
      GPU_CHECK_CLOSE(-3.0/newSpeed, newDirAndSpeed.direction()[2], 1e-6);
    }

    {
      Position_t pos( -0.5, -1.5,  0.5 );
      auto indices = pCart->calcIndices(pos);
      GPU_CHECK_EQUAL(0, indices[0]);
      GPU_CHECK_EQUAL(-1, indices[1]);
      GPU_CHECK_EQUAL(1, indices[2]);
    }

    {
      Position_t pos( -0.5, 0.2,  0.3 );
      Position_t dir(1, 0, 0);
      auto indices = pCart->calcIndices(pos);
      auto distAndDir = pCart->getMinDistToSurface(pos, dir, indices.data());
      GPU_CHECK_CLOSE(0.5, distAndDir.distance(), 1e-6);
      GPU_CHECK_EQUAL(0, distAndDir.dimension());

      dir = {0, -1, 0};
      distAndDir = pCart->getMinDistToSurface(pos, dir, indices.data());
      GPU_CHECK_CLOSE(0.2, distAndDir.distance(), 1e-6);
      GPU_CHECK_EQUAL(1, distAndDir.dimension());

      dir = {0, 0, 1};
      distAndDir = pCart->getMinDistToSurface(pos, dir, indices.data());
      GPU_CHECK_CLOSE(0.7, distAndDir.distance(), 1e-6);
      GPU_CHECK_EQUAL(2, distAndDir.dimension());
    }
  }

  TEST_FIXTURE( CartesianGridTesterTwo, RayTraceWithMovingGridComponentsOnGPU ) {
    bool* testVal;
    cudaMallocManaged(&testVal, sizeof(bool));
    kernelTestCartesianGrid<<<1, 1>>>(testVal, pCart.get());
    cudaDeviceSynchronize();
    CHECK(*testVal);
  }

#endif

}

} // end namespace
