#include <UnitTest++.h>

#include "MonteRay_CartesianGrid.t.hh"
#include "MonteRay_SpatialGrid.hh"
#include "MonteRayVector3D.hh"
#include "MonteRay_GridBins.hh"
#include "MonteRayDefinitions.hh"
#include "MonteRayCopyMemory.t.hh"
#include "RayWorkInfo.hh"
#include "MaterialProperties.hh"
#include "UnitTestHelper.hh"

namespace MonteRay_CartesianGrid_rayTraceWithMovingMaterials_tests{

using namespace MonteRay;

SUITE( MonteRay_CartesianGrid_rayTraceWithMovingMaterials_Tests) {

  using Position_t = Vector3D<gpuRayFloat_t>;
  using Direction_t = Vector3D<gpuRayFloat_t>;
  using GridBins_t = MonteRay_GridBins;
  using CartesianGrid = MonteRay_CartesianGrid;

  class CartesianGridTester{
    public:
    std::unique_ptr<CartesianGrid> pCart;

    CartesianGridTester(){
      std::vector<gpuRayFloat_t> vertices{-1, 0, 1};
      pCart = std::make_unique<CartesianGrid>(3, GridBins_t{vertices}, GridBins_t{vertices}, GridBins_t{vertices});
    }
  };

  TEST_FIXTURE( CartesianGridTester, ConvertToCellReferenceFrame ) {
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

  TEST_FIXTURE( CartesianGridTester, CalcIndices ){
    Position_t pos( -0.5, -1.5,  0.5 );
    auto indices = pCart->calcIndices(pos);
    CHECK_EQUAL(0, indices[0]);
    CHECK_EQUAL(-1, indices[1]);
    CHECK_EQUAL(1, indices[2]);
  }

  TEST_FIXTURE( CartesianGridTester, GetMinDistToSurface){
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

  TEST_FIXTURE( CartesianGridTester, RayTraceWithMovingGridComponentsOnGPU ) {
    bool* testVal;
    cudaMallocManaged(&testVal, sizeof(bool));
    kernelTestCartesianGrid<<<1, 1>>>(testVal, pCart.get());
    cudaDeviceSynchronize();
    CHECK(*testVal);
  }

#endif

}

} // end namespace

