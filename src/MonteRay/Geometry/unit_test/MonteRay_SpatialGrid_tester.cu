#include <UnitTest++.h>

#include "MonteRay_SpatialGrid.hh"
#include <type_traits>
#include "UnitTestHelper.hh"

using namespace MonteRay;

namespace MonteRaySpatialGridTester{


SUITE( MonteRay_SpatialGrid_Tester ) {
  using Grid_t = MonteRay_SpatialGrid;

  class SpatialGridTester{
    public:
      std::array<MonteRay_GridBins, 3> cartGridBins{
        MonteRay_GridBins{-10.0, 10.0, 1},
        MonteRay_GridBins{-10.0, 10.0, 2},
        MonteRay_GridBins{-10.0, 10.0, 3}
      };

      std::array<MonteRay_GridBins, 3> cylGridBins{
        MonteRay_GridBins{0.0, 10.0, 10, MonteRay_GridBins::RADIAL},
        MonteRay_GridBins{-10.0, 10.0, 20},
        MonteRay_GridBins{-10.0, 10.0, 30}
      };
  };

  TEST_FIXTURE(SpatialGridTester, ConstructorsAndGetters){
    auto grid = MonteRay_SpatialGrid(TransportMeshType::Cartesian, cartGridBins);
    CHECK(TransportMeshType::Cartesian == grid.getCoordinateSystem());
    CHECK_EQUAL(3, grid.getDimension());
    CHECK_EQUAL(1, grid.getNumGridBins(0));
    CHECK_EQUAL(2, grid.getNumGridBins(1));
    CHECK_EQUAL(3, grid.getNumGridBins(2));

    grid = MonteRay_SpatialGrid(TransportMeshType::Cylindrical, cylGridBins);
    CHECK(TransportMeshType::Cylindrical == grid.getCoordinateSystem());
    CHECK_EQUAL(2, grid.getDimension());
    CHECK_EQUAL(10, grid.getNumGridBins(0));
    CHECK_EQUAL(20, grid.getNumGridBins(1));

    CHECK_EQUAL(1, grid.getMinVertex(0));
    CHECK_EQUAL(-10, grid.getMinVertex(1));
    CHECK_EQUAL(10, grid.getMaxVertex(0));
    CHECK_EQUAL(10, grid.getMaxVertex(1));

    CHECK_EQUAL(-4.0, grid.getVertex(1, 6));

    MonteRay_SpatialGrid::Position_t pos{9.5, 0, 9.5};
    CHECK_EQUAL(199, grid.getIndex(pos));

    CHECK_CLOSE(M_PI*(10*10 - 9*9), grid.getVolume(199), 1E-5);
    CHECK_EQUAL(10, grid.getNumVertices(0));
    CHECK_EQUAL(10, grid.getNumVerticesSq(0));
    CHECK_EQUAL(0, grid.getNumVerticesSq(1));
  }

#ifdef __CUDACC__
  __global__ void testSpatialGrid(bool* testVal, MonteRay_SpatialGrid* pGrid) {
    *testVal = true;

    GPU_CHECK(2 == pGrid->getDimension());
    GPU_CHECK(10 == pGrid->getNumGridBins(0));
    GPU_CHECK(20 == pGrid->getNumGridBins(1));

    GPU_CHECK(1 == pGrid->getMinVertex(0));
    GPU_CHECK(-10 == pGrid->getMinVertex(1));
    GPU_CHECK(10 == pGrid->getMaxVertex(0));
    GPU_CHECK(10 == pGrid->getMaxVertex(1));

    GPU_CHECK(-4.0 == pGrid->getVertex(1, 6));

    MonteRay_SpatialGrid::Position_t pos{9.5, 0, 9.5};
    GPU_CHECK(199 == pGrid->getIndex(pos));

    GPU_CHECK(10 == pGrid->getNumVertices(0));
    GPU_CHECK(10 == pGrid->getNumVerticesSq(0));
    GPU_CHECK(0 == pGrid->getNumVerticesSq(1));
  }

  TEST_FIXTURE(SpatialGridTester, GettersOnGPU){
    auto upGrid = std::make_unique<MonteRay_SpatialGrid>(TransportMeshType::Cylindrical, cylGridBins);
    auto pGrid = upGrid.get();
    bool* pTestVal;
    cudaMallocManaged(&pTestVal, sizeof(bool));
    *pTestVal = false;
    testSpatialGrid<<<1, 1>>>(pTestVal, pGrid);
    cudaDeviceSynchronize();
    CHECK(*pTestVal);
    cudaFree(pTestVal);
  }

#endif

}

}// end namespace 
