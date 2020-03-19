#include <UnitTest++.h>

#include <memory>

#include "MonteRay_GridSystemInterface.hh"
#include "UnitTestHelper.hh"
#include <array>

using namespace MonteRay;

namespace GridSystemInterfaceTester {


SUITE(GridSystemInterface_test) {

  class GridSystemInterfaceTester{
    public:
    std::unique_ptr<MonteRay_GridSystemInterface> pGrid;

    GridSystemInterfaceTester(){

      std::array<MonteRay_GridBins, 3> gridBins{
        MonteRay_GridBins(0, 10, 10, MonteRay_GridBins::RADIAL),
        MonteRay_GridBins(-10, 10, 20),
        MonteRay_GridBins(-10, 10, 30)
      };

      pGrid = std::make_unique<MonteRay_GridSystemInterface>(gridBins, 3);
    }
  };

  TEST_FIXTURE(GridSystemInterfaceTester, ConstructorsAndGetters){

    CHECK_EQUAL(3, pGrid->getDimension());
    CHECK_EQUAL(10, pGrid->getNumGridBins(0));
    CHECK_EQUAL(20, pGrid->getNumGridBins(1));
    CHECK_EQUAL(30, pGrid->getNumGridBins(2));

    CHECK_EQUAL(1, pGrid->getMinVertex(0));
    CHECK_EQUAL(-10, pGrid->getMinVertex(1));
    CHECK_EQUAL(10, pGrid->getMaxVertex(0));
    CHECK_EQUAL(10, pGrid->getMaxVertex(1));

    CHECK_EQUAL(-4.0, pGrid->getVertex(1, 6));

    CHECK_EQUAL(10, pGrid->getNumVertices(0));
    CHECK_EQUAL(10, pGrid->getNumVerticesSq(0));
    CHECK_EQUAL(0, pGrid->getNumVerticesSq(1));
    CHECK_EQUAL(10*20*30, pGrid->getNumCells());

    std::array<int, 3> indices{9, 19, 29};
    CHECK(not pGrid->isOutside(indices.data()));
    indices = {10, 19, 29};
    CHECK(pGrid->isOutside(indices.data()));
    indices = {9, 20, 29};
    CHECK(pGrid->isOutside(indices.data()));
    indices = {9, 19, 30};
    CHECK(pGrid->isOutside(indices.data()));

    indices = {0, 0, 0};
    CHECK_EQUAL(0, pGrid->calcIndex(indices));
    indices = {0, 2, 0};
    CHECK_EQUAL(10*2, pGrid->calcIndex(indices));
    indices = {0, 0, 2};
    CHECK_EQUAL(10*20*2, pGrid->calcIndex(indices));
    indices = {9, 19, 29};
    CHECK_EQUAL(10*20*30 - 1, pGrid->calcIndex(indices));

  }

#ifdef __CUDACC__
  __global__ void testSpatialGrid(bool* testVal, MonteRay_GridSystemInterface* pGrid) {
    *testVal = true;

    GPU_CHECK_EQUAL(3, pGrid->getDimension());
    GPU_CHECK_EQUAL(10, pGrid->getNumGridBins(0));
    GPU_CHECK_EQUAL(20, pGrid->getNumGridBins(1));
    GPU_CHECK_EQUAL(30, pGrid->getNumGridBins(2));

    GPU_CHECK_EQUAL(1, pGrid->getMinVertex(0));
    GPU_CHECK_EQUAL(-10, pGrid->getMinVertex(1));
    GPU_CHECK_EQUAL(10, pGrid->getMaxVertex(0));
    GPU_CHECK_EQUAL(10, pGrid->getMaxVertex(1));

    GPU_CHECK_EQUAL(-4.0, pGrid->getVertex(1, 6));

    GPU_CHECK_EQUAL(10, pGrid->getNumVertices(0));
    GPU_CHECK_EQUAL(10, pGrid->getNumVerticesSq(0));
    GPU_CHECK_EQUAL(0, pGrid->getNumVerticesSq(1));
    GPU_CHECK_EQUAL(10*20*30, pGrid->getNumCells());

    Array<int, 3> indices{9, 19, 29};
    GPU_CHECK(not pGrid->isOutside(indices.data()));
    indices = {10, 19, 29};
    GPU_CHECK(pGrid->isOutside(indices.data()));
    indices = {9, 20, 29};
    GPU_CHECK(pGrid->isOutside(indices.data()));
    indices = {9, 19, 30};
    GPU_CHECK(pGrid->isOutside(indices.data()));

    indices = {0, 0, 0};
    GPU_CHECK_EQUAL(0, pGrid->calcIndex(indices));
    indices = {0, 2, 0};
    GPU_CHECK_EQUAL(10*2, pGrid->calcIndex(indices));
    indices = {0, 0, 2};
    GPU_CHECK_EQUAL(10*20*2, pGrid->calcIndex(indices));
    indices = {9, 19, 29};
    GPU_CHECK_EQUAL(10*20*30 - 1, pGrid->calcIndex(indices));
  }

  TEST_FIXTURE(GridSystemInterfaceTester, ConstructorsAndGettersOnGPU){
    bool* pTestVal;
    cudaMallocManaged(&pTestVal, sizeof(bool));
    testSpatialGrid<<<1, 1>>>(pTestVal, pGrid.get());
    cudaDeviceSynchronize();
    CHECK(*pTestVal);
    cudaFree(pTestVal);
  }
#endif
}

} // end namespace
