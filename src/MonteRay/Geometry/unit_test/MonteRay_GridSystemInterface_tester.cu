#include <UnitTest++.h>

#include "MonteRay_GridSystemInterface.hh"
#include <array>

using namespace MonteRay;

namespace GridSystemInterfaceTester {


SUITE(GridSystemInterface_test) {

  std::array<MonteRay_GridBins, 3> gridBins{
    MonteRay_GridBins(0, 10, 10, MonteRay_GridBins::RADIAL),
    MonteRay_GridBins(-10, 10, 20),
    MonteRay_GridBins(-10, 10, 30)
  };

  TEST(ConstructorsAndGetters){

    MonteRay_GridSystemInterface grid{gridBins, 3};
    CHECK_EQUAL(3, grid.getDimension());
    CHECK_EQUAL(10, grid.getNumGridBins(0));
    CHECK_EQUAL(20, grid.getNumGridBins(1));
    CHECK_EQUAL(30, grid.getNumGridBins(2));

    CHECK_EQUAL(1, grid.getMinVertex(0));
    CHECK_EQUAL(-10, grid.getMinVertex(1));
    CHECK_EQUAL(10, grid.getMaxVertex(0));
    CHECK_EQUAL(10, grid.getMaxVertex(1));

    CHECK_EQUAL(-4.0, grid.getVertex(1, 6));

    CHECK_EQUAL(10, grid.getNumVertices(0));
    CHECK_EQUAL(10, grid.getNumVerticesSq(0));
    CHECK_EQUAL(0, grid.getNumVerticesSq(1));
    CHECK_EQUAL(10*20*30, grid.getNumCells());

    std::array<int, 3> indices{9, 19, 29};
    CHECK_EQUAL(false, grid.isOutside(indices.data()));
    indices = {10, 19, 29};
    CHECK_EQUAL(true, grid.isOutside(indices.data()));
    indices = {9, 20, 29};
    CHECK_EQUAL(true, grid.isOutside(indices.data()));
    indices = {9, 19, 30};
    CHECK_EQUAL(true, grid.isOutside(indices.data()));

    indices = {0, 0, 0};
    CHECK_EQUAL(0, grid.calcIndex(indices));
    indices = {0, 2, 0};
    CHECK_EQUAL(10*2, grid.calcIndex(indices));
    indices = {0, 0, 2};
    CHECK_EQUAL(10*20*2, grid.calcIndex(indices));
    indices = {9, 19, 29};
    CHECK_EQUAL(10*20*30 - 1, grid.calcIndex(indices));

  }

#ifdef __CUDACC__
  __global__ void testSpatialGrid(bool* pval, MonteRay_GridSystemInterface* pGrid) {
    *pval = true;

    *pval = *pval && (3 == pGrid->getDimension());
    *pval = *pval && (10 == pGrid->getNumGridBins(0));
    *pval = *pval && (20 == pGrid->getNumGridBins(1));
    *pval = *pval && (30 == pGrid->getNumGridBins(2));

    *pval = *pval && (1 == pGrid->getMinVertex(0));
    *pval = *pval && (-10 == pGrid->getMinVertex(1));
    *pval = *pval && (10 == pGrid->getMaxVertex(0));
    *pval = *pval && (10 == pGrid->getMaxVertex(1));

    *pval = *pval && (-4.0 == pGrid->getVertex(1, 6));

    *pval = *pval && (10 == pGrid->getNumVertices(0));
    *pval = *pval && (10 == pGrid->getNumVerticesSq(0));
    *pval = *pval && (0 == pGrid->getNumVerticesSq(1));
    *pval = *pval && (10*20*30 == pGrid->getNumCells());

    Array<int, 3> indices{9, 19, 29};
    *pval = *pval && (false == pGrid->isOutside(indices.data()));
    indices = {10, 19, 29};
    *pval = *pval && (true == pGrid->isOutside(indices.data()));
    indices = {9, 20, 29};
    *pval = *pval && (true == pGrid->isOutside(indices.data()));
    indices = {9, 19, 30};
    *pval = *pval && (true == pGrid->isOutside(indices.data()));

    indices = {0, 0, 0};
    *pval = *pval && (0 == pGrid->calcIndex(indices));
    indices = {0, 2, 0};
    *pval = *pval && (10*2 == pGrid->calcIndex(indices));
    indices = {0, 0, 2};
    *pval = *pval && (10*20*2 == pGrid->calcIndex(indices));
    indices = {9, 19, 29};
    *pval = *pval && (10*20*30 == pGrid->calcIndex(indices));
  }

  TEST(ConstructorsAndGettersOnGPU){
    auto upGrid = std::make_unique<MonteRay_GridSystemInterface>(gridBins, 3);
    auto pGrid = upGrid.get();
    bool* pTestVal;
    cudaMallocManaged(&pTestVal, sizeof(bool));
    testSpatialGrid<<<1, 1>>>(pTestVal, pGrid);
    CHECK(pTestVal);
    cudaFree(pTestVal);
  }
#endif
}

} // end namespace
