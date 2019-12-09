#include <UnitTest++.h>

#include "MonteRay_GridBins.hh"

using namespace MonteRay;

SUITE( MonteRay_GridBins_Tester ) {

    TEST( default_ctor ) {
      auto bins = MonteRay_GridBins();
    }

    TEST( ctor_takes_min_and_max ) {
      const auto bins = MonteRay_GridBins(-10, 10, 20);
      CHECK_CLOSE( -10.0, bins.getMinVertex(), 1e-11 );
      CHECK_CLOSE( 10.0, bins.getMaxVertex(), 1e-11 );
    }

    TEST( ctor_with_vector ) {
        std::vector<gpuRayFloat_t> vertices = {
                -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
                0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10
        };

        const auto bins = MonteRay_GridBins(vertices);
        CHECK_CLOSE( -10.0, bins.getMinVertex(), 1e-11 );
        CHECK_CLOSE( 10.0, bins.getMaxVertex(), 1e-11 );
        CHECK_CLOSE( 20.0, bins.getNumBins(), 1e-11 );
        CHECK_CLOSE( -10.0, bins.vertices[0], 1e-11);
        CHECK_CLOSE( -9.0, bins.vertices[1], 1e-11);
    }


  const auto linearBins = MonteRay_GridBins(-10, 10, 20);
  TEST(LinearGridBins_Getters){

    CHECK(linearBins.isLinear());
    CHECK(not linearBins.isRadial());

    CHECK_EQUAL(20, linearBins.getNumBins());

    CHECK_EQUAL(-10, linearBins.getMinVertex());
    CHECK_EQUAL(10, linearBins.getMaxVertex());
    CHECK_EQUAL(21, linearBins.getNumVertices());
    CHECK_EQUAL(0, linearBins.getNumVerticesSq());

    const auto pVertices = linearBins.getVerticesData();
    CHECK_EQUAL(-4.0, pVertices[6]);


    CHECK_EQUAL(19, linearBins.getLinearIndex(9.5));
    CHECK_EQUAL(10, linearBins.getLinearIndex(0.1));
    CHECK_EQUAL(0, linearBins.getLinearIndex(-9.9));
    CHECK_EQUAL(-1, linearBins.getLinearIndex(-44.0));
    CHECK_EQUAL(20, linearBins.getLinearIndex(44.0));

  }

  const auto radialBins = MonteRay_GridBins(0, 10, 10, MonteRay_GridBins::RADIAL);
  TEST(RadialGridBins_Getters){

    CHECK(radialBins.isRadial());
    CHECK(not radialBins.isLinear());

    CHECK_EQUAL(10, radialBins.getNumBins());

    CHECK_EQUAL(10, radialBins.getNumVertices());
    CHECK_EQUAL(10, radialBins.getNumVerticesSq());

    const auto pVertices = radialBins.getVerticesData();
    CHECK_EQUAL(1.0, pVertices[0]);
    CHECK_EQUAL(10.0, pVertices[9]);

    CHECK_EQUAL(1, radialBins.getMinVertex());
    CHECK_EQUAL(10, radialBins.getMaxVertex());
  }

   
  TEST(RadialGridBins_CalcIndex){
    CHECK_EQUAL(0, radialBins.getRadialIndexFromR(0.1));
    CHECK_EQUAL(9, radialBins.getRadialIndexFromR(9.9));
    CHECK_EQUAL(10, radialBins.getRadialIndexFromR(10.1));

    CHECK_EQUAL(0, radialBins.getRadialIndexFromRSq(0.1*0.1));
    CHECK_EQUAL(9, radialBins.getRadialIndexFromRSq(9.9*9.9));
    CHECK_EQUAL(10, radialBins.getRadialIndexFromRSq(10.1*10.1));
  }

  TEST(IsIndexOutside){
    CHECK( radialBins.isIndexOutside(-1) ); // even though it doesn't make sense to have a negative radial index, a negative index is certainly not inside the grid.
    CHECK( radialBins.isIndexOutside(10) );
    CHECK( not radialBins.isIndexOutside(0) );
    CHECK( not radialBins.isIndexOutside(1) );
    CHECK( not radialBins.isIndexOutside(9) );

    CHECK( linearBins.isIndexOutside(20) );
    CHECK( linearBins.isIndexOutside(-1) );
    CHECK( not linearBins.isIndexOutside(19) );
    CHECK( not linearBins.isIndexOutside(0) );
  }

  TEST( LinearBins_DistanceToGetInsideLinearMesh ) {
    CHECK_CLOSE(2.0, linearBins.distanceToGetInsideLinearMesh(-11.0, 0.5), 1E-5);
    CHECK_CLOSE(4.0, linearBins.distanceToGetInsideLinearMesh(11.0, -0.25), 1E-5);
    CHECK_CLOSE(std::numeric_limits<gpuRayFloat_t>::epsilon(), linearBins.distanceToGetInsideLinearMesh(10.0, -0.25), 1E-5);
    CHECK_CLOSE(std::numeric_limits<gpuRayFloat_t>::epsilon(), linearBins.distanceToGetInsideLinearMesh(-10.0, 0.25), 1E-5);
    CHECK_EQUAL( std::numeric_limits<gpuRayFloat_t>::infinity(), linearBins.distanceToGetInsideLinearMesh(-11, -0.1));
    CHECK_EQUAL( std::numeric_limits<gpuRayFloat_t>::infinity(), linearBins.distanceToGetInsideLinearMesh(11, 0.1));
  }

  TEST( ReadWrite ){

    std::stringstream stream;
    const auto linearBins = MonteRay_GridBins(-1, 1, 2);
    linearBins.write(stream);
    const auto otherLinearBins = MonteRay_GridBins::read(stream);
    CHECK_EQUAL(3, otherLinearBins.getNumVertices());
    CHECK_EQUAL(0, otherLinearBins.getNumVerticesSq());
    CHECK(otherLinearBins.isLinear());
    {
      const auto pVertices = otherLinearBins.getVerticesData();
      CHECK_EQUAL(-1, pVertices[0]);
      CHECK_EQUAL(0, pVertices[1]);
      CHECK_EQUAL(1, pVertices[2]);
    }
    
    const auto radialBins = MonteRay_GridBins(0, 2, 2, MonteRay_GridBins::RADIAL);
    radialBins.write(stream);
    const auto otherRadialBins = MonteRay_GridBins::read(stream);
    CHECK_EQUAL(2, otherRadialBins.getNumVertices());
    CHECK_EQUAL(2, otherRadialBins.getNumVerticesSq());
    CHECK(otherRadialBins.isRadial());
    {
      const auto pVertices = otherRadialBins.getVerticesData();
      CHECK_EQUAL(1, pVertices[0]);
      CHECK_EQUAL(2, pVertices[1]);
      const auto pVerticesSq = otherRadialBins.getVerticesSqData();
      CHECK_EQUAL(1, pVerticesSq[0]);
      CHECK_EQUAL(4, pVerticesSq[1]);
    }
  }


#ifdef __CUDACC__
  __global__ void testMonteRayGridBinsOnGPU(bool* testVal, MonteRay_GridBins* pRadialBins){
    *testVal = true;
    *testVal = *testVal && (pRadialBins->isRadial());
    *testVal = *testVal && (not pRadialBins->isLinear());

    *testVal = *testVal && (10 == pRadialBins->getNumBins());

    *testVal = *testVal && (10 == pRadialBins->getNumVertices());
    *testVal = *testVal && (10 == pRadialBins->getNumVerticesSq());

    const auto pVertices = pRadialBins->getVerticesData();
    *testVal = *testVal && (1.0 == pVertices[0]);
    *testVal = *testVal && (10.0 == pVertices[9]);

    *testVal = *testVal && (1 == pRadialBins->getMinVertex());
    *testVal = *testVal && (10 == pRadialBins->getMaxVertex());

    *testVal = *testVal && (0 == pRadialBins->getRadialIndexFromR(0.1));
    *testVal = *testVal && (9 == pRadialBins->getRadialIndexFromR(9.9));
    *testVal = *testVal && (10 == pRadialBins->getRadialIndexFromR(10.1));

    *testVal = *testVal && (0 == pRadialBins->getRadialIndexFromRSq(0.1*0.1));
    *testVal = *testVal && (9 == pRadialBins->getRadialIndexFromRSq(9.9*9.9));
    *testVal = *testVal && (10 == pRadialBins->getRadialIndexFromRSq(10.1*10.1));

    *testVal = *testVal && ( pRadialBins->isIndexOutside(-1) ); // even though it doesn't make sense to have a negative radial index, a negative index is certainly not inside the grid.
    *testVal = *testVal && ( pRadialBins->isIndexOutside(10) );
    *testVal = *testVal && ( not pRadialBins->isIndexOutside(0) );
    *testVal = *testVal && ( not pRadialBins->isIndexOutside(1) );
    *testVal = *testVal && ( not pRadialBins->isIndexOutside(9) );
  }

  TEST(GPU_RadialGridBins){
    auto pRadialBins = std::make_unique<MonteRay_GridBins>(radialBins);
    bool* testVal;
    cudaMallocManaged(&testVal, sizeof(bool));
    testMonteRayGridBinsOnGPU<<<1, 1>>>(testVal, pRadialBins.get());
    CHECK(testVal);
  }

#endif

}
