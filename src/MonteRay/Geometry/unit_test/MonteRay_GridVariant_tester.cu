#include <UnitTest++.h>
#include <iostream>

#include "MonteRay_GridBins.hh"
#include "MonteRay_GridVariant.hh"

using namespace MonteRay;

namespace MonteRay_GridVariant_tester{

SUITE( MonteRay_GridVariant_tester ) {

  TEST( Construct_and_visit ) {

    auto func = [](auto&& geom){
      return geom.getNumBins(0);
    };

    std::array<MonteRay_GridBins, 3> cartGridBins{
      MonteRay_GridBins{-10.0, 10.0, 10},
      MonteRay_GridBins{-10.0, 10.0, 10},
      MonteRay_GridBins{-10.0, 10.0, 10}
    };

    MonteRay::GridVariant gridVariant{TransportMeshType::Cartesian, cartGridBins};

    auto val = gridVariant.visit(func);
    CHECK_EQUAL(10, val);

    std::array<MonteRay_GridBins, 3> cylGridBins{
      MonteRay_GridBins{0.0, 10.0, 10, MonteRay_GridBins::RADIAL},
      MonteRay_GridBins{-10.0, 10.0, 10},
      MonteRay_GridBins{-10.0, 10.0, 10}
    };

    gridVariant = MonteRay::GridVariant{TransportMeshType::Cylindrical, cylGridBins};
    val = gridVariant.visit(func);
    CHECK_EQUAL(10, val);
  }
}

}
