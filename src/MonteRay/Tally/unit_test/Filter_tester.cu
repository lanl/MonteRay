#include <UnitTest++.h>

#include <iostream>
#include <iomanip>
#include <sstream>

#include <vector>

#include "MonteRayDefinitions.hh"
#include "GPUUtilityFunctions.hh"
#include "MonteRayTypes.hh"
#include "Filter.hh"
#include "Containers.hh"

SUITE( MonteRayFilter_tester ) {

  TEST( Constructor ){
    MonteRay::Vector<double> bins = {-100.0, 0.0, 1.5};
    MonteRay::BinEdgeFilter< MonteRay::Vector<double> > filter{std::move(bins)};
    CHECK_EQUAL(4, filter.size());
    CHECK_EQUAL(3, filter.binEdges().size());
    CHECK_EQUAL(-100.0, filter.binEdges()[0]);
    CHECK_EQUAL(0.0, filter.binEdges()[1]);
    CHECK_EQUAL(1.5, filter.binEdges()[2]);
  }

  TEST( ParensOperator ){
    MonteRay::Vector<double> bins = {-100.0, 0.0, 1.5};
    MonteRay::BinEdgeFilter< MonteRay::Vector<double> > filter{std::move(bins)};
    CHECK_EQUAL(0, filter(-111.0));
    CHECK_EQUAL(1, filter(-5.0));
    CHECK_EQUAL(2, filter(1.0));
    CHECK_EQUAL(3, filter(10.0));
  }

  TEST(ReadAndWrite){
    using Filter = MonteRay::BinEdgeFilter<MonteRay::Vector<double>>;
    Filter filter{ MonteRay::Vector<double>{0, 100, 200} };
    std::stringstream file;
    CHECK_EQUAL(4, filter.size());
    filter.write(file);
    auto newFilter = Filter::read(file);
    CHECK_EQUAL(4, newFilter.size());
    auto binEdges = newFilter.binEdges();
    CHECK_EQUAL(3, binEdges.size());
    CHECK_EQUAL(0.0, binEdges[0]);
    CHECK_EQUAL(100.0, binEdges[1]);
    CHECK_EQUAL(200.0, binEdges[2]);
  }

}
