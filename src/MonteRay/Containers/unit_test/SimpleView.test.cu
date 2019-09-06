#include <UnitTest++.h>

#include <vector>
#include <array>

#include "SimpleView.hh"

namespace SimpleViewTest{

using namespace MonteRay;

SUITE(SimpleView_test) {

  std::vector<int> vec{0, 1, 2, 3};
  TEST(Constructing_SimpleViewFromContainer){
    SimpleView<int> view(vec);
    CHECK_EQUAL(0, view[0]);
    CHECK_EQUAL(1, view[1]);
    CHECK_EQUAL(2, view[2]);
    CHECK_EQUAL(3, view[3]);
  }
  TEST(Constructing_SimpleViewFromPointers){
    std::array<int, 3> array{0, 1, 2};
    SimpleView<int> view(&array[0], &array[0] + 3);
    CHECK_EQUAL(0, view[0]);
    CHECK_EQUAL(1, view[1]);
    CHECK_EQUAL(2, view[2]);
  }

  TEST(SimpleViewRangeBasedFor){
    SimpleView<int> view(vec);
    for (auto& val: view){
      val *= 2;
    }
    CHECK_EQUAL(vec[0], 0);
    CHECK_EQUAL(vec[1], 2);
    CHECK_EQUAL(vec[2], 4);
    CHECK_EQUAL(vec[3], 6);
  }

}

} // end namespace SimpleViewTest


