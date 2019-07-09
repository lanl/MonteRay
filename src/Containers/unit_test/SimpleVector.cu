#include <UnitTest++.h>

#include <memory>

#include "SimpleVector.hh"

template <typename T>
using simple_vector = MonteRay::SimpleVector<T, std::allocator<T> >;

SUITE(SimpleVector_test) {

  simple_vector<double> vec1(3);
  TEST(Constructing_SimpleVector){
    simple_vector<double> vec2(3, 5.0);
    simple_vector<double> vec3{0.0, 1.0, 2.0};

    CHECK(vec1.size() == 3);
    CHECK(vec2.size() == 3);
    CHECK(vec3.size() == 3);

    CHECK(vec2[0] == 5.0);
    CHECK(vec2[1] == 5.0);
    CHECK(vec2[2] == 5.0);

    CHECK(vec3[0] == 0.0);
    CHECK(vec3[1] == 1.0);
    CHECK(vec3[2] == 2.0);

    simple_vector<int> vec{1, 2, 3};
    simple_vector<int> anotherVec(vec);
    CHECK(anotherVec[0] == 1);
    CHECK(anotherVec[1] == 2);
    CHECK(anotherVec[2] == 3);

    simple_vector<int> yetAnotherVec(std::move(anotherVec));
    CHECK(yetAnotherVec[0] == 1);
    CHECK(yetAnotherVec[1] == 2);
    CHECK(yetAnotherVec[2] == 3);
  }

  TEST(assignment){
    simple_vector<int> vec{1, 2, 3};
    simple_vector<int> anotherVec;
    anotherVec = vec;
    CHECK(anotherVec[0] == 1);
    CHECK(anotherVec[1] == 2);
    CHECK(anotherVec[2] == 3);

    simple_vector<int> yetAnotherVec;
    yetAnotherVec = std::move(anotherVec);
    CHECK(yetAnotherVec[0] == 1);
    CHECK(yetAnotherVec[1] == 2);
    CHECK(yetAnotherVec[2] == 3);
  }

  TEST(begin_end_back){
    auto start = &vec1[0];
    CHECK(start == &(*vec1.begin()));
    CHECK(start + 3 == &(*vec1.end()));
    CHECK(start + 2 == &(vec1.back()));
  }

  TEST(resize){
    vec1.resize(10);
    CHECK(vec1.size() == 10);
    vec1.resize(1);
    CHECK(vec1.size() == 1);
  }

  TEST(swap){
    simple_vector<double> vec1(3, 5.0);
    simple_vector<double> vec2{0.0, 1.0, 2.0, 3.0};
    vec1.swap(vec2);
    CHECK(vec1.size() == 4);
    CHECK(vec1[3] == 3.0);
    CHECK(vec2.size() == 3);
    CHECK(vec2[2] == 5.0);
  }
}


