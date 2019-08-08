#include <UnitTest++.h>

#include <memory>
#include <tuple>

#include "SimpleVector.hh"
#include "GPUUtilityFunctions.hh"


namespace SimpleVectorTest{

using namespace MonteRay;
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
    CHECK(vec1.capacity() == 10);
    vec1.resize(1);
    CHECK(vec1.size() == 1);
    CHECK(vec1.capacity() == 10);
  }

  TEST(swap){
    simple_vector<double> vec1(3, 5.0);
    simple_vector<double> vec2{0.0, 1.0, 2.0, 3.0};
    vec1.swap(vec2);
    CHECK(vec1.size() == 4);
    CHECK(vec1.capacity() == 4);
    CHECK(vec1[3] == 3.0);
    CHECK(vec2.size() == 3);
    CHECK(vec2.capacity() == 3);
    CHECK(vec2[2] == 5.0);
  }

  TEST(erase){
    simple_vector<int> vec{0, 1, 2, 3, 4, 5};
    vec.erase(vec.begin(), vec.begin() + 3);
    CHECK(vec.size() == 3);
    CHECK(vec.capacity() == 6);
    CHECK(vec[0] == 3);
    CHECK(vec[1] == 4);
    CHECK(vec[2] == 5);
  }

  TEST(assign){
    simple_vector<int> vec;
    std::vector<int> another_vec{0, 1, 2, 3};
    vec.assign(another_vec.begin(), another_vec.end());
    int i = 0;
    for (auto& val : vec) {
      CHECK_EQUAL(i, val);
      i++;
    }
    CHECK(vec.capacity() == 4);
    another_vec = {-1};
    vec.assign(another_vec.begin(), another_vec.end());
    CHECK_EQUAL(-1, vec[0]);
    CHECK_EQUAL(1, vec.size());
    CHECK(vec.capacity() == 4);
  }

  TEST(adding_to_vector){
    using T = std::tuple<double, int>;
    simple_vector<T> vec;
    T tup(1.5, 5);
    vec.push_back(tup);
    CHECK(vec.size() == 1);
    CHECK(vec.capacity() == 1);
    CHECK(std::get<0>(vec[0]) == 1.5);
    CHECK(std::get<1>(vec[0]) == 5);
    vec.emplace_back(3.5, 5);
    CHECK(std::get<0>(vec[1]) == 3.5);
    CHECK(std::get<1>(vec[1]) == 5);
    CHECK(vec.size() == 2);
    CHECK(vec.capacity() == 2);
    const T anotherTup(4.0, 6);
    vec.push_back(anotherTup);
    CHECK(std::get<0>(vec[2]) == 4.0);
    CHECK(std::get<1>(vec[2]) == 6);;
  }


#ifdef __CUDACC__
  __global__ void vecKernel(SimpleVector<int>* vec){
    for (auto& val : *vec){
      val *= 2;
    }
  }

  TEST(accessing_cuda_data_on_gpu){
    auto vec = std::make_unique<SimpleVector<int>>(3);
    *vec = SimpleVector<int>{1, 2, 3};

    vecKernel<<<1, 1>>>(vec.get());
    cudaDeviceSynchronize();
    CHECK_EQUAL((*vec)[0], 2);
    CHECK_EQUAL((*vec)[1], 4);
    CHECK_EQUAL((*vec)[2], 6);
  }
#endif

  TEST(inserting_into_vector){
    simple_vector<int> vecA{0, 1, 2};
    std::vector<int> vecB{3, 4, 5};
    vecA.insert(vecA.begin() + 1, vecB.begin(), vecB.end());

    CHECK_EQUAL(vecA.size(), 6);
    CHECK_EQUAL(vecA.capacity(), 6);
    CHECK_EQUAL(0, vecA[0]);
    CHECK_EQUAL(3, vecA[1]);
    CHECK_EQUAL(4, vecA[2]);
    CHECK_EQUAL(5, vecA[3]);
    CHECK_EQUAL(1, vecA[4]);
    CHECK_EQUAL(2, vecA[5]);
  }


}

} // end namespace SimpleVectorTest


