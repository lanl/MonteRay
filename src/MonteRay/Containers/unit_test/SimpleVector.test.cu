#include <UnitTest++.h>

#include <memory>
#include <tuple>

#include "SimpleVector.hh"
#include "GPUUtilityFunctions.hh"

template <typename T>
using SimpleVector = MonteRay::SimpleVector<T, std::allocator<T> >;
template <typename T>
using CudaSimpleVector = MonteRay::SimpleVector<T>;

SUITE(SimpleVector_test) {

  SimpleVector<double> vec1(3);
  TEST(Constructing_SimpleVector){
    SimpleVector<double> vec2(3, 5.0);
    SimpleVector<double> vec3{0.0, 1.0, 2.0};

    CHECK(vec1.size() == 3);
    CHECK(vec2.size() == 3);
    CHECK(vec3.size() == 3);

    CHECK(vec2[0] == 5.0);
    CHECK(vec2[1] == 5.0);
    CHECK(vec2[2] == 5.0);

    CHECK(vec3[0] == 0.0);
    CHECK(vec3[1] == 1.0);
    CHECK(vec3[2] == 2.0);

    SimpleVector<int> vec{1, 2, 3};
    SimpleVector<int> anotherVec(vec);
    CHECK(anotherVec[0] == 1);
    CHECK(anotherVec[1] == 2);
    CHECK(anotherVec[2] == 3);

    SimpleVector<int> yetAnotherVec(std::move(anotherVec));
    CHECK(yetAnotherVec[0] == 1);
    CHECK(yetAnotherVec[1] == 2);
    CHECK(yetAnotherVec[2] == 3);
  }

  TEST(ConstructingSimpleVectorFromVector){
    SimpleVector<int> vector(std::vector<int>{0, 1, 2});
    CHECK_EQUAL(3, vector.size());
    CHECK_EQUAL(0, vector[0]);
    CHECK_EQUAL(1, vector[1]);
    CHECK_EQUAL(2, vector[2]);
  }

  TEST(assignment){
    SimpleVector<int> vec{1, 2, 3};
    SimpleVector<int> anotherVec;
    anotherVec = vec;
    CHECK(anotherVec[0] == 1);
    CHECK(anotherVec[1] == 2);
    CHECK(anotherVec[2] == 3);

    SimpleVector<int> yetAnotherVec;
    yetAnotherVec = std::move(anotherVec);
    CHECK(yetAnotherVec[0] == 1);
    CHECK(yetAnotherVec[1] == 2);
    CHECK(yetAnotherVec[2] == 3);
  }

  TEST(begin_end_back){
    const auto start = &vec1[0];
    CHECK(start == &(*vec1.cbegin()));
    CHECK(start + 3 == &(*vec1.cend()));
    CHECK(start == &(*vec1.begin()));
    CHECK(start + 3 == &(*vec1.end()));
    CHECK(start + 2 == &(vec1.back()));
  }

  TEST(resize){
    struct foo{
      double bar = 1;
      foo(){ bar = 10; }
    };
    SimpleVector<foo> vec;
    vec.resize(10);
    CHECK(vec.size() == 10);
    CHECK(vec.capacity() == 10);
    for (auto& val : vec){
      CHECK(val.bar == 10);
    }
    vec.resize(1);
    CHECK(vec.size() == 1);
    CHECK(vec.capacity() == 10);
  }

  TEST(resizeWithoutConstructing){
    struct foo{
      double bar = 1;
      foo(){ bar = -10; }
    };

    SimpleVector<foo> vec;
    vec.resizeWithoutConstructing(10);
    for (auto& val : vec){
      CHECK(val.bar != -10);
    }
    CHECK(vec.size() == 10);
    CHECK(vec.capacity() == 10);
    vec.resizeWithoutConstructing(1);
    CHECK(vec.size() == 1);
    CHECK(vec.capacity() == 10);
  }

  TEST(swap){
    SimpleVector<double> vec1(3, 5.0);
    SimpleVector<double> vec2{0.0, 1.0, 2.0, 3.0};
    vec1.swap(vec2);
    CHECK(vec1.size() == 4);
    CHECK(vec1.capacity() == 4);
    CHECK(vec1[3] == 3.0);
    CHECK(vec2.size() == 3);
    CHECK(vec2.capacity() == 3);
    CHECK(vec2[2] == 5.0);
  }

  TEST(erase){
    SimpleVector<int> vec{0, 1, 2, 3, 4, 5};
    vec.erase(vec.begin(), vec.begin() + 3);
    CHECK(vec.size() == 3);
    CHECK(vec.capacity() == 6);
    CHECK(vec[0] == 3);
    CHECK(vec[1] == 4);
    CHECK(vec[2] == 5);
  }

  TEST(assign){
    SimpleVector<int> vec;
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
    SimpleVector<T> vec;
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
  __global__ void vecKernel(CudaSimpleVector<int>* vec){
    for (auto& val : *vec){
      val *= 2;
    }
  }

  TEST(accessing_cuda_data_on_gpu){
    auto vec = std::make_unique<CudaSimpleVector<int>>(3);
    *vec = CudaSimpleVector<int>{1, 2, 3};
    CudaSimpleVector<int>* pVec;
    cudaMallocManaged(&pVec, sizeof(pVec));
    *pVec = CudaSimpleVector<int>{1, 2, 3};

    vecKernel<<<1, 1>>>(pVec);
    cudaDeviceSynchronize();
    CHECK_EQUAL(2, (*pVec)[0]);
    CHECK_EQUAL(4, (*pVec)[1]);
    CHECK_EQUAL(6, (*pVec)[2]);
  }
#endif

  TEST(inserting_into_vector){
    SimpleVector<int> vecA{0, 1, 2};
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
