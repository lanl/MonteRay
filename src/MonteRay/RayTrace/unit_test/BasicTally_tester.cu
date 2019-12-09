#include <UnitTest++.h>

#include "BasicTally.hh"
#include "ReadAndWriteFiles.hh"

#include <memory>
#include <sstream>

namespace BasicTally_tester{
  using namespace MonteRay;

SUITE( gpuTally_tester ) {
  TEST(Constructor){
    BasicTally tally(10);
    CHECK_EQUAL(10,  tally.size());
    auto tallies = tally.getTallies();
    for (const auto& val : tallies){
      CHECK_EQUAL(0, val);
    }
  }

  TEST(Score){
    BasicTally tally(10);
    tally.scoreByIndex(4, 1.2);
    CHECK_EQUAL(1.2, tally.getTally(4));
  }

  TEST(Clear){
    BasicTally tally(10);
    tally.scoreByIndex(4, 1.2);
    tally.clear();
    for (const auto& val : tally.getTallies()){
      CHECK_EQUAL(0, val);
    }
  }

  TEST(ReadAndWrite){
    BasicTally tally(10);
    tally.scoreByIndex(4, 1.2);

    std::stringstream stream;
    tally.write(stream);
    auto otherTally = BasicTally::read(stream);
    int i = 0;
    for (const auto& val : otherTally.getTallies()){
      if (i == 4){
        CHECK_EQUAL(1.2, val);
      } else {
        CHECK_EQUAL(0, val);
      }
      i++;
    }
  }

#ifdef __CUDACC__
  __global__ void testBasicTallyOnGPU(bool* testVal, BasicTally* const pTally){
    *testVal = true;

    *testVal = *testVal && (pTally->size() == 10);

    pTally->scoreByIndex(3, 1.4);
    *testVal = *testVal && (1.4 == pTally->getTally(3));

  }

  TEST(BasicTallyOnGPU){
    std::unique_ptr<BasicTally> pTally = std::make_unique<BasicTally>(10);

    bool* testVal;
    cudaMallocManaged(&testVal, sizeof(bool));
    testBasicTallyOnGPU<<<1,1>>>(testVal, pTally.get());
    cudaDeviceSynchronize();
    CHECK(*testVal);
  }
#endif

}

} // end namespace
