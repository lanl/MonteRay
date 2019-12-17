#include <UnitTest++.h>

#include <memory>

#include "ManagedAllocator.hh"
#include "NextEventEstimator.t.hh"
#include "RayWorkInfo.hh"
#include "Ray.hh"
#include "RayList.hh"
#include "MonteRayParallelAssistant.hh"

namespace NextEventEsimator_punit_tests{
using Managed = MonteRay::Managed;

struct Geometry : public Managed {
  template <typename Position_t, typename float_t>
  constexpr void rayTrace( unsigned threadID, MonteRay::RayWorkInfo& rayInfo, const Position_t& pos,
      const Position_t& dir, float_t distance,  bool outsideDistances) const {
      rayInfo.addCrossingCell(0, threadID, 0, distance);
      rayInfo.addRayCastCell(threadID, 0, distance);
  }
  const auto getDevicePtr() const { return this; }
};

struct MaterialList : public Managed {
  struct Material{
    constexpr double getTotalXS(double, double) const {return 0.5;}
  };
  constexpr int numMaterials() const {return 1;}
  constexpr double getTotalXS(int, double, double) const {return 0.5;}
  constexpr auto material(int) const {return Material{}; }
};

struct MaterialProperties : public Managed {
  constexpr int numMats(int) const {return 1;}
  constexpr int getMaterialID(int, int) const {return 0;}
  constexpr double getDensity(int, int) const {return 2.0;}
};

class NEE_Fixture{
  public:
  int max_n_rays = 10;
  std::unique_ptr<MonteRay::NextEventEstimator> pNee;
  MonteRay::Ray_t<3> ray;
  std::unique_ptr<MonteRay::RayWorkInfo> pRayWorkInfo = std::make_unique<MonteRay::RayWorkInfo>(max_n_rays);

  std::unique_ptr<Geometry> pGeometry = std::make_unique<Geometry>();
  std::unique_ptr<MaterialProperties> pMatProps = std::make_unique<MaterialProperties>();
  std::unique_ptr<MaterialList> pMatList = std::make_unique<MaterialList>();

  const MonteRay::MonteRayParallelAssistant& PA;

  NEE_Fixture() :
      PA( MonteRay::MonteRayParallelAssistant::getInstance() )
  {
    MonteRay::NextEventEstimator::Builder nee_builder(1);
    nee_builder.addTallyPoint(1.0, 2.0, 3.0);
    nee_builder.addTallyPoint(2.0, 2.0, 3.0);
    nee_builder.setExclusionRadius(0.0);
    std::vector<double> timeEdges{1.0, 2.0};
    nee_builder.setTimeBinEdges( timeEdges );
    pNee = std::make_unique<MonteRay::NextEventEstimator>(nee_builder.build());

    ray.dir = {1.0, 0.0, 0.0};
    ray.pos = {0.0, 2.0, 3.0};
    ray.particleType = 1;
    ray.weight = {0.34, 0.33, 0.33};
    ray.energy = {1.0, 1.0, 1.0};
  }
};

SUITE( NextEventEstimator_pTester ) {

  void check_build(const MonteRay::NextEventEstimator& nee){
    CHECK_EQUAL(2, nee.size());
    CHECK_CLOSE(1.0, nee.getPoint(0)[0], 1e-6);
    CHECK_CLOSE(2.0, nee.getPoint(0)[1], 1e-6);
    CHECK_CLOSE(3.0, nee.getPoint(0)[2], 1e-6);
    CHECK_CLOSE(0.0, nee.getExclusionRadius(), 1e-6);
    const auto& testTimeEdges = nee.getTimeBinEdges();
    CHECK_CLOSE(1.0, testTimeEdges[0], 1e-6);
    CHECK_CLOSE(2.0, testTimeEdges[1], 1e-6);
  }

  template<typename ParallelAssistant_t>
  void printHostInfo(const ParallelAssistant_t& PA) {
    char hostname[1024];
    gethostname(hostname, 1024);
    std::cout << "MonteRayNextEventEstimator::launch_ScoreRayList_on_GPU -- hostname = " << hostname <<
        ", world_rank=" << PA.getWorldRank() <<
        ", world_size=" << PA.getWorldSize() <<
        ", shared_memory_rank=" << PA.getSharedMemoryRank() <<
        ", shared_memory_size=" << PA.getSharedMemorySize() <<
        "\n";
  }

  TEST_FIXTURE( NEE_Fixture, Builder ) {
    auto& nee = *pNee;
    check_build(nee);
  }

  TEST_FIXTURE(NEE_Fixture, calcScore){

    int threadID = 0;

    auto& geometry = *pGeometry;
    auto& matProps = *pMatProps;
    auto& matList = *pMatList;
    auto& rayWorkInfo = *pRayWorkInfo;

    auto score = pNee->calcScore(threadID, ray, rayWorkInfo, geometry, matProps, matList);

    pNee->gatherWorkGroup(); // used for testing only
    pNee->gather();

    if( PA.getWorldRank() == 0 ) {
    // check basic score in parallel
        double expected = 1.0/(2.0*M_PI)*std::exp(-1.0);
        CHECK_CLOSE( expected, score, 1E-6);
        CHECK_CLOSE( expected*PA.getWorldSize(), pNee->getTally(0, 0), 1E-6);
    }

  }

  TEST_FIXTURE(NEE_Fixture, cpuScoreRayList){
    auto bank = MonteRay::RayList_t<3>(max_n_rays);
    for(int i =0; i < max_n_rays; i++){
      bank.add(ray);
    }
    cpuScoreRayList(pNee.get(), &bank, pRayWorkInfo.get(), pGeometry.get(), pMatProps.get(), pMatList.get());

    pNee->gatherWorkGroup(); // used for testing only
    pNee->gather();

    if( PA.getWorldRank() == 0 ) {
        // check basic score in parallel
        double expected = 1.0/(2.0*M_PI)*std::exp(-1.0)*max_n_rays;
        CHECK_CLOSE( expected*PA.getWorldSize(), pNee->getTally(0, 0), 1E-6);
    }
  }

  TEST_FIXTURE(NEE_Fixture, launch_ScoreRayList_on_GPU){
    // With 1 GPU on a single node this looks like a serial test.
    // With 2 GPUs per node or 2 nodes it will test multi-GPU parallel behavior

    const bool debug = false;

    // print hostname and ranks to make sure multiple node, multiple GPU test is correct.
    if( debug ) {
      printHostInfo(PA);
    }

    auto pBank = std::make_unique<MonteRay::RayList_t<3>>(max_n_rays);

    if( PA.getWorkGroupRank() == 0 ) {
        for(int i =0; i < max_n_rays; i++){
            pBank->add(ray);
        }
    }

#ifdef __CUDACC__
    pBank->copyToGPU();
    auto stream = std::make_unique<cudaStream_t>();
    *stream = 0; // default stream

    if( PA.getWorkGroupRank() == 0 ) {
        launch_ScoreRayList(pNee.get(), 1, 1, pBank.get(), pRayWorkInfo.get(), pGeometry.get(), pMatProps.get(), pMatList.get(), stream.get() );
        cudaDeviceSynchronize();
    }

    pNee->gatherWorkGroup(); // used for testing only
    pNee->gather();

    if( PA.getWorldRank() == 0 ) {
        // check basic score in parallel
        double expected = 1.0/(2.0*M_PI)*std::exp(-1.0)*max_n_rays;
        CHECK_CLOSE( expected * PA.getInterWorkGroupSize(), pNee->getTally(0, 0), 1E-6);
    } else {
        CHECK_CLOSE( 0.0, pNee->getTally(0, 0), 1E-6);
    }
#endif
  }

}

} // end namespace

