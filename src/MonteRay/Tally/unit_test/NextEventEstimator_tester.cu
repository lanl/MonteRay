#include <UnitTest++.h>

#include <memory>

#include "ManagedAllocator.hh"
#include "NextEventEstimator.t.hh"
#include "RayWorkInfo.hh"
#include "Ray.hh"
#include "RayList.hh"

namespace NextEventEsimator_unittest{
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

  NEE_Fixture() {
    MonteRay::NextEventEstimator::Builder nee_builder(1);
    nee_builder.addTallyPoint(1.0, 2.0, 3.0);
    nee_builder.addTallyPoint(2.0, 2.0, 3.0, 10.0);
    std::vector<double> timeEdges{1.0, 2.0};
    nee_builder.timeBinEdges( std::move(timeEdges) );
    std::vector<double> energyEdges{5E-6, 2.0, 10.0};
    nee_builder.energyBinEdges( std::move(energyEdges) );
    pNee = std::make_unique<MonteRay::NextEventEstimator>(nee_builder.build());

    ray.dir = {1.0, 0.0, 0.0};
    ray.pos = {0.0, 2.0, 3.0}; 
    ray.particleType = 1;
    ray.weight = {0.34, 0.33, 0.33};
    ray.energy = {1.0, 1.0, 1.0};
    ray.time = 0.0;
  }
};

SUITE( NextEventEstimator_Tester ) {

  void check_build(const MonteRay::NextEventEstimator& nee){
    CHECK_EQUAL(2, nee.size());
    CHECK_CLOSE(1.0, nee.getPoint(0)[0], 1e-6);
    CHECK_CLOSE(2.0, nee.getPoint(0)[1], 1e-6);
    CHECK_CLOSE(3.0, nee.getPoint(0)[2], 1e-6);
    CHECK_CLOSE(0.0, nee.getExclusionRadius(0), 1e-6);
    CHECK_CLOSE(10.0, nee.getExclusionRadius(1), 1e-6);
    const auto& timeEdges = nee.timeBinEdges();
    CHECK_CLOSE(1.0, timeEdges[0], 1e-6);
    CHECK_CLOSE(2.0, timeEdges[1], 1e-6);
    const auto& energyEdges = nee.energyBinEdges();
    CHECK_CLOSE(5E-6, energyEdges[0], 1e-8);
    CHECK_CLOSE(2.0, energyEdges[1], 1e-6);
    CHECK_CLOSE(10.0, energyEdges[2], 1e-6);
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

    pNee->calcScore(threadID, ray, rayWorkInfo, geometry, matProps, matList);
    // test to ensure test ray tracer operates as expected
    CHECK_CLOSE(1.0, rayWorkInfo.getCrossingDist(0, 0, 0), 1E-6);

    // check basic score
    auto score = 1.0/(2.0*M_PI)*std::exp(-1.0);
    CHECK_CLOSE(score, pNee->contribution(0, 1, 0), 1E-6); 

    // check basic score in second time bin
    ray.time = 1.0;
    rayWorkInfo.clear();
    pNee->calcScore(threadID, ray, rayWorkInfo, geometry, matProps, matList);
    CHECK_CLOSE(score, pNee->contribution(0, 1, 1), 1E-6); 
    // make sure original contribution didn't change
    CHECK_CLOSE(score, pNee->contribution(0, 1, 0), 1E-6); 
    CHECK_CLOSE(0.0, pNee->contribution(0, 1, 2), 1E-6); 

    // check basic score in overflow time bin
    ray.time = 100.0;
    rayWorkInfo.clear();
    pNee->calcScore(threadID, ray, rayWorkInfo, geometry, matProps, matList);
    CHECK_CLOSE(score, pNee->contribution(0, 1, 2), 1E-6); 
    // make sure other scores didn't change
    CHECK_CLOSE(score, pNee->contribution(0, 1, 2), 1E-6); 
    CHECK_CLOSE(score, pNee->contribution(0, 1, 0), 1E-6); 
    CHECK_CLOSE(score, pNee->contribution(0, 1, 1), 1E-6); 
    
    // NOTE: following tests make sure score didn't change (aka 0 contribution)
    // check rejection of neutrons
    rayWorkInfo.clear();
    ray.particleType = 0;
    pNee->calcScore(threadID, ray, rayWorkInfo, geometry, matProps, matList);
    CHECK_CLOSE(score, pNee->contribution(0, 1, 2), 1E-6); 
    CHECK_CLOSE(score, pNee->contribution(0, 1, 0), 1E-6); 
    CHECK_CLOSE(score, pNee->contribution(0, 1, 1), 1E-6); 

    // check rejection of zero weights
    ray.particleType = 1;
    ray.weight = {0.0, 0.0, 0.0};
    pNee->calcScore(threadID, ray, rayWorkInfo, geometry, matProps, matList);
    CHECK_CLOSE(score, pNee->contribution(0, 1, 2), 1E-6); 
    CHECK_CLOSE(score, pNee->contribution(0, 1, 0), 1E-6); 
    CHECK_CLOSE(score, pNee->contribution(0, 1, 1), 1E-6); 

    // check rejection of for zero energies
    ray.weight = {1.0, 1.0, 1.0};
    ray.energy = {0.0, 0.0, 0};
    pNee->calcScore(threadID, ray, rayWorkInfo, geometry, matProps, matList);
    CHECK_CLOSE(score, pNee->contribution(0, 1, 2), 1E-6); 
    CHECK_CLOSE(score, pNee->contribution(0, 1, 0), 1E-6); 
    CHECK_CLOSE(score, pNee->contribution(0, 1, 1), 1E-6); 
  }

  TEST_FIXTURE(NEE_Fixture, cpuScoreRayList){
    auto bank = MonteRay::RayList_t<3>(max_n_rays);
    for(int i =0; i < max_n_rays; i++){
      bank.add(ray);
    }
    cpuScoreRayList(pNee.get(), &bank, pRayWorkInfo.get(), pGeometry.get(), pMatProps.get(), pMatList.get());
  }

  TEST_FIXTURE(NEE_Fixture, launch_ScoreRayList){
    auto pBank = std::make_unique<MonteRay::RayList_t<3>>(max_n_rays);
    for(int i =0; i < max_n_rays; i++){
      pBank->add(ray);
    }
#ifdef __CUDACC__
    launch_ScoreRayList(pNee.get(), 1, 1, pBank.get(), pRayWorkInfo.get(), pGeometry.get(), pMatProps.get(), pMatList.get());
    cudaDeviceSynchronize();
    CHECK_CLOSE(max_n_rays*1.0/(2.0*M_PI)*std::exp(-1.0), pNee->contribution(0, 1, 0), 1E-6); 
#endif
  }

  TEST_FIXTURE(NEE_Fixture, Read_write_NEE){
    std::stringstream file;
    pNee->write(file);
    auto nee_builder = MonteRay::NextEventEstimator::Builder();
    auto nee = nee_builder.read(file);
    check_build(nee);
  }

}

} // end namespace

