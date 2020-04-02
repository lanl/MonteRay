#include <UnitTest++.h>

#include "MonteRay_CylindricalGrid.t.hh"
#include "MonteRay_CartesianGrid.t.hh"
#include "MonteRay_SpatialGrid.hh"
#include "MonteRayVector3D.hh"
#include "MonteRay_GridBins.hh"
#include "MonteRayDefinitions.hh"
#include "MaterialProperties.hh"
#include "ExpectedPathLength.t.hh"
#include "Ray.hh"

namespace MonteRay_CartesianGrid_rayTraceWithMovingMaterials_tests{


using namespace MonteRay;

struct MockMaterial{
  constexpr gpuRayFloat_t getTotalXS(gpuRayFloat_t, gpuRayFloat_t) const {return 2.0; }
};
struct MockMaterialList {
  constexpr auto material(int) const {return MockMaterial{};}
};
struct MockVoidMaterial{
  constexpr gpuRayFloat_t getTotalXS(gpuRayFloat_t, gpuRayFloat_t) const {return 0.0; }
};
struct MockVoidMaterialList {
  constexpr auto material(int) const {return MockVoidMaterial{};}
};

#ifdef __CUDACC__
  template<unsigned N, typename Geometry, typename MaterialList>
  __global__ void kernelRayTraceOnGridWithMovingMaterials(
          Ray_t<N> ray,
          gpuRayFloat_t timeRemaining,
          const Geometry* geometry,
          const MaterialProperties* pMatProps,
          const MaterialList matList,
          ExpectedPathLengthTally* const tally) {
    tally->rayTraceOnGridWithMovingMaterials(ray, timeRemaining, *geometry, *pMatProps, matList);
  }
#endif

SUITE( MonteRay_CartesianGrid_rayTraceWithMovingMaterials_Tests ) {

  using Position_t = Vector3D<gpuRayFloat_t>;
  using Direction_t = Vector3D<gpuRayFloat_t>;
  using GridBins_t = MonteRay_GridBins;

  enum coord {X,Y,Z,DIM};

  class CartesianTallyFixture{
    public:
    std::unique_ptr<ExpectedPathLengthTally> pTally;
    std::unique_ptr<MonteRay_CartesianGrid> pCart;
    CartesianTallyFixture() {
      pCart = std::make_unique<MonteRay_CartesianGrid>(3, 
        std::array<MonteRay_GridBins, 3> {
              MonteRay_GridBins{-1, 1, 2},
              MonteRay_GridBins{-1, 1, 2},
              MonteRay_GridBins{-1, 1, 2} 
        }
      );
      ExpectedPathLengthTally::Builder builder;
      builder.spatialBins(pCart->size());
      std::vector<double> energyBinEdges = {1.0, 10.0};
      builder.energyBinEdges(energyBinEdges);
      std::vector<double> timeBinEdges = {1.0, 10.0};
      builder.timeBinEdges(timeBinEdges);
      pTally = std::make_unique<ExpectedPathLengthTally>(builder.build());
    }
  };

  TEST_FIXTURE( CartesianTallyFixture, RayTraceWithNonMovingMaterials){
    // set material properties
    std::unique_ptr<MaterialProperties> pMatProps;
    auto mpb = MaterialProperties::Builder();
    using Cell = MaterialProperties::Builder::Cell;
    Cell cell{ {0}, {1.0} }; // set material IDs and densities
    int size = 1;
    for (int i = 0; i < 3; i++){
      size *= pCart->getNumBins(i);
    }
    for (int i = 0; i < size; i++){
      mpb.addCell(cell);
      mpb.setCellVelocity(i, {0.0, 0.0, 0.0});
    }
    pMatProps = std::make_unique<MaterialProperties>(mpb.build());

    Ray_t<1> ray;
    gpuFloatType_t speed = 1.0;
    auto energy = speed*speed*inv_neutron_speed_from_energy_const()*inv_neutron_speed_from_energy_const();
    ray.setEnergy(energy);
    ray.setTime(0.0);

    gpuFloatType_t timeRemaining = 10.0E6;

    ray.position() = {-0.5, -0.5, -0.5};
    gpuFloatType_t inv_sqrt_2 = Math::sqrt(2.0)/2.0;
    ray.direction() = {0.0, inv_sqrt_2, inv_sqrt_2};
    MockVoidMaterialList voidMatList{};
    pTally->rayTraceOnGridWithMovingMaterials(ray, timeRemaining, *pCart, *pMatProps, voidMatList);
    CHECK_CLOSE(Math::sqrt(2.0)/2.0, pTally->contribution(0), 1e-6);
    CHECK_CLOSE(Math::sqrt(2.0), pTally->contribution(6), 1e-6);

    ray.position() = {-0.5, -0.5, -0.5};
    ray.direction() = {1.0, 0.0, 0.0};
    pTally->clear();
    MockMaterialList matList{};
    pTally->rayTraceOnGridWithMovingMaterials(ray, timeRemaining, *pCart, *pMatProps, matList);
    auto score0 = 1.0/2.0*(1.0 - Math::exp(-1.0));
    CHECK_CLOSE(score0, pTally->contribution(0), 1e-6);
    auto score1 = Math::exp(-1.0)*(1.0/2.0)*(1.0 - Math::exp(-2.0));
    CHECK_CLOSE(score1, pTally->contribution(1), 1e-6);

#ifdef __CUDACC__
    {
      ray.position() = {-0.5, -0.5, -0.5};
      ray.direction() = {0.0, inv_sqrt_2, inv_sqrt_2};
      pTally->clear();
      kernelRayTraceOnGridWithMovingMaterials<<<1, 1>>>(ray, timeRemaining, pCart.get(), pMatProps.get(), voidMatList, pTally.get());
      cudaDeviceSynchronize();
      CHECK_CLOSE(Math::sqrt(2.0)/2.0, pTally->contribution(0), 1e-6);
      CHECK_CLOSE(Math::sqrt(2.0), pTally->contribution(6), 1e-6);

      ray.position() = {-0.5, -0.5, -0.5};
      ray.direction() = {1.0, 0.0, 0.0};
      pTally->clear();
      kernelRayTraceOnGridWithMovingMaterials<<<1, 1>>>(ray, timeRemaining, pCart.get(), pMatProps.get(), matList, pTally.get());
      cudaDeviceSynchronize();
      CHECK_CLOSE(score0, pTally->contribution(0), 1e-6);
      CHECK_CLOSE(score1, pTally->contribution(1), 1e-6);
    }

#endif
  }

  TEST_FIXTURE( CartesianTallyFixture, RayTraceWithMovingMaterials){
    auto mpb = MaterialProperties::Builder();
    using Cell = MaterialProperties::Builder::Cell;

    Cell cell{ {0}, {1.0} }; // set material IDs and densities
    int size = 1;
    for (int i = 0; i < 3; i++){
      size *= pCart->getNumBins(i);
    }
    for (int i = 0; i < size; i++){
      mpb.addCell(cell);
    }
    // set artificial velocities to take ray on a tour of the "world"
    mpb.setCellVelocity(0, {2.0, -1.0, 0.0});
    mpb.setCellVelocity(2, {0.0, -1.0, 0.0});
    mpb.setCellVelocity(3, {0.0, 0.0, -1.0});
    mpb.setCellVelocity(7, {2.0, 0.0, -1.0});
    mpb.setCellVelocity(6, {2.0, 1.0, 0.0});
    mpb.setCellVelocity(4, {0.0, 1.0, 0.0});
    mpb.setCellVelocity(5, {0.0, 0.0, 1.0});
    mpb.setCellVelocity(1, {0.0, 0.0, 1.0});
    auto pMatProps = std::make_unique<MaterialProperties>(mpb.build());

    Ray_t<1> ray;
    gpuFloatType_t speed = 1.0;
    auto energy = speed*speed*inv_neutron_speed_from_energy_const()*inv_neutron_speed_from_energy_const();
    ray.setEnergy(energy);
    ray.setTime(0.0);

    gpuFloatType_t timeRemaining = 10.0E6;
    ray.position() = { -0.000001, -0.5,  -0.5 };
    ray.direction() = { 1.0, 0.0, 0.0 };
    MockVoidMaterialList voidMatList{};
    pTally->rayTraceOnGridWithMovingMaterials(ray, timeRemaining, *pCart, *pMatProps, voidMatList);

    CHECK_CLOSE( 0.5*std::sqrt(2.0), pTally->contribution(0), 1e-5 );
    CHECK_CLOSE( 0.5*std::sqrt(2.0), pTally->contribution(2), 1e-5 );
    CHECK_CLOSE( 0.5*std::sqrt(2.0), pTally->contribution(3), 1e-5 );
    CHECK_CLOSE( 0.5*std::sqrt(2.0), pTally->contribution(7), 1e-5 );
    CHECK_CLOSE( 0.5*std::sqrt(2.0), pTally->contribution(6), 1e-5 );
    CHECK_CLOSE( 0.5*std::sqrt(2.0), pTally->contribution(4), 1e-5 );
    CHECK_CLOSE( 0.5*std::sqrt(2.0), pTally->contribution(5), 1e-5 );
    CHECK_CLOSE( 0.5*std::sqrt(2.0), pTally->contribution(1), 1e-5 );

#ifdef __CUDACC__
    {
      pTally->clear();
      kernelRayTraceOnGridWithMovingMaterials<<<1, 1>>>(ray, timeRemaining, pCart.get(), pMatProps.get(), voidMatList, pTally.get());
      cudaDeviceSynchronize();
      CHECK_CLOSE( 0.5*std::sqrt(2.0), pTally->contribution(0), 1e-5 );
      CHECK_CLOSE( 0.5*std::sqrt(2.0), pTally->contribution(2), 1e-5 );
      CHECK_CLOSE( 0.5*std::sqrt(2.0), pTally->contribution(3), 1e-5 );
      CHECK_CLOSE( 0.5*std::sqrt(2.0), pTally->contribution(7), 1e-5 );
      CHECK_CLOSE( 0.5*std::sqrt(2.0), pTally->contribution(6), 1e-5 );
      CHECK_CLOSE( 0.5*std::sqrt(2.0), pTally->contribution(4), 1e-5 );
      CHECK_CLOSE( 0.5*std::sqrt(2.0), pTally->contribution(5), 1e-5 );
      CHECK_CLOSE( 0.5*std::sqrt(2.0), pTally->contribution(1), 1e-5 );
    }
#endif
  }
}

SUITE( MonteRay_CylindricalGrid_rayTraceWithMovingMaterials_Tests ) {

  using Position_t = Vector3D<gpuRayFloat_t>;
  using Direction_t = Vector3D<gpuRayFloat_t>;
  using GridBins_t = MonteRay_GridBins;

  enum cartesian_coord {x=0,y=1,z=2};
  enum cylindrical_coord {R=0,CZ=1,Theta=2,DIM=2};

  class CylindricalTallyFixture{
    public:
    gpuRayFloat_t two = 2.0;
    gpuRayFloat_t one = 1.0;

    std::unique_ptr<ExpectedPathLengthTally> pTally;
    std::unique_ptr<MonteRay_CylindricalGrid> pCyl;
    CylindricalTallyFixture() {
      pCyl = std::make_unique<MonteRay_CylindricalGrid>(2, 
        std::array<MonteRay_GridBins, 3>{
              MonteRay_GridBins{0.0, 2.0, 2, MonteRay_GridBins::RADIAL},
              MonteRay_GridBins{-1.0, 1.0, 2},
              MonteRay_GridBins{-1, 0, 1} 
         }
      );
      CHECK_EQUAL(4, pCyl->size());

      ExpectedPathLengthTally::Builder builder;
      builder.spatialBins(pCyl->size());
      std::vector<double> energyBinEdges = {1.0, 10.0};
      builder.energyBinEdges(energyBinEdges);
      std::vector<double> timeBinEdges = {1.0, 10.0};
      builder.timeBinEdges(timeBinEdges);
      pTally = std::make_unique<ExpectedPathLengthTally>(builder.build());
    }
  };

  TEST_FIXTURE( CylindricalTallyFixture, RayTraceWithMovingMaterials){

    std::unique_ptr<MaterialProperties> pMatProps;
    auto mpb = MaterialProperties::Builder();
    using Cell = MaterialProperties::Builder::Cell;
    Cell cell{ {0}, {1.0} }; // set material IDs and densities
    // velocities are (r, z, t), t is not used
    mpb.addCell(cell);
    mpb.setCellVelocity(0, {0.0, -0.5, 0.0});
    mpb.addCell(cell);
    mpb.setCellVelocity(1, {0.0,  0.5, 0.0});
    mpb.addCell(cell);
    mpb.setCellVelocity(2, {-2.0, -0.5, 0.0});
    mpb.addCell(cell);
    mpb.setCellVelocity(3, {-2.0, 0.5, 0.0});

    pMatProps = std::make_unique<MaterialProperties>(mpb.build());

    Ray_t<1> ray;
    const double speed = 1.0/sqrt(4.0/5.0);
    auto energy = speed*speed*inv_neutron_speed_from_energy_const()*inv_neutron_speed_from_energy_const();
    ray.setEnergy(energy);
    ray.setTime(0.0);

    gpuFloatType_t timeRemaining = 10.0E6;

    gpuFloatType_t val = -2.5/Math::sqrt(2.0);
    ray.position() = Position_t{ val, val, -1.0 };
    ray.direction() = Direction_t{Math::sqrt(two), Math::sqrt(two), one}.normalize();

    MockVoidMaterialList voidMatList{};
    pTally->rayTraceOnGridWithMovingMaterials(ray, timeRemaining, *pCyl, *pMatProps, voidMatList);

    CHECK_CLOSE( 1.0, pTally->contribution(1), 1e-6 );
    CHECK_CLOSE( 0.75*sqrt(2.0), pTally->contribution(0), 1e-6 );
    CHECK_CLOSE( 0.75*sqrt(2.0), pTally->contribution(2), 1e-6 );
    CHECK_CLOSE( 1.0, pTally->contribution(3), 1e-6 );

#ifdef __CUDACC__
    {
      pTally->clear();
      kernelRayTraceOnGridWithMovingMaterials<<<1, 1>>>(ray, timeRemaining, pCyl.get(), pMatProps.get(), voidMatList, pTally.get());
      cudaDeviceSynchronize();
      CHECK_CLOSE( 1.0, pTally->contribution(1), 1e-6 );
      CHECK_CLOSE( 0.75*sqrt(2.0), pTally->contribution(0), 1e-6 );
      CHECK_CLOSE( 0.75*sqrt(2.0), pTally->contribution(2), 1e-6 );
      CHECK_CLOSE( 1.0, pTally->contribution(3), 1e-6 );
    }
#endif


  }

  TEST_FIXTURE( CylindricalTallyFixture, RayTraceCylWithoutMovingMaterials){

    std::unique_ptr<MaterialProperties> pMatProps;
    auto mpb = MaterialProperties::Builder();
    using Cell = MaterialProperties::Builder::Cell;
    Cell cell{ {0}, {1.0} }; // set material IDs and densities
    // velocities are (r, z, t), t is not used
    for (int i = 0; i < 4; i++){
      mpb.addCell(cell);
      mpb.setCellVelocity(i, {0.0, 0.0, 0.0});
    }

    pMatProps = std::make_unique<MaterialProperties>(mpb.build());

    Ray_t<1> ray;
    const double speed = 1.0/sqrt(4.0/5.0);
    auto energy = speed*speed*inv_neutron_speed_from_energy_const()*inv_neutron_speed_from_energy_const();
    ray.setEnergy(energy);
    ray.setTime(0.0);

    gpuFloatType_t timeRemaining = 10.0E6;

    ray.position() = Position_t{ -1.5, 0, -0.5 };
    ray.direction() = Direction_t{1.0, 0.0, 0.0};

    MockVoidMaterialList voidMatList{};
    pTally->rayTraceOnGridWithMovingMaterials(ray, timeRemaining, *pCyl, *pMatProps, voidMatList);

    CHECK_CLOSE( 1.5, pTally->contribution(1), 1e-6 );
    CHECK_CLOSE( 2, pTally->contribution(0), 1e-6 );
    CHECK_CLOSE( 0, pTally->contribution(2), 1e-6 );
    CHECK_CLOSE( 0, pTally->contribution(3), 1e-6 );

  }
}

} // end namespace

