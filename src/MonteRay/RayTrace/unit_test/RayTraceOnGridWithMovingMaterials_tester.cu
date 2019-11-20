#include <UnitTest++.h>

#include "MonteRay_CartesianGrid.t.hh"
#include "MonteRay_SpatialGrid.hh"
#include "MonteRayVector3D.hh"
#include "MonteRay_GridBins.hh"
#include "MonteRayDefinitions.hh"
#include "MonteRayCopyMemory.t.hh"
#include "MaterialProperties.hh"
#include "ExpectedPathLength.t.hh"
#include "Ray.hh"

namespace MonteRay_CartesianGrid_rayTraceWithMovingMaterials_tests{

using namespace MonteRay;

struct MockMaterial{
  constexpr gpuRayFloat_t getTotalXS(gpuRayFloat_t) const {return 1.0; }
};
struct MockMaterialList {
  constexpr auto material(int) const {return MockMaterial{};}
};

struct MockVoidMaterial{
  constexpr gpuRayFloat_t getTotalXS(gpuRayFloat_t) const {return 0.0; }
};
struct MockVoidMaterialList {
  constexpr auto material(int) const {return MockVoidMaterial{};}
};

SUITE( MonteRay_CartesianGrid_rayTraceWithMovingMaterials_Tests) {

  using Position_t = Vector3D<gpuRayFloat_t>;
  using Direction_t = Vector3D<gpuRayFloat_t>;
  using GridBins_t = MonteRay_GridBins;

  enum coord {X,Y,Z,DIM};

  class GridTestData {
  public:

    GridTestData(){
      std::vector<gpuRayFloat_t> vertices{ -1, 0, 1 };

      pGridInfo[X] = new GridBins_t();
      pGridInfo[Y] = new GridBins_t();
      pGridInfo[Z] = new GridBins_t();

      pGridInfo[X]->initialize( vertices );
      pGridInfo[Y]->initialize( vertices );
      pGridInfo[Z]->initialize( vertices );

      }
      ~GridTestData(){
          delete pGridInfo[X];
          delete pGridInfo[Y];
          delete pGridInfo[Z];
      }

      MonteRay_SpatialGrid::pArrayOfpGridInfo_t pGridInfo;
  };

  class CartesianGrid{
    public:
      GridTestData gridTestData;
      MonteRay_CartesianGrid cart;
      CartesianGrid(): gridTestData(GridTestData{}), cart(3, gridTestData.pGridInfo){}
  };

  TEST_FIXTURE( CartesianGrid, RayTraceWithNonMovingMaterials){
    // set material properties
    std::unique_ptr<MaterialProperties> pMatProps;
    auto mpb = MaterialProperties::Builder();
    using Cell = MaterialProperties::Builder::Cell;
    Cell cell{ {1}, {1.0} }; // set material IDs and densities
    int size = 1;
    for (int i = 0; i < 3; i++){
      size *= cart.getNumBins(i);
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

    gpuFloatType_t timeRemaining = 10.0E6;
    std::vector<gpuTallyType_t> tallyData(8, 0);

    ray.position() = {-0.5, -0.5, -0.5};
    gpuFloatType_t inv_sqrt_2 = Math::sqrt(2.0)/2.0;
    ray.direction() = {0.0, inv_sqrt_2, inv_sqrt_2};
    MockVoidMaterialList voidMatList{};
    rayTraceOnGridWithMovingMaterials(ray, timeRemaining, cart, *pMatProps, voidMatList, tallyData.data());
    CHECK_CLOSE(Math::sqrt(2.0)/2.0, tallyData[0], 1e-6);
    CHECK_CLOSE(Math::sqrt(2.0), tallyData[6], 1e-6);

    ray.position() = {-0.5, -0.5, -0.5};
    ray.direction() = {1.0, 0.0, 0.0};
    tallyData = {0.0};
    MockMaterialList matList{};
    rayTraceOnGridWithMovingMaterials(ray, timeRemaining, cart, *pMatProps, matList, tallyData.data());
    auto score0 = 1.0 - Math::exp(-0.5);
    CHECK_CLOSE(score0, tallyData[0], 1e-6);
    auto score1 = exp(-0.5)*(1.0 - Math::exp(-1.0));
    CHECK_CLOSE(score1, tallyData[1], 1e-6);
  }

  TEST_FIXTURE( CartesianGrid, RayTraceWithMovingMaterials){
    auto mpb = MaterialProperties::Builder();
    using Cell = MaterialProperties::Builder::Cell;

    Cell cell{ {1}, {1.0} }; // set material IDs and densities
    int size = 1;
    for (int i = 0; i < 3; i++){
      size *= cart.getNumBins(i);
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

    gpuFloatType_t timeRemaining = 10.0E6;
    std::vector<gpuTallyType_t> tallyData(8, 0);

    ray.position() = { -0.000001, -0.5,  -0.5 };
    ray.direction() = { 1.0, 0.0, 0.0 };
    MockVoidMaterialList voidMatList{};
    rayTraceOnGridWithMovingMaterials(ray, timeRemaining, cart, *pMatProps, voidMatList, tallyData.data());

    CHECK_CLOSE( 0.5*std::sqrt(2.0), tallyData[0], 1e-5 );
    CHECK_CLOSE( 0.5*std::sqrt(2.0), tallyData[2], 1e-5 );
    CHECK_CLOSE( 0.5*std::sqrt(2.0), tallyData[3], 1e-5 );
    CHECK_CLOSE( 0.5*std::sqrt(2.0), tallyData[7], 1e-5 );
    CHECK_CLOSE( 0.5*std::sqrt(2.0), tallyData[6], 1e-5 );
    CHECK_CLOSE( 0.5*std::sqrt(2.0), tallyData[4], 1e-5 );
    CHECK_CLOSE( 0.5*std::sqrt(2.0), tallyData[5], 1e-5 );
    CHECK_CLOSE( 0.5*std::sqrt(2.0), tallyData[1], 1e-5 );
  }
}

} // end namespace

