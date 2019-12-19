#include <UnitTest++.h>

#include "MonteRay_CylindricalGrid.t.hh"
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
          gpuTallyType_t* const tallyData) {
    rayTraceOnGridWithMovingMaterials(ray, timeRemaining, *geometry, *pMatProps, matList, tallyData);
  }
#endif

SUITE( MonteRay_CartesianGrid_rayTraceWithMovingMaterials_Tests ) {

  using Position_t = Vector3D<gpuRayFloat_t>;
  using Direction_t = Vector3D<gpuRayFloat_t>;
  using GridBins_t = MonteRay_GridBins;

  enum coord {X,Y,Z,DIM};

  class CartesianGrid{
    public:
      std::unique_ptr<MonteRay_CartesianGrid> pCart;
      CartesianGrid() {
        pCart = std::make_unique<MonteRay_CartesianGrid>(3, 
            std::array<MonteRay_GridBins, 3> {
                  MonteRay_GridBins{-1, 1, 2},
                  MonteRay_GridBins{-1, 1, 2},
                  MonteRay_GridBins{-1, 1, 2} 
            }
          );
        }
  };

  TEST_FIXTURE( CartesianGrid, RayTraceWithNonMovingMaterials){
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

    gpuFloatType_t timeRemaining = 10.0E6;
    SimpleVector<gpuTallyType_t> tallyData(8, 0.0);

    ray.position() = {-0.5, -0.5, -0.5};
    gpuFloatType_t inv_sqrt_2 = Math::sqrt(2.0)/2.0;
    ray.direction() = {0.0, inv_sqrt_2, inv_sqrt_2};
    MockVoidMaterialList voidMatList{};
    rayTraceOnGridWithMovingMaterials(ray, timeRemaining, *pCart, *pMatProps, voidMatList, tallyData.data());
    CHECK_CLOSE(Math::sqrt(2.0)/2.0, tallyData[0], 1e-6);
    CHECK_CLOSE(Math::sqrt(2.0), tallyData[6], 1e-6);

    ray.position() = {-0.5, -0.5, -0.5};
    ray.direction() = {1.0, 0.0, 0.0};
    tallyData = {0.0};
    MockMaterialList matList{};
    rayTraceOnGridWithMovingMaterials(ray, timeRemaining, *pCart, *pMatProps, matList, tallyData.data());
    auto score0 = 1.0/2.0*(1.0 - Math::exp(-1.0));
    CHECK_CLOSE(score0, tallyData[0], 1e-6);
    auto score1 = Math::exp(-1.0)*(1.0/2.0)*(1.0 - Math::exp(-2.0));
    CHECK_CLOSE(score1, tallyData[1], 1e-6);

#ifdef __CUDACC__
    {
      SimpleVector<gpuTallyType_t> gpuTallyData(8, 0.0);
      ray.position() = {-0.5, -0.5, -0.5};
      ray.direction() = {0.0, inv_sqrt_2, inv_sqrt_2};
      kernelRayTraceOnGridWithMovingMaterials<<<1, 1>>>(ray, timeRemaining, pCart.get(), pMatProps.get(), voidMatList, gpuTallyData.data());
      cudaDeviceSynchronize();
      CHECK_CLOSE(Math::sqrt(2.0)/2.0, gpuTallyData[0], 1e-6);
      CHECK_CLOSE(Math::sqrt(2.0), gpuTallyData[6], 1e-6);

      ray.position() = {-0.5, -0.5, -0.5};
      ray.direction() = {1.0, 0.0, 0.0};
      gpuTallyData = {0.0};
      kernelRayTraceOnGridWithMovingMaterials<<<1, 1>>>(ray, timeRemaining, pCart.get(), pMatProps.get(), matList, gpuTallyData.data());
      cudaDeviceSynchronize();
      CHECK_CLOSE(score0, gpuTallyData[0], 1e-6);
      CHECK_CLOSE(score1, gpuTallyData[1], 1e-6);
    }

#endif
  }

  TEST_FIXTURE( CartesianGrid, RayTraceWithMovingMaterials){
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

    gpuFloatType_t timeRemaining = 10.0E6;
    SimpleVector<gpuTallyType_t> tallyData(8, 0.0);

    ray.position() = { -0.000001, -0.5,  -0.5 };
    ray.direction() = { 1.0, 0.0, 0.0 };
    MockVoidMaterialList voidMatList{};
    rayTraceOnGridWithMovingMaterials(ray, timeRemaining, *pCart, *pMatProps, voidMatList, tallyData.data());

    CHECK_CLOSE( 0.5*std::sqrt(2.0), tallyData[0], 1e-5 );
    CHECK_CLOSE( 0.5*std::sqrt(2.0), tallyData[2], 1e-5 );
    CHECK_CLOSE( 0.5*std::sqrt(2.0), tallyData[3], 1e-5 );
    CHECK_CLOSE( 0.5*std::sqrt(2.0), tallyData[7], 1e-5 );
    CHECK_CLOSE( 0.5*std::sqrt(2.0), tallyData[6], 1e-5 );
    CHECK_CLOSE( 0.5*std::sqrt(2.0), tallyData[4], 1e-5 );
    CHECK_CLOSE( 0.5*std::sqrt(2.0), tallyData[5], 1e-5 );
    CHECK_CLOSE( 0.5*std::sqrt(2.0), tallyData[1], 1e-5 );

#ifdef __CUDACC__
    {
      SimpleVector<gpuTallyType_t> gpuTallyData(8, 0.0);
      kernelRayTraceOnGridWithMovingMaterials<<<1, 1>>>(ray, timeRemaining, pCart.get(), pMatProps.get(), voidMatList, gpuTallyData.data());
      cudaDeviceSynchronize();
      CHECK_CLOSE( 0.5*std::sqrt(2.0), gpuTallyData[0], 1e-5 );
      CHECK_CLOSE( 0.5*std::sqrt(2.0), gpuTallyData[2], 1e-5 );
      CHECK_CLOSE( 0.5*std::sqrt(2.0), gpuTallyData[3], 1e-5 );
      CHECK_CLOSE( 0.5*std::sqrt(2.0), gpuTallyData[7], 1e-5 );
      CHECK_CLOSE( 0.5*std::sqrt(2.0), gpuTallyData[6], 1e-5 );
      CHECK_CLOSE( 0.5*std::sqrt(2.0), gpuTallyData[4], 1e-5 );
      CHECK_CLOSE( 0.5*std::sqrt(2.0), gpuTallyData[5], 1e-5 );
      CHECK_CLOSE( 0.5*std::sqrt(2.0), gpuTallyData[1], 1e-5 );
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

  class CylindricalGrid{
    public:
      gpuRayFloat_t two = 2.0;
      gpuRayFloat_t one = 1.0;
      std::unique_ptr<MonteRay_CylindricalGrid> pCyl;
      CylindricalGrid(){
        pCyl = std::make_unique<MonteRay_CylindricalGrid>(2, 
            std::array<MonteRay_GridBins, 3>{
                  MonteRay_GridBins{0.0, 2.0, 2, MonteRay_GridBins::RADIAL},
                  MonteRay_GridBins{-1.0, 1.0, 2},
                  MonteRay_GridBins{-1, 0, 1} 
             }
        );
        int size = pCyl->getNumBins(0)*pCyl->getNumBins(1);
        CHECK_EQUAL(4, size);
      }
  };

  TEST_FIXTURE( CylindricalGrid, RayTraceWithMovingMaterials){

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

    SimpleVector<gpuTallyType_t> tallyData(4, 0.0);

    Ray_t<1> ray;
    const double speed = 1.0/sqrt(4.0/5.0);
    auto energy = speed*speed*inv_neutron_speed_from_energy_const()*inv_neutron_speed_from_energy_const();
    ray.setEnergy(energy);

    gpuFloatType_t timeRemaining = 10.0E6;

    gpuFloatType_t val = -2.5/Math::sqrt(2.0);
    ray.position() = Position_t{ val, val, -1.0 };
    ray.direction() = Direction_t{Math::sqrt(two), Math::sqrt(two), one}.normalize();

    MockVoidMaterialList voidMatList{};
    rayTraceOnGridWithMovingMaterials(ray, timeRemaining, *pCyl, *pMatProps, voidMatList, tallyData.data());

    CHECK_CLOSE( 1.0, tallyData[1], 1e-6 );
    CHECK_CLOSE( 0.75*sqrt(2.0), tallyData[0], 1e-6 );
    CHECK_CLOSE( 0.75*sqrt(2.0), tallyData[2], 1e-6 );
    CHECK_CLOSE( 1.0, tallyData[3], 1e-6 );

#ifdef __CUDACC__
    {
      SimpleVector<gpuTallyType_t> gpuTallyData(4, 0.0);
      kernelRayTraceOnGridWithMovingMaterials<<<1, 1>>>(ray, timeRemaining, pCyl.get(), pMatProps.get(), voidMatList, gpuTallyData.data());
      cudaDeviceSynchronize();
      CHECK_CLOSE( 1.0, gpuTallyData[1], 1e-6 );
      CHECK_CLOSE( 0.75*sqrt(2.0), gpuTallyData[0], 1e-6 );
      CHECK_CLOSE( 0.75*sqrt(2.0), gpuTallyData[2], 1e-6 );
      CHECK_CLOSE( 1.0, gpuTallyData[3], 1e-6 );
    }
#endif


  }

  TEST_FIXTURE( CylindricalGrid, RayTraceCylWithoutMovingMaterials){

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

    SimpleVector<gpuTallyType_t> tallyData(4, 0.0);

    Ray_t<1> ray;
    const double speed = 1.0/sqrt(4.0/5.0);
    auto energy = speed*speed*inv_neutron_speed_from_energy_const()*inv_neutron_speed_from_energy_const();
    ray.setEnergy(energy);

    gpuFloatType_t timeRemaining = 10.0E6;

    ray.position() = Position_t{ -1.5, 0, -0.5 };
    ray.direction() = Direction_t{1.0, 0.0, 0.0};

    MockVoidMaterialList voidMatList{};
    rayTraceOnGridWithMovingMaterials(ray, timeRemaining, *pCyl, *pMatProps, voidMatList, tallyData.data());

    CHECK_CLOSE( 1.5, tallyData[1], 1e-6 );
    CHECK_CLOSE( 2, tallyData[0], 1e-6 );
    CHECK_CLOSE( 0, tallyData[2], 1e-6 );
    CHECK_CLOSE( 0, tallyData[3], 1e-6 );

  }
}

} // end namespace

