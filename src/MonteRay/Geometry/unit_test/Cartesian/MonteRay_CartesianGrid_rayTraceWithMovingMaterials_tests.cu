#include <UnitTest++.h>

#include "MonteRay_CartesianGrid.t.hh"
#include "MonteRay_SpatialGrid.hh"
#include "MonteRayVector3D.hh"
#include "MonteRay_GridBins.hh"
#include "MonteRayDefinitions.hh"
#include "MonteRayCopyMemory.t.hh"
#include "RayWorkInfo.hh"
#include "MaterialProperties.hh"

namespace MonteRay_CartesianGrid_rayTraceWithMovingMaterials_tests{

using namespace MonteRay;

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

  TEST_FIXTURE( CartesianGrid, ConvertToCellReferenceFrame ) {
    Direction_t dir(1.0,   0.0,    0.0);
    gpuRayFloat_t speed = 4.0;
    Direction_t velocity{3.0, -2.0, 3.0};
    auto newDirAndSpeed = cart.convertToCellReferenceFrame(velocity, dir, speed);
    auto newSpeed = std::sqrt(1.0 + 2.0*2.0 + 3.0*3.0);
    CHECK_CLOSE(newSpeed, newDirAndSpeed.speed(), 1e-6);
    CHECK_CLOSE(1.0/newSpeed, newDirAndSpeed.direction()[0], 1e-6);
    CHECK_CLOSE(-3.0/newSpeed, newDirAndSpeed.direction()[2], 1e-6);
  }

  TEST_FIXTURE( CartesianGrid, CalcIndices ){
    Position_t pos( -0.5, -1.5,  0.5 );
    auto indices = cart.calcIndices(pos);
    CHECK_EQUAL(0, indices[0]);
    CHECK_EQUAL(-1, indices[1]);
    CHECK_EQUAL(1, indices[2]);
  }

  TEST_FIXTURE( CartesianGrid, GetMinDistToSurface){
    Position_t pos( -0.5, 0.2,  0.3 );
    Position_t dir(1, 0, 0);
    auto indices = cart.calcIndices(pos);
    auto distAndDim = cart.getMinDistToSurface(pos, dir, indices.data());
    CHECK_CLOSE(0.5, distAndDim.distance(), 1e-6);
    CHECK_EQUAL(0, distAndDim.dimension());

    dir = {0, -1, 0};
    distAndDim = cart.getMinDistToSurface(pos, dir, indices.data());
    CHECK_CLOSE(0.2, distAndDim.distance(), 1e-6);
    CHECK_EQUAL(1, distAndDim.dimension());

    dir = {0, 0, 1};
    distAndDim = cart.getMinDistToSurface(pos, dir, indices.data());
    CHECK_CLOSE(0.7, distAndDim.distance(), 1e-6);
    CHECK_EQUAL(2, distAndDim.dimension());
  }

  TEST_FIXTURE( CartesianGrid, RayTraceWithNonMovingMaterials){
    gpuRayFloat_t dist = 1.0;

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

    RayWorkInfo rayInfo(1);
    const double speed = 1.0;
    constexpr int threadID = 0;
    Position_t position ( -0.5, -0.5,  -0.5 );
    Direction_t dir(    1,   0,    0 );
    cart.rayTraceWithMovingMaterials(threadID, rayInfo, position, dir, dist, speed, *pMatProps);

    rayTraceList_t distances( rayInfo, threadID );
    CHECK_EQUAL( 2, distances.size() );
    CHECK_EQUAL( 0, distances.id(0) );
    CHECK_CLOSE( 0.5, distances.dist(0), 1e-6 );
    CHECK_EQUAL( 1, distances.id(1) );
    CHECK_CLOSE( 0.5, distances.dist(1), 1e-6 );

    rayInfo = RayWorkInfo{1};
    position = { -100, -0.5,  -0.5 };
    dir = {1,   0,    0};
    cart.rayTraceWithMovingMaterials(threadID, rayInfo, position, dir, dist, speed, *pMatProps);

    distances = rayTraceList_t( rayInfo, threadID );
    CHECK_EQUAL( 0, distances.size() );

    rayInfo = RayWorkInfo{1};
    position = { -100, -0.5,  -0.5 };
    dir = {-1,   0,    0};
    cart.rayTraceWithMovingMaterials(threadID, rayInfo, position, dir, dist, speed, *pMatProps);

    distances = rayTraceList_t{ rayInfo, threadID};
    CHECK_EQUAL( 0, distances.size() );
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
    mpb.setCellVelocity(5, {1.0, 0.0, 1.0});
    mpb.setCellVelocity(1, {0.0, 0.0, 0.0});
    auto pMatProps = std::make_unique<MaterialProperties>(mpb.build());

    gpuRayFloat_t dist = 10.0;

    RayWorkInfo rayInfo(1);
    constexpr int threadID = 0;
    Position_t pos( -0.000001, -0.5,  -0.5 );
    gpuRayFloat_t speed = 1.0;
    Direction_t dir( 1.0, 0.0, 0.0);
    cart.rayTraceWithMovingMaterials(threadID, rayInfo, pos, dir, dist, speed, *pMatProps);

    rayTraceList_t distances( rayInfo, 0 );
    CHECK_EQUAL( 8, distances.size() );

    CHECK_EQUAL( 0, distances.id(0) );
    CHECK_EQUAL( 2, distances.id(1) );
    CHECK_EQUAL( 3, distances.id(2) );
    CHECK_EQUAL( 7, distances.id(3) );
    CHECK_EQUAL( 6, distances.id(4) );
    CHECK_EQUAL( 4, distances.id(5) );
    CHECK_EQUAL( 5, distances.id(6) );
    CHECK_EQUAL( 1, distances.id(7) );

    CHECK_CLOSE( 0.5*std::sqrt(2.0), distances.dist(0), 1e-5 );
    CHECK_CLOSE( 0.5*std::sqrt(2.0), distances.dist(1), 1e-5 );
    CHECK_CLOSE( 0.5*std::sqrt(2.0), distances.dist(2), 1e-5 );
    CHECK_CLOSE( 0.5*std::sqrt(2.0), distances.dist(3), 1e-5 );
    CHECK_CLOSE( 0.5*std::sqrt(2.0), distances.dist(4), 1e-5 );
    CHECK_CLOSE( 0.5*std::sqrt(2.0), distances.dist(5), 1e-5 );
    CHECK_CLOSE( 0.5, distances.dist(6), 1e-5 );
    CHECK_CLOSE( 1.0, distances.dist(7), 1e-5 );
  }
}

} // end namespace

