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
    Position_t pos(0.0,   0.0,    0.0);
    Direction_t dir(1.0,   0.0,    0.0);
    gpuRayFloat_t speed = 4.0;
    Direction_t velocity{3.0, -2.0, 3.0};
    auto newDirAndSpeed = cart.convertToCellReferenceFrame(velocity, pos, dir, speed);
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
}

} // end namespace

