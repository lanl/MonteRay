#include <UnitTest++.h>

#include "MonteRay_CylindricalGrid.t.hh"
#include "MonteRay_SpatialGrid.hh"
#include "MonteRayVector3D.hh"
#include "MonteRay_GridBins.hh"
#include "MonteRayDefinitions.hh"
#include "MonteRayCopyMemory.t.hh"
#include "RayWorkInfo.hh"
#include "MaterialProperties.hh"

namespace MonteRay_CylindricalGrid_rayTraceWithMovingMaterials_tests{

using namespace MonteRay;

SUITE( MonteRay_CylindricalGrid_rayTraceWithMovingMaterials_Tests) {

  using Position_t = Vector3D<gpuRayFloat_t>;
  using Direction_t = Vector3D<gpuRayFloat_t>;
  using GridBins_t = MonteRay_GridBins;

  enum cartesian_coord {x=0,y=1,z=2};
  enum cylindrical_coord {R=0,CZ=1,Theta=2,DIM=2};

  class GridTestData {
  public:

    GridTestData(){
      std::vector<gpuRayFloat_t> Rverts = { 1.0, 2.0 };
      std::vector<gpuRayFloat_t> Zverts = { -1, 0, 1};

      pGridInfo[R] = new GridBins_t();
      pGridInfo[CZ] = new GridBins_t();

      pGridInfo[R]->initialize( Rverts );
      pGridInfo[CZ]->initialize( Zverts );

      }
      ~GridTestData(){
          delete pGridInfo[R];
          delete pGridInfo[CZ];
      }

      MonteRay_SpatialGrid::pArrayOfpGridInfo_t pGridInfo;
  };

  class CylindricalGrid{
    public:
      gpuRayFloat_t two = 2.0;
      gpuRayFloat_t one = 1.0;
      GridTestData gridTestData;
      MonteRay_CylindricalGrid cyl;
      CylindricalGrid(): gridTestData(GridTestData{}), cyl(2,gridTestData.pGridInfo){
        int size = cyl.getNumBins(0)*cyl.getNumBins(1);
        CHECK_EQUAL(4, size);
      }
  };

  TEST_FIXTURE( CylindricalGrid, ConvertToCellReferenceFrame ) {
    Position_t pos(0.5, 0.0, 0.5);
    Direction_t dir(1.0,   0.0,    0.0);
    gpuRayFloat_t speed = 1.0;
    Direction_t velocity(1.0, 2.0, 3.0);
    auto newDirAndSpeed = cyl.convertToCellReferenceFrame(velocity, pos, dir, speed);
    CHECK_CLOSE(2.0, newDirAndSpeed.speed(), 1e-6);
    CHECK_CLOSE(0.0, newDirAndSpeed.direction()[0], 1e-6);
    CHECK_CLOSE(-1.0, newDirAndSpeed.direction()[2], 1e-6);
  }

  TEST_FIXTURE( CylindricalGrid, CalcIndices ){
    Position_t pos( 0.5, 0.0, 0.5 );
    auto indices = cyl.calcIndices(pos);
    CHECK_EQUAL(0, indices[0]);
    CHECK_EQUAL(1, indices[1]);
  }

  TEST_FIXTURE( CylindricalGrid, getMinRadialDistAndDir){
    Position_t pos = { -1.0, -1.0, 0.5};
    Direction_t dir = Direction_t{1.0, 1.0, 0.0}.normalize();
    auto indices = cyl.calcIndices(pos);
    auto distAndDir = cyl.getMinRadialDistAndDir(pos, dir, indices[R]);
    CHECK_CLOSE(sqrt(2) - 1.0, distAndDir.distance(), 1e-6);
    CHECK_EQUAL(R, distAndDir.dimension());
    CHECK_EQUAL(false, distAndDir.isPositiveDir());
  }

  TEST_FIXTURE( CylindricalGrid, GetMinDistToSurface){
    Position_t pos( 0.5, 0.0,  0.5 );
    Position_t dir(1, 0, 0);
    auto indices = cyl.calcIndices(pos);
    auto distAndDir = cyl.getMinDistToSurface(pos, dir, indices.data());
    CHECK_CLOSE(0.5, distAndDir.distance(), 1e-6);
    CHECK_EQUAL(R, distAndDir.dimension());
    CHECK_EQUAL(true, distAndDir.isPositiveDir());

    pos = { 0.0, 0.5,  0.5 };
    dir = {0, -1, 0};
    distAndDir = cyl.getMinDistToSurface(pos, dir, indices.data());
    CHECK_CLOSE(1.5, distAndDir.distance(), 1e-6);
    CHECK_EQUAL(R, distAndDir.dimension());
    CHECK_EQUAL(true, distAndDir.isPositiveDir());

    pos = { 0.0, 0.0,  0.5 };
    dir = {0, 0, 1};
    distAndDir = cyl.getMinDistToSurface(pos, dir, indices.data());
    CHECK_CLOSE(0.5, distAndDir.distance(), 1e-6);
    CHECK_EQUAL(CZ, distAndDir.dimension());
    CHECK_EQUAL(true, distAndDir.isPositiveDir());

    pos = { 0.0, 0.0,  0.5 };
    dir = {0, 0, -1};
    distAndDir = cyl.getMinDistToSurface(pos, dir, indices.data());
    CHECK_CLOSE(0.5, distAndDir.distance(), 1e-6);
    CHECK_EQUAL(CZ, distAndDir.dimension());
    CHECK_EQUAL(false, distAndDir.isPositiveDir());

    pos = { 1.1, 1.1,  0.5 };
    dir = {-1, 0, 0};
    indices = cyl.calcIndices(pos);
    distAndDir = cyl.getMinDistToSurface(pos, dir, indices.data());
    CHECK_CLOSE(1.1 + Math::sqrt(4.0 - 1.1*1.1), distAndDir.distance(), 1e-6);
    CHECK_EQUAL(R, distAndDir.dimension());
    CHECK_EQUAL(true, distAndDir.isPositiveDir());
  }

  TEST_FIXTURE( CylindricalGrid, isMovingInward){
    Position_t pos = { -1.5, -1.5, -0.75 };
    Direction_t dir = Direction_t{-Math::sqrt(two), -Math::sqrt(two), one}.normalize();
    CHECK(not cyl.isMovingInward(pos, dir));

    dir = Direction_t{Math::sqrt(two), Math::sqrt(two), one}.normalize();
    CHECK(cyl.isMovingInward(pos, dir));
  }

  TEST_FIXTURE( CylindricalGrid, DistanceToInsideOfMesh){
    Position_t pos = { -1.5, -1.5, -0.75 };
    Direction_t dir = Direction_t{-1.0, -1.0, 0.0};
    auto distanceToInsideOfMesh = cyl.getDistanceToInsideOfMesh(pos, dir);
    CHECK_EQUAL(std::numeric_limits<gpuRayFloat_t>::infinity(), distanceToInsideOfMesh);

    dir = Direction_t{Math::sqrt(two), Math::sqrt(two), one}.normalize();
    distanceToInsideOfMesh = cyl.getDistanceToInsideOfMesh(pos, dir);
    auto answer = (Math::sqrt(2*1.5*1.5) - 2.0)/Math::sqrt(dir[0]*dir[0] + dir[1]*dir[1]);
    CHECK_CLOSE(answer, distanceToInsideOfMesh, 1e-6);

    pos = { -1.5, -1.5, -1.5 };
    distanceToInsideOfMesh = cyl.getDistanceToInsideOfMesh(pos, dir);
    answer = 0.5/dir[2];
    CHECK_CLOSE(answer, distanceToInsideOfMesh, 1e-6);

    dir = Direction_t{Math::sqrt(two), Math::sqrt(two), -one}.normalize();
    distanceToInsideOfMesh = cyl.getDistanceToInsideOfMesh(pos, dir);
    CHECK_EQUAL(std::numeric_limits<gpuRayFloat_t>::infinity(), distanceToInsideOfMesh);
  }
}

} // end namespace

