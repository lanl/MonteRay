#include <UnitTest++.h>

#include "MaterialProperties.hh"
#include "MonteRay_ReadLnk3dnt.hh"

namespace MaterialProperties_tester{

using namespace MonteRay;

/* typedef MonteRay_CellProperties CellProperties; */
/* typedef MonteRay_ReadLnk3dnt ReadLnk3dnt; */
/* typedef MonteRay_MaterialSpec MaterialSpec; */
/* typedef MonteRay_MaterialProperties MaterialProperties; */
using Offset_t = size_t;
using MatID_t = MaterialProperties::MatID_t;
using Density_t = MaterialProperties::Density_t;
using Cell = MaterialProperties::Builder::Cell;

SUITE( NewMaterialProperties_tests ) {

  TEST(buildAndAccess_MaterialProperties){
    auto mpb = MaterialProperties::Builder();
    Cell cell1{ {1, 2}, {1.1, 2.2} };
    Cell cell2{ {3}, {3.3} };
    mpb.addCell(cell1);
    mpb.addCell(cell2);
    mpb.addCellMaterial(1, 4, 4.4);
    auto matProps = mpb.build();

    CHECK_EQUAL(2, matProps.numCells());
    CHECK_EQUAL(2, matProps.numMats(0));
    CHECK_EQUAL(2, matProps.numMats(1));
    CHECK_EQUAL(4, matProps.numMaterialComponents());

    CHECK_EQUAL(static_cast<gpuFloatType_t>(1.1), matProps.getDensity(0, 0));
    CHECK_EQUAL(static_cast<gpuFloatType_t>(2.2), matProps.getDensity(0, 1));
    CHECK_EQUAL(static_cast<gpuFloatType_t>(3.3), matProps.getDensity(1, 0));
    CHECK_EQUAL(static_cast<gpuFloatType_t>(4.4), matProps.getDensity(1, 1));

    CHECK_EQUAL(1, matProps.getMatID(0, 0));
    CHECK_EQUAL(2, matProps.getMatID(0, 1));
    CHECK_EQUAL(3, matProps.getMatID(1, 0));
    CHECK_EQUAL(4, matProps.getMatID(1, 1));

    CHECK_EQUAL(false, matProps.usingMaterialMotion());
    CHECK_EQUAL(0, matProps.numVelocities());
  }


  TEST(build_via_lnk3dnt){
    using ReadLnk3dnt = MonteRay_ReadLnk3dnt;
    ReadLnk3dnt readerObject( "lnk3dnt/3iso_3shell_godiva.lnk3dnt" );
    readerObject.ReadMatData();
    auto mpb = MaterialProperties::Builder(readerObject);
    auto matProps = mpb.build();

    CHECK_EQUAL(size_t(8000), matProps.numCells());

    int cellNumber = 7010;
    CHECK_EQUAL( size_t(2) , matProps.numMaterials(cellNumber) );
    CHECK_EQUAL( 2         , matProps.getMaterialID(cellNumber,0) );
    CHECK_EQUAL( 3         , matProps.getMaterialID(cellNumber,1) );
    CHECK_CLOSE( 17.8772   , matProps.getMaterialDensity(cellNumber,0), 1e-04);
    CHECK_CLOSE(  0.8228   , matProps.getMaterialDensity(cellNumber,1), 1e-04);

    cellNumber = 7410;
    CHECK_EQUAL( size_t(2) , matProps.numMaterials(cellNumber) );
    CHECK_EQUAL( 3         , matProps.getMaterialID(cellNumber,0) );
    CHECK_EQUAL( 4         , matProps.getMaterialID(cellNumber,1) );
    CHECK_CLOSE( 8.6768    , matProps.getMaterialDensity(cellNumber,0), 1e-04);
    CHECK_CLOSE( 4.4132    , matProps.getMaterialDensity(cellNumber,1), 1e-04);

    CHECK_EQUAL(false, matProps.usingMaterialMotion());
    CHECK_EQUAL(0, matProps.numVelocities());

  }

  TEST(renumber_mat_ids){
    struct MaterialList{
      MatID_t materialIDtoIndex(MatID_t id) const {
        if (id == 12){ return 10; }
        if (id == 27){ return 20; }
        else { return 30; }
      }
    };

    auto mpb = MaterialProperties::Builder();
    Cell cell1{ {12, 27}, {1.1, 2.2} };
    Cell cell2{ {39}, {3.3} };
    mpb.addCell(cell1);
    mpb.addCell(cell2);
    mpb.renumberMaterialIDs(MaterialList{});
    auto matProps = mpb.build();

    CHECK_EQUAL(10, matProps.getMatID(0, 0));
    CHECK_EQUAL(20, matProps.getMatID(0, 1));
    CHECK_EQUAL(30, matProps.getMatID(1, 0));

    CHECK_EQUAL(false, matProps.usingMaterialMotion());
    CHECK_EQUAL(0, matProps.numVelocities());
  }

  TEST(read_and_write){
    auto mpb = MaterialProperties::Builder();
    Cell cell1{ {1, 2}, {1.1, 2.2} };
    Cell cell2{ {3}, {3.3} };
    mpb.addCell(cell1);
    mpb.addCell(cell2);
    auto matProps = mpb.build();

    std::stringstream stream;
    matProps.write(stream);
    MaterialProperties::Builder another_mpb;
    auto another_matProps = another_mpb.read(stream);

    CHECK_EQUAL(2, matProps.numCells());
    CHECK_EQUAL(2, matProps.numMats(0));
    CHECK_EQUAL(1, matProps.numMats(1));
    CHECK_EQUAL(3, matProps.numMaterialComponents());

    CHECK_EQUAL(static_cast<gpuFloatType_t>(1.1), matProps.getDensity(0, 0));
    CHECK_EQUAL(static_cast<gpuFloatType_t>(2.2), matProps.getDensity(0, 1));
    CHECK_EQUAL(static_cast<gpuFloatType_t>(3.3), matProps.getDensity(1, 0));

    CHECK_EQUAL(1, matProps.getMatID(0, 0));
    CHECK_EQUAL(2, matProps.getMatID(0, 1));
    CHECK_EQUAL(3, matProps.getMatID(1, 0));

    CHECK_EQUAL(false, matProps.usingMaterialMotion());
    CHECK_EQUAL(0, matProps.numVelocities());
  }

  TEST(usingMaterialMotion){
    auto mpb = MaterialProperties::Builder();
    Cell cell1{ {1}, {1.1} };
    Cell cell2{ {2}, {2.2} };

    mpb.addCell(cell1);
    mpb.addCell(cell2);
    Vector3D<gpuRayFloat_t> cell0Velocity = {1.0, 2.0, 3.0};
    std::vector<Vector3D<gpuRayFloat_t>> cellVelocities{ cell0Velocity, cell0Velocity };
    mpb.setVelocities(cellVelocities);
    Vector3D<gpuRayFloat_t> cell1Velocity = {3.0, 4.0, 5.0};

    mpb.setCellVelocity(1, cell1Velocity);
    auto matProps = mpb.build();
    CHECK_EQUAL(true, matProps.usingMaterialMotion());
    CHECK_EQUAL( cell0Velocity, matProps.getVelocity(0));
    CHECK_EQUAL( cell1Velocity, matProps.getVelocity(1));

    std::stringstream stream;
    matProps.write(stream);
    MaterialProperties::Builder another_mpb;
    auto another_matProps = another_mpb.read(stream);

    CHECK_EQUAL(true, another_matProps.usingMaterialMotion());
    CHECK_EQUAL( cell0Velocity, another_matProps.getVelocity(0));
    CHECK_EQUAL( cell1Velocity, another_matProps.getVelocity(1));
  }

  TEST(incorrectly_setting_velocities){
    auto mpb = MaterialProperties::Builder();
    Vector3D<gpuRayFloat_t> velocity = {1.0, 2.0, 3.0};
    std::vector<Vector3D<gpuRayFloat_t>> velocities{ velocity };
    CHECK_THROW(mpb.setVelocities(velocities), std::exception);
  }

  TEST(initialize_material_description_memReductionEnabled){
    std::vector<MatID_t> ids{1, 2, 3};
    std::vector<Density_t> densities{1.1, 2.2, 3.3};
    int nCells = 3;
    auto mpb = MaterialProperties::Builder{};

    mpb.initializeMaterialDescription(ids, densities, nCells);
    auto matProps = mpb.build();
    CHECK_EQUAL(3, matProps.numCells());

    CHECK_EQUAL(1, matProps.numMats(0));
    CHECK_EQUAL(1, matProps.numMats(1));
    CHECK_EQUAL(1, matProps.numMats(2));

    CHECK_EQUAL(1, matProps.getMatID(0, 0));
    CHECK_EQUAL(2, matProps.getMatID(1, 0));
    CHECK_EQUAL(3, matProps.getMatID(2, 0));

    CHECK_EQUAL(static_cast<gpuFloatType_t>(1.1), matProps.getDensity(0, 0));
    CHECK_EQUAL(static_cast<gpuFloatType_t>(2.2), matProps.getDensity(1, 0));
    CHECK_EQUAL(static_cast<gpuFloatType_t>(3.3), matProps.getDensity(2, 0));

    CHECK_EQUAL(false, matProps.usingMaterialMotion());
    CHECK_EQUAL(0, matProps.numVelocities());
  }

  TEST(cellMaterialIDs){
    auto mpb = MaterialProperties::Builder{};

    std::vector<MatID_t> ids{1, 2, 3};
    std::vector<Density_t> densities{1.1, 2.2, 3.3};
    int nCells = 3;
    mpb.initializeMaterialDescription(ids, densities, nCells);

    mpb.addCellMaterial(0, 4, 4.4);
    mpb.addCellMaterial(2, 5, 5.5);

    const auto matProps = mpb.build();

    auto cellZeroIDs = matProps.cellMaterialIDs(0);
    CHECK_EQUAL(2, cellZeroIDs.size());
    CHECK_EQUAL(1, cellZeroIDs[0]);
    CHECK_EQUAL(4, cellZeroIDs[1]);

    auto cellTwoIDs = matProps.cellMaterialIDs(2);
    CHECK_EQUAL(2, cellTwoIDs.size());
    CHECK_EQUAL(3, cellTwoIDs[0]);
    CHECK_EQUAL(5, cellTwoIDs[1]);

  }


}

}// end namespace MaterialProperties_tester
