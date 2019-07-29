#include <UnitTest++.h>

#include "MaterialProperties.hh"
#include "MonteRay_MaterialProperties.hh"
#include "MonteRay_SetupMaterialProperties.hh"
#include "MonteRay_ReadLnk3dnt.hh"

namespace MaterialProperties_tester{

using namespace MonteRay;

/* typedef MonteRay_CellProperties CellProperties; */
/* typedef MonteRay_ReadLnk3dnt ReadLnk3dnt; */
/* typedef MonteRay_MaterialSpec MaterialSpec; */
/* typedef MonteRay_MaterialProperties MaterialProperties; */
using Offset_t = size_t;
using Material_Index_t = MonteRay_CellProperties::Material_Index_t;
using Temperature_t = MonteRay_CellProperties::Temperature_t;
using MatID_t = MonteRay_CellProperties::MatID_t;
using Density_t = MonteRay_CellProperties::Density_t;

struct Cell{
  std::vector<MatID_t> ids;
  std::vector<Density_t> densities;
  auto getNumMaterials() const { return ids.size(); }
  auto getMaterialID(int i) const { return ids[i]; }
  auto getMaterialDensity(int i) const { return densities[i]; }
};

SUITE( NewMaterialProperties_tests ) {

  TEST(buildAndAccess_MaterialProperties_Data){
    auto mpb = MaterialProperties_Data::Builder();
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

    CHECK_EQUAL(1.1f, matProps.getDensity(0, 0));
    CHECK_EQUAL(2.2f, matProps.getDensity(0, 1));
    CHECK_EQUAL(3.3f, matProps.getDensity(1, 0));
    CHECK_EQUAL(4.4f, matProps.getDensity(1, 1));

    CHECK_EQUAL(1, matProps.getMatID(0, 0));
    CHECK_EQUAL(2, matProps.getMatID(0, 1));
    CHECK_EQUAL(3, matProps.getMatID(1, 0));
    CHECK_EQUAL(4, matProps.getMatID(1, 1));
  }

}

}// end namespace MaterialProperties_tester
