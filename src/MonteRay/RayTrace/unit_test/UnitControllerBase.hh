#ifndef MR_TEST_UNITCONTROLLERBASE
#define MR_TEST_UNITCONTROLLERBASE
#include "CrossSection.hh"
#include "CrossSectionList.hh"
#include "Material.hh"
#include "MaterialList.hh"
namespace MonteRay{

class UnitControllerBase{
  protected:
    std::unique_ptr<MaterialList> pMatList;
    std::unique_ptr<CrossSectionList> pXsList;

  public:
    UnitControllerBase(){

      CrossSectionList::Builder xsListBuilder;
      // create ficticous xs w/ zaid 12345
      std::array<gpuFloatType_t, 2> energies = {0.00001, 100.0};
      std::array<gpuFloatType_t, 2> xs_values = {1.0, 1.0};
      constexpr gpuFloatType_t AWR = 1.0;
      CrossSectionBuilder xsBuilder(12345, energies, xs_values, neutron, AWR);

      xsListBuilder.add(xsBuilder.construct());
      pXsList = std::make_unique<CrossSectionList>(xsListBuilder.build());

      // using CrossSectionList as a dictionary, create materials and material list
      auto matBuilder = Material::make_builder(*pXsList);

      matBuilder.addIsotope(1.0, 12345);
      // create list builder and add material to list w/ id 0 
      MaterialList::Builder matListBuilder(0, matBuilder.build());
      pMatList = std::make_unique<MaterialList>(matListBuilder.build());
    }
};
} // end namespace MonteRay
#endif
