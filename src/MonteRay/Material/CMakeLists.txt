target_sources(MonteRay 
PUBLIC 
  "${prefix}/MonteRay/Material/MonteRayMaterial.hh"
  "${prefix}/MonteRay/Material/MonteRayMaterialList.hh"
PRIVATE
  "${prefix}/MonteRay/Material/MonteRayMaterial.cu"
  "${prefix}/MonteRay/Material/MonteRayMaterialList.cu"
)

if (MonteRay.tests)
  add_subdirectory(unit_test)
endif()
