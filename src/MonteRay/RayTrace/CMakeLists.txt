target_sources(MonteRay 
PUBLIC 
  "${prefix}/MonteRay/RayTrace/ExpectedPathLength.hh"
  "${prefix}/MonteRay/RayTrace/ExpectedPathLength.t.hh"
  "${prefix}/MonteRay/RayTrace/RayListController.hh"
  "${prefix}/MonteRay/RayTrace/RayListController.t.hh"
  "${prefix}/MonteRay/RayTrace/gpuTally.hh"
PRIVATE
  "${prefix}/MonteRay/RayTrace/gpuTally.cu"
  "${prefix}/MonteRay/RayTrace/ExpectedPathLength.cu"
  "${prefix}/MonteRay/RayTrace/RayListController.cu"
)

if (MonteRay.tests)
  add_subdirectory(unit_test)
  add_subdirectory(fi_test)
  add_subdirectory(nightly)
  add_subdirectory(pfi_test)
endif()






