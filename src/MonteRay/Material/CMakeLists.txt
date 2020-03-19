target_sources(MonteRay 
PUBLIC 
  "${prefix}/MonteRay/Material/Material.hh"
  "${prefix}/MonteRay/Material/MaterialList.hh"
)

if (MonteRay.tests)
  add_subdirectory(unit_test)
endif()
