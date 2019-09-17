target_sources(MonteRay 
PUBLIC 
  "${prefix}/MonteRay/Containers/SimpleView.hh"
  "${prefix}/MonteRay/Containers/SimpleVector.hh"
)

if (MonteRay.tests)
  add_subdirectory(unit_test)
endif()











