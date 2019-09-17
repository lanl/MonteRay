target_sources(MonteRay 
PRIVATE
  "${prefix}/MonteRay/OutputControls/MonteRayMultiStream.cc"
PUBLIC
  "${prefix}/MonteRay/OutputControls/MonteRayMultiStream.hh"
)

if (MonteRay.tests)
  add_subdirectory(unit_test)
endif()
