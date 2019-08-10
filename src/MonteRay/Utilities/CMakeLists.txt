target_sources(MonteRay 
PUBLIC 
  "${prefix}/MonteRay/Utilities/MonteRay_binaryIO.hh"
  "${prefix}/MonteRay/Utilities/MonteRay_timer.hh"
  "${prefix}/MonteRay/Utilities/MonteRayVersion.hh"
)

if (MonteRay.tests)
  add_subdirectory(unit_test)
endif()
