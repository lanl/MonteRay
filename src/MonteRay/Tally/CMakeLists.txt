target_sources(MonteRay 
PUBLIC 
  "${prefix}/MonteRay/Tally/MonteRayNextEventEstimator.hh"
  "${prefix}/MonteRay/Tally/MonteRayNextEventEstimator.t.hh"
  "${prefix}/MonteRay/Tally/MonteRayTally.hh"
PRIVATE
  "${prefix}/MonteRay/Tally/MonteRayNextEventEstimator.cu"
  "${prefix}/MonteRay/Tally/MonteRayTally.cu"
)

if (MonteRay.tests)
  add_subdirectory(unit_test)
  add_subdirectory(punit_test)
endif()
