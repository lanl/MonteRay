target_sources(MonteRay 
PUBLIC 
  "${prefix}/MonteRay/Tally/MonteRayNextEventEstimator.hh"
  "${prefix}/MonteRay/Tally/MonteRayNextEventEstimator.t.hh"
  "${prefix}/MonteRay/Tally/MonteRayTally.hh"
  "${prefix}/MonteRay/Tally/Tally.hh"
  "${prefix}/MonteRay/Tally/NextEventEstimator.hh"
  "${prefix}/MonteRay/Tally/NextEventEstimator.t.hh"
PRIVATE
  "${prefix}/MonteRay/Tally/MonteRayNextEventEstimator.cu"
  "${prefix}/MonteRay/Tally/MonteRayTally.cu"
  "${prefix}/MonteRay/Tally/Tally.cu"
)

if (MonteRay.tests)
  add_subdirectory(unit_test)
  add_subdirectory(punit_test)
endif()
