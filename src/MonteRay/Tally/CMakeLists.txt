target_sources(MonteRay 
PUBLIC 
  "${prefix}/MonteRay/Tally/Tally.hh"
  "${prefix}/MonteRay/Tally/ExpectedPathLength.hh"
  "${prefix}/MonteRay/Tally/NextEventEstimator.hh"
  "${prefix}/MonteRay/Tally/NextEventEstimator.t.hh"
PRIVATE
  "${prefix}/MonteRay/Tally/Tally.cu"
  "${prefix}/MonteRay/Tally/NextEventEstimator.cu"
)

if (MonteRay.tests)
  add_subdirectory(unit_test)
  add_subdirectory(punit_test)
endif()
