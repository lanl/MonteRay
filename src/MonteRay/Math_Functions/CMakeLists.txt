target_sources(MonteRay 
PUBLIC 
  "${prefix}/MonteRay/Math_Functions/BinarySearch.hh"
  "${prefix}/MonteRay/Math_Functions/LinearSearch.hh"
  "${prefix}/MonteRay/Math_Functions/MonteRay_QuadraticRootFinder.hh"
  "${prefix}/MonteRay/Math_Functions/MonteRayVector3D.hh"
PRIVATE
  "${prefix}/MonteRay/Math_Functions/MonteRay_QuadraticRootFinder.cu"
)

if (MonteRay.tests)
  add_subdirectory(unit_test)
endif()
