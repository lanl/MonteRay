target_sources(MonteRay 
PUBLIC 
  "${prefix}/MonteRay/Particle/Ray.hh"
  "${prefix}/MonteRay/Particle/RayList.hh"
  "${prefix}/MonteRay/Particle/RayListInterface.hh"
  "${prefix}/MonteRay/Particle/Ray.t.hh"
  "${prefix}/MonteRay/Particle/RayWorkInfo.hh"
PRIVATE
  "${prefix}/MonteRay/Particle/Ray.cu"
  "${prefix}/MonteRay/Particle/RayList.cu"
  "${prefix}/MonteRay/Particle/RayListInterface.cu"
)

if (MonteRay.tests)
  add_subdirectory(unit_test)
  add_subdirectory(fi_test)
endif()
