target_sources(MonteRay 
PUBLIC 
  "${prefix}/MonteRay/ThirdParty/ManagedAllocator.hh"
  "${prefix}/MonteRay/ThirdParty/ManagedResource.hh"
  "${prefix}/MonteRay/ThirdParty/Math.hh"
PRIVATE
  "${prefix}/MonteRay/ThirdParty/ManagedAllocator.cu"
  "${prefix}/MonteRay/ThirdParty/ManagedResource.cu"
)

if (MonteRay.tests)
  add_subdirectory(unit_test)
endif()
