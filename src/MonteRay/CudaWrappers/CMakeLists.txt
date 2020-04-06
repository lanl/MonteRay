target_sources(MonteRay 
PUBLIC 
  "${prefix}/MonteRay/CudaWrappers/StreamAndEvent.hh"
PRIVATE
  "${prefix}/MonteRay/CudaWrappers/StreamAndEvent.cu"
)

if (MonteRay.tests)
  add_subdirectory(unit_test)
endif()











