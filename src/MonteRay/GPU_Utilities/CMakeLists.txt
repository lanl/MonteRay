target_sources(MonteRay 
PUBLIC 
  "${prefix}/MonteRay/GPU_Utilities/Array.hh"
  "${prefix}/MonteRay/GPU_Utilities/GPUAtomicAdd.hh"
  "${prefix}/MonteRay/GPU_Utilities/GPUErrorCheck.hh"
  "${prefix}/MonteRay/GPU_Utilities/GPUTiming.hh"
  "${prefix}/MonteRay/GPU_Utilities/GPUUtilityFunctions.hh"
  "${prefix}/MonteRay/GPU_Utilities/Invokers.hh"
  "${prefix}/MonteRay/GPU_Utilities/MonteRayManagedMemory.hh"
  "${prefix}/MonteRay/GPU_Utilities/MonteRayMemory.hh"
PRIVATE
  "${prefix}/MonteRay/GPU_Utilities/GPUTiming.cu"
  "${prefix}/MonteRay/GPU_Utilities/GPUUtilityFunctions.cu"
  "${prefix}/MonteRay/GPU_Utilities/MonteRayManagedMemory.cu"
  "${prefix}/MonteRay/GPU_Utilities/MonteRayMemory.cu"
)

if (MonteRay.tests)
  add_subdirectory(unit_test)
endif()
