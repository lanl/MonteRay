target_sources(MonteRay 
PUBLIC 
  "${prefix}/MonteRay/GPU_Utilities/Array.hh"
  "${prefix}/MonteRay/GPU_Utilities/gpuAddTwoDoubles.hh"
  "${prefix}/MonteRay/GPU_Utilities/GPUAtomicAdd.hh"
  "${prefix}/MonteRay/GPU_Utilities/GPUErrorCheck.hh"
  "${prefix}/MonteRay/GPU_Utilities/GPUTiming.hh"
  "${prefix}/MonteRay/GPU_Utilities/GPUUtilityFunctions.hh"
  "${prefix}/MonteRay/GPU_Utilities/Invokers.hh"
  "${prefix}/MonteRay/GPU_Utilities/MonteRayCopyMemory.hh"
  "${prefix}/MonteRay/GPU_Utilities/MonteRayCopyMemory.t.hh"
  "${prefix}/MonteRay/GPU_Utilities/MonteRayManagedMemory.hh"
  "${prefix}/MonteRay/GPU_Utilities/MonteRayMemory.hh"
  "${prefix}/MonteRay/GPU_Utilities/MonteRay_SingleValueCopyMemory.hh"
  "${prefix}/MonteRay/GPU_Utilities/MonteRay_SingleValueCopyMemory.t.hh"
PRIVATE
  "${prefix}/MonteRay/GPU_Utilities/gpuAddTwoDoubles.cu"
  "${prefix}/MonteRay/GPU_Utilities/GPUTiming.cu"
  "${prefix}/MonteRay/GPU_Utilities/GPUUtilityFunctions.cu"
  "${prefix}/MonteRay/GPU_Utilities/MonteRayManagedMemory.cu"
  "${prefix}/MonteRay/GPU_Utilities/MonteRayMemory.cu"
)

if (MonteRay.tests)
  add_subdirectory(unit_test)
endif()
