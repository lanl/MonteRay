target_sources(MonteRay PUBLIC 
  "${prefix}/MonteRay.hh"
  "${prefix}/MonteRay/Constants.hh"
  "${prefix}/MonteRay/Containers.hh"
  "${prefix}/MonteRay/CrossSection.hh"
  "${prefix}/MonteRay/Geometry.hh"
  "${prefix}/MonteRay/GPU_Utilities.hh"
  "${prefix}/MonteRay/IPComm.hh"
  "${prefix}/MonteRay/Material.hh"
  "${prefix}/MonteRay/Math_Functions.hh"
  "${prefix}/MonteRay/OutputControls.hh"
  "${prefix}/MonteRay/Parameters.hh"
  "${prefix}/MonteRay/Particle.hh"
  "${prefix}/MonteRay/RayTrace.hh"
  "${prefix}/MonteRay/Tally.hh"
  "${prefix}/MonteRay/ThirdParty.hh"
  "${prefix}/MonteRay/Utilities.hh"
)

target_include_directories(MonteRay PUBLIC
  "${prefix}/MonteRay"
  "${prefix}/MonteRay/Constants"
  "${prefix}/MonteRay/Containers"
  "${prefix}/MonteRay/CrossSection"
  "${prefix}/MonteRay/Geometry"
  "${prefix}/MonteRay/GPU_Utilities"
  "${prefix}/MonteRay/IPComm"
  "${prefix}/MonteRay/Material"
  "${prefix}/MonteRay/Math_Functions"
  "${prefix}/MonteRay/OutputControls"
  "${prefix}/MonteRay/Parameters"
  "${prefix}/MonteRay/Particle"
  "${prefix}/MonteRay/RayTrace"
  "${prefix}/MonteRay/Tally"
  "${prefix}/MonteRay/ThirdParty"
  "${prefix}/MonteRay/Utilities"
)

target_sources(MonteRay::testing INTERFACE
  "${prefix}/MonteRay/TestUtilities.hh"
)

target_include_directories(MonteRay::testing INTERFACE
  "${prefix}/MonteRay/TestUtilities"
)

add_subdirectory(Constants)        
add_subdirectory(Containers)
add_subdirectory(OutputControls)
add_subdirectory(Math_Functions)
add_subdirectory(Geometry)
add_subdirectory(GPU_Utilities)
add_subdirectory(CrossSection)
add_subdirectory(IPComm)
add_subdirectory(Material)
add_subdirectory(Parameters)
add_subdirectory(Particle)
add_subdirectory(RayTrace)
add_subdirectory(Tally)
add_subdirectory(ThirdParty)
add_subdirectory(Utilities)
