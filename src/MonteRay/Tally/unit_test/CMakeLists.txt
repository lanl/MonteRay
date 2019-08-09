include( template_select )

set( ToolkitLibs ThirdParty lnk3dnt Utilities CrossSection Material Geometry Particle Math_Functions IPComm GPU_Utilities )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()
