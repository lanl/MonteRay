include( template_select )

set( ToolkitLibs lnk3dnt Utilities Math_Functions RayTrace Tally CrossSection Material Particle IPComm GPU_Utilities )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()