include( template_select )

set( ToolkitLibs lnk3dnt Utilities GPU_Utilities Math_Functions RayTrace CrossSection Material Particle)

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()