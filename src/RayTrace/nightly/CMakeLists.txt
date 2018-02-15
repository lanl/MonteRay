include( template_select )

set( ToolkitLibs Utilities Math_Functions GPU_Utilities CrossSection Material Geometry Particle lnk3dnt )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()