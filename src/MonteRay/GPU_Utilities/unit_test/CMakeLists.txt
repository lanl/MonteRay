include( template_select )

set( ToolkitLibs Utilities Math_Functions CrossSection Material Geometry Particle IPComm )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()