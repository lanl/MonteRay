include( template_select )

set( ToolkitLibs Utilities CrossSection GPU_Utilities IPComm )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()
