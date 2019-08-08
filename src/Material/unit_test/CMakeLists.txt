include( template_select )

set( ToolkitLibs Utilities CrossSection GPU_Utilities IPComm ThirdParty )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()
