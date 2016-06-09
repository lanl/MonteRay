include( template_select )

set( config_srcs ToolkitVersion.cc )

list( APPEND ${UnitName}_packages UnitTest )

set( ToolkitLibs )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()

