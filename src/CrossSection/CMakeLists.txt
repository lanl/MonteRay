include( template_select )

list( APPEND ${UnitName}_packages Utilities CUDA )

set( ToolkitLibs )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()

