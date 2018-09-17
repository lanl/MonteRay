include( template_select )

list( APPEND ${UnitName}_packages MPI )

set( ToolkitLibs Utilities pthread )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()

