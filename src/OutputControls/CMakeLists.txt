include( template_select )

file(GLOB LIBRARY_SRCS "*.hh" "*.cc" )
add_library(OutputControls ${LIBRARY_SRCS} )
set_target_properties(OutputControls PROPERTIES LINKER_LANGUAGE CXX)

list( APPEND ${UnitName}_packages )

set( ToolkitLibs )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()
