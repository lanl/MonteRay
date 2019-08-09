include( template_select )

list( APPEND ${UnitName}_packages Utilities )

set( ToolkitLibs )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()

configure_file(MonteRayPreprocessorDefinitions.hh.in MonteRayPreprocessorDefinitions.hh @ONLY)
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/MonteRayPreprocessorDefinitions.hh DESTINATION include )

