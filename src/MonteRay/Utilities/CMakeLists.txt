include( template_select )

#file(GLOB LIBRARY_SRCS "*.hh" "*.cc" )
#add_library(Utilities ${LIBRARY_SRCS} )
#set_target_properties(Utilities PROPERTIES LINKER_LANGUAGE CXX) 

set( config_srcs MonteRayVersion.cc )

list( APPEND ${UnitName}_packages UnitTest )

set( ToolkitLibs )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()

