file(GLOB CUDA_SRCS "*.h" "*.cpp" "*.hh" "*.cu" "*.cc")

include( template_select )

list( APPEND ${UnitName}_packages MPI )

set( ToolkitLibs ThirdParty )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()

