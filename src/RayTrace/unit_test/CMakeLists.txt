file(GLOB CUDA_TEST_SRCS "*.cu")
cuda_add_library(gpuTestHelper STATIC ${CUDA_TEST_SRCS} ${CUDA_SRCS})

include( template_select )

list( APPEND ${UnitName}_packages )

set( ToolkitLibs gpuTestHelper Utilities cuda )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()