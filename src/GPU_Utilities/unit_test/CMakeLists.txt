file(GLOB CUDA_TEST_SRCS "*.cu")
cuda_add_library(GPU_UtilitiesUnitTestHelper STATIC ${CUDA_TEST_SRCS} ${CUDA_SRCS})

include( template_select )

list( APPEND ${UnitName}_packages CUDA )

set( ToolkitLibs GPU_UtilitiesUnitTestHelper GPU_Utilities Utilities cuda rt )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()