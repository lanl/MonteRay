file(GLOB CUDA_TEST_SRCS "*.cu")
cuda_add_library(RayTraceUnitTestHelper STATIC ${CUDA_TEST_SRCS} ${CUDA_SRCS})
#cuda_add_library(RayTraceUnitTestHelper STATIC ${CUDA_TEST_SRCS})

include( template_select )

list( APPEND ${UnitName}_packages CUDA )

set( ToolkitLibs RayTraceUnitTestHelper GPU_Utilities Utilities cuda rt )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()
#include_directories(${CMAKE_CURRENT_BINARY_DIR}/..)