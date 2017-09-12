SET(CUDA_SEPARABLE_COMPILATION ON)
file(GLOB CUDA_TEST_SRCS "*.cu" "*.cc")
CUDA_INCLUDE_DIRECTORIES( ${UnitTest_INCLUDE_DIRS} )
cuda_add_library(MaterialUnitTestHelper SHARED ${CUDA_TEST_SRCS} ${AllCudaFiles} )

include( template_select )

list( APPEND ${UnitName}_packages CUDA )

set( ToolkitLibs MaterialUnitTestHelper Utilities cuda rt )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()
#include_directories(${CMAKE_CURRENT_BINARY_DIR}/..)
message( STATUS RayTrace unit_test -- UnitTest_INCLUDE_DIRS = ${UnitTest_INCLUDE_DIRS} )
