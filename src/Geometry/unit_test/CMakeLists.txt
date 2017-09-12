SET(CUDA_SEPARABLE_COMPILATION ON)
file(GLOB CUDA_TEST_SRCS "*.cu" "*.cc" )
CUDA_INCLUDE_DIRECTORIES( ${UnitTest_INCLUDE_DIRS} )
cuda_add_library(GeometryUnitTestHelper STATIC ${CUDA_TEST_SRCS} ${AllCudaFiles} )

include( template_select )

list( APPEND ${UnitName}_packages CUDA )

set( ToolkitLibs GeometryUnitTestHelper lnk3dnt Utilities cuda rt )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()