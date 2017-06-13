SET(CUDA_SEPARABLE_COMPILATION ON)
file(GLOB CUDA_TEST_SRCS "*.cu")
cuda_add_library(CrossSectionUnitTestHelper SHARED ${CUDA_TEST_SRCS} ${AllCudaFiles})

include( template_select )

list( APPEND ${UnitName}_packages CUDA )

set( ToolkitLibs CrossSectionUnitTestHelper Utilities cuda rt )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()