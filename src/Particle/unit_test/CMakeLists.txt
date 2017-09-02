SET(CUDA_SEPARABLE_COMPILATION ON)
file(GLOB CUDA_TEST_SRCS "*.cu" )
CUDA_INCLUDE_DIRECTORIES( ${UnitTest_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR} )
cuda_add_library(ParticleUnitTestHelper SHARED ${CUDA_TEST_SRCS} ${AllCudaFiles} )

include( template_select )

list( APPEND ${UnitName}_packages CUDA )

set( ToolkitLibs ParticleUnitTestHelper Utilities cuda )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()
#message( STATUS Particle unit_test -- UnitTest_INCLUDE_DIRS = ${UnitTest_INCLUDE_DIRS} )

