file(GLOB CUDA_RAYTRACE_FITEST_SRCS "*.cu")
cuda_add_library(gpuFITestHelper STATIC ${CUDA_RAYTRACE_FITEST_SRCS} ${CUDA_SRCS})

include( template_select )

list( APPEND ${UnitName}_packages )

set( ToolkitLibs gpuFITestHelper Utilities cuda )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()