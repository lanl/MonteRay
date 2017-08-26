include( template_select )
SET(CUDA_SEPARABLE_COMPILATION ON)
file(GLOB CUDA_TEST_SRCS "*.cu")
cuda_add_library(_Depreciated_Stuff_TestHelper SHARED ${CUDA_TEST_SRCS} ${AllCudaFiles} )

list( APPEND ${UnitName}_packages CUDA )

set( ToolkitLibs _Depreciated_Stuff_TestHelper Utilities OutputControls cuda )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()

