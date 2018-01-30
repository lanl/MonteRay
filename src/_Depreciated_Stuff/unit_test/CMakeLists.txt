include( template_select )

if( DEFINED enable_cuda )
  SET(CUDA_SEPARABLE_COMPILATION ON)
  file(GLOB CUDA_TEST_SRCS "*.cc")
  cuda_add_library(_Depreciated_Stuff_TestHelper SHARED ${CUDA_TEST_SRCS} ${AllCudaFiles} )
else()
  file(GLOB CUDA_TEST_SRCS "*.cc")
  add_library(_Depreciated_Stuff_TestHelper SHARED ${CUDA_TEST_SRCS} ${AllCudaFiles} )
endif()

set( ToolkitLibs _Depreciated_Stuff_TestHelper Utilities OutputControls )

if( DEFINED enable_cuda )
  list( APPEND ${UnitName}_packages CUDA )
  list( APPEND ToolkitLibs cuda  )
endif()


if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()

