file(GLOB CUDA_SRCS "*.h" "*.cpp" "*.hh" "*.cu")

FIND_PACKAGE(CUDA REQUIRED)

INCLUDE(FindCUDA)

SET(CUDA_SEPARABLE_COMPILATION ON)
SET(CUDA_NVCC_FLAGS -DCUDA; -arch=sm_52;-O3; --use_fast_math; -Xcompiler -fPIC; --relocatable-device-code=true;--cudart static)

#CUDA_COMPILE( CUDA_GEN_FILES ${CUDA_SRCS} ${CPP_SRCS} )
#CUDA_COMPILE_PTX( CUDA_PTXGEN_FILES ${CUDA_SRCS} ${CPP_SRCS} )
#message( "CPP_SRCS=${CPP_SRCS}" )
#cuda_add_library(MonteRay STATIC ${CUDA_SRCS} ${CPP_SRCS}  ${CUDA_PTXGEN_FILES})
cuda_add_library(MonteRay STATIC ${CUDA_SRCS} )

include( template_select )

list( APPEND ${UnitName}_packages )

set( ToolkitLibs )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()

