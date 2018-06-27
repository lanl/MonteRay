FIND_PACKAGE(CUDA REQUIRED)

INCLUDE(FindCUDA)

#CUDA_VERSION_MAJOR
#CUDA_VERSION_MAJOR
#CUDA_TOOLKIT_ROOT_DIR
#CUDA_SDK_ROOT_DIR
#CUDA_INCLUDE_DIRS

set( GPUTYPE "NONE" )
set( GPUCOMPUTECAPABILITY "-arch=sm_30" )

include(IdentifyGPU)
IdentifyGPU( GPUTYPE GPUCOMPUTECAPABILITY )
message( STATUS "config_nvcc.cmake -- GPU type = ${GPUTYPE}, Compute capability = ${GPUCOMPUTECAPABILITY}"  )

if( DEFINED CUDA_TOOLKIT_ROOT_DIR) 
    message( STATUS "CUDA library found. CUDA_TOOLKIT_ROOT_DIR= ${CUDA_TOOLKIT_ROOT_DIR}"  )
else()
    message( WARNING "CUDA library not found." )
endif()

find_library( CUDART_LIB_PATH 
                  NAMES cudart
                  PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
                        ${CUDA_TOOLKIT_ROOT_DIR}/lib
                  DOC "The cudart library."
                  NO_DEFAULT_PATH 
            )
get_filename_component( CUDART_LIB_DIR ${CUDART_LIB_PATH} PATH )
set( CUDA_LIBRARY_DIRS ${CUDART_LIB_DIR} )

find_library( CUDA_LIB_PATH 
                  NAMES cuda
                  PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
                        ${CUDA_TOOLKIT_ROOT_DIR}/lib
                  PATH_SUFFIXES stubs
                  DOC "The cuda stub library."
                  NO_DEFAULT_PATH 
            )    
get_filename_component( CUDA_LIB_DIR ${CUDA_LIB_PATH} PATH )            
list(APPEND CUDA_LIBRARY_DIRS ${CUDA_LIB_DIR} )   
             
if( DEFINED CUDA_LIBRARY_DIRS) 
    message( STATUS "CUDA_LIBRARY_DIRS= ${CUDA_LIBRARY_DIRS}"  )
else()
    message( WARNING "CUDA_LIBRARY_DIRS not set." )
endif()             

if( DEFINED CUDA_INCLUDE_DIRS) 
    message( STATUS "CUDA_INCLUDE_DIRS= ${CUDA_INCLUDE_DIRS}"  )
else()
    message( WARNING "CUDA_INCLUDE_DIRS not set." )
endif()

if( DEFINED CUDA_LIBRARIES )
    message( STATUS "CUDA_LIBRARIES= ${CUDA_LIBRARIES}"  )
else()
    message( WARNING "CUDA_LIBRARIES not set." )
endif() 

if( NOT CUDA_INCLUDE_DRIVER_TYPES ) 
    find_path( CUDA_INCLUDE_DRIVER_TYPES
        NAMES driver_types.h
        PATHS ${CUDA_TOOLKIT_ROOT_DIR}/include
              ${CUDA_INCLUDE_DIRS}
              DOC "Location of CUDA driver types include file." 
              NO_DEFAULT_PATH )
               
    if( NOT CUDA_INCLUDE_DRIVER_TYPES )
        message( FATAL_ERROR "Location of CUDA driver types 'driver_types.h' include file not found." )
    endif()
    
endif()
set( CUDA_INCLUDE_DRIVER_TYPES_PATH "${CUDA_INCLUDE_DRIVER_TYPES}/driver_types.h" )

# Find driver_types.h, such as cuda/7.5/include/driver_types.h

if(DEFINED ENV{CUDA_ROOT}) 
    set( CUDA_ROOT $ENV{CUDA_ROOT} )
    message( STATUS "config_nvcc.cmake -- Environment variable CUDA_ROOT=${CUDA_ROOT}" )
endif()

#SET(CUDA_SEPARABLE_COMPILATION ON)
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DCUDA; -Xcompiler -fPIC;--relocatable-device-code=true;--cudart static)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DCUDA;-Xcompiler -fPIC;--relocatable-device-code=true;--cudart shared)

# Titan X -arch=sm_52
# K40 -arch=compute_35 -code=sm_35
# Moonlight Tesla M2090 -arch=compute_20 -code=sm_20
# Quadro K420 - 3.0
#list(APPEND CUDA_NVCC_FLAGS ${GPUCOMPUTECAPABILITY})
add_definitions( -DCMAKE_CUDA_FLAGS=${GPUCOMPUTECAPABILITY} )
add_definitions( -DCMAKE_CUDA_FLAGS_DEBUG="${GPUCOMPUTECAPABILITY} -g -G" )

if( CMAKE_BUILD_TYPE STREQUAL "Debug" ) 
    list(APPEND CUDA_NVCC_FLAGS "-G")
    list(APPEND CUDA_NVCC_FLAGS "-g")
else()
    list(APPEND CUDA_NVCC_FLAGS "-O3")
    list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
endif()
#list(APPEND CUDA_NVCC_FLAGS "-O3")
#list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
#list(APPEND CUDA_NVCC_FLAGS "--x cu")

message( STATUS "Using CUDA_NVCC_FLAGS=${CUDA_NVCC_FLAGS}")
