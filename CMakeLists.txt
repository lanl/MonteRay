# Cmake 3.8 is needed for proper CUDA support
CMAKE_MINIMUM_REQUIRED(VERSION 3.8)

if( enable_cuda )
    message( "Base CmakeLists.txt - enabling CUDA." ) 
    enable_language(CUDA)
#    set(CMAKE_CXX_IGNORE_EXTENSIONS cu )
#    set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu )
    PROJECT(MonteRayProject LANGUAGES CXX CUDA )
else()
    enable_language(CXX)
    PROJECT(MonteRayProject LANGUAGES CXX )
endif()

SET(CMAKE_INCLUDE_CURRENT_DIR ON)

if( POLICY CMP0022 )
    cmake_policy( SET CMP0022 OLD )
endif()

set( cmake_dir "${CMAKE_SOURCE_DIR}/cmake_files" )
list( APPEND CMAKE_MODULE_PATH ${cmake_dir} )


# This directory has a more robust script for identifying host platform
list( APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/scripts )

message( STATUS "-- Loading main.cmake")
include( main )

if( DEFINED enable_cuda )
  message( STATUS "-- Building with CUDA.")
  message( STATUS "-- Loading config_nvcc.cmake")
  include(config_nvcc )
else()
  message( STATUS "-- Building without CUDA.")
endif()

set( SubProjectName "MonteRayLib" )

# Creates an uninstall script/target, i.e. "make uninstall"
message( STATUS "-- Loading CMakeUninstallTarget.cmake")
include(CMakeUninstallTarget)

add_subdirectory( src )

message( STATUS "-- Loading BuildSubPackage.cmake")
include( BuildSubPackage )

set( CPACK_PACKAGE_NAME "MONTERAY" )
if( isSubversionDir )
    set( CPACK_PACKAGE_NAME "${CPACK_PACKAGE_NAME}_${RevisionName}" )
endif()
set( CPACK_GENERATOR "TBZ2" )

# Creates a sh based mcatk-config in the build tree
include(ConfigureConfigScript)
include(ConfigureEnvScript)

include( CPack )




