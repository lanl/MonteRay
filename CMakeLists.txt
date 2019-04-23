# Cmake 3.8 is needed for proper CUDA support
CMAKE_MINIMUM_REQUIRED(VERSION 3.8 FATAL_ERROR)

message( STATUS " -- CMakeList.txt -- Configuring MonteRay ")

if( enable_cuda )
    message( "Base CmakeLists.txt - enabling CUDA." ) 
    message( STATUS "-- Building with CUDA.")
    message( STATUS "-- Loading config_nvcc.cmake")
    
    PROJECT(MonteRayProject LANGUAGES CXX CUDA )
    
    #include( cmake_files/config_nvcc.cmake )
else()
    message( STATUS "-- Building without CUDA.")
    enable_language(CXX)
    PROJECT(MonteRayProject LANGUAGES CXX )
endif()

SET(CMAKE_INCLUDE_CURRENT_DIR ON)

# TRA - These need to be used and tested in other places besides Shark?
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# if( POLICY CMP0022 )
#   cmake_policy( SET CMP0022 OLD )
# endif()

set( cmake_dir "${CMAKE_SOURCE_DIR}/cmake_files" )
list( APPEND CMAKE_MODULE_PATH ${cmake_dir} )


# This directory has a more robust script for identifying host platform
list( APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/scripts )

message( STATUS "-- Loading main.cmake")
include( main )

if( enable_cuda )
  # call after main so that MPI include directories are properly set
  include(config_nvcc )
endif()

set( SubProjectName "MonteRayLib" )

# Creates an uninstall script/target, i.e. "make uninstall"
message( STATUS "-- Loading CMakeUninstallTarget.cmake")
include(CMakeUninstallTarget)

add_subdirectory( src )

add_subdirectory( apps )

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




