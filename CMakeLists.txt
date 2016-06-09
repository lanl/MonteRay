CMAKE_MINIMUM_REQUIRED(VERSION 3.1d)

PROJECT(MonteRayProject)

SET(CMAKE_INCLUDE_CURRENT_DIR ON)

#set( cmake_dir "${CMAKE_SOURCE_DIR}/cmake_files" )
#list( APPEND CMAKE_MODULE_PATH ${cmake_dir} )
#set( cmake_dir "${CMAKE_SOURCE_DIR}/scripts" )
#list( APPEND CMAKE_MODULE_PATH ${cmake_dir} )

#message( STATUS "-- Invoking LocatePackagesDir")
#include( LocatePackagesDir )
#message( STATUS "-- Invoking concatenateInstallSubdirs()")
#include( GeneralFunctions )
#concatenateInstallSubdirs()

#include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/external )
#include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/src )

if( POLICY CMP0022 )
    cmake_policy( SET CMP0022 OLD )
endif()

set( cmake_dir "${CMAKE_SOURCE_DIR}/cmake_files" )
list( APPEND CMAKE_MODULE_PATH ${cmake_dir} )

include(config_nvcc )

# This directory has a more robust script for identifying host platform
list( APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/scripts )

message( STATUS "-- Loading main.cmake")
include( main )

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




