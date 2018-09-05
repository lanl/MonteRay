###############################################################################
message( STATUS "Running Builder.cmake")

CMAKE_MINIMUM_REQUIRED(VERSION 3.7)

include( $ENV{PWD}/cmake_files/GeneralFunctions.cmake )

if( buildDir )
     get_filename_component( buildDirAbsolute ${buildDir} ABSOLUTE )
     set( ENV{BINARY_DIR} ${buildDirAbsolute} )
else()
     set( buildDir $ENV{BINARY_DIR} )
endif()
message( "Building in [ ${buildDir} ] from [ $ENV{PWD} ]" )
 
if( NOT EXISTS ${buildDir} )
    execute_process( COMMAND ${CMAKE_COMMAND} -E make_directory ${buildDir} )
endif()

string( REGEX MATCH "(gnu|intel|xl|clang)" toolset ${buildDir} )
configureToolset( ${toolset} )

if( BatchMode )
    set( BatchFlag "-DBatchMode=ON" )
endif()

if( GNU_VER )
    set( VersionFlag "-DGNU_VER=${GNU_VER}" )
endif()

if( Standalone )
    set( StandaloneFlag "-DStandalone=ON" )
endif()

if( enable_cuda )
    set( CudaFlag "-Denable_cuda=ON" )
endif()

if( InstallDir )
    set( InstallFlag "-DInstallDir=${InstallDir}" )
endif()

if( DISABLE_NDATK_MT71X ) 
	set( DISABLE_NDATK_MT71X_Flag "-DDISABLE_NDATK_MT71X=ON" )
endif()
if( DISABLE_NDATK_MENDF71X ) 
	set( DISABLE_NDATK_MENDF71X_Flag "-DDISABLE_NDATK_MENDF71X=ON" )
endif()
set( NDATK_Flags ${DISABLE_NDATK_MT71X_Flag} ${DISABLE_NDATK_MENDF71X_Flag} )
message ("NDATK_Flags = ${NDATK_Flags}")

# Set the build type Debug, Release, or RelWithDebInfo based on the
# name of the buildDir.
string( REGEX MATCH "(debug|db|Debug|DEBUG)" withDebug ${buildDir} )
if( withDebug )
  set( DEBUG_LEVEL_DEFAULT "2" )
  set( DebugOption "-DCMAKE_BUILD_TYPE=Debug" )
  set( CMAKE_BUILD_TYPE Debug )
else()
  string( REGEX MATCH "(release|Release|RELEASE)" withRelease ${buildDir} )
  if( withRelease )
    set( DEBUG_LEVEL_DEFAULT "0" )
    set( DebugOption "-DCMAKE_BUILD_TYPE=Release" )
    set( CMAKE_BUILD_TYPE Release )
  else()
    set( DEBUG_LEVEL_DEFAULT "1" )
    set( DebugOption "-DCMAKE_BUILD_TYPE=RelWithDebInfo" )
    set( CMAKE_BUILD_TYPE RelWithDebInfo )
  endif()
endif()
message( "Building with CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}" )

if( NOT DEBUG_LEVEL ) 
  set( DEBUG_LEVEL "-DDEBUG_LEVEL=${DEBUG_LEVEL_DEFAULT}" )
else()
  set( DEBUG_LEVEL_DEFAULT ${DEBUG_LEVEL} )
  set( DEBUG_LEVEL "-DDEBUG_LEVEL=${DEBUG_LEVEL}" )
endif()

message( "Building with CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}, Debug level=${DEBUG_LEVEL_DEFAULT} " )
set( DebugOptions ${DEBUG_LEVEL} ${DebugOption} )

message( "Executing COMMAND=${CMAKE_COMMAND} ${DebugOption} ${BatchFlag} ${UnityBuildFlag} ${VersionFlag} ${StandaloneFlag} ${CudaFlag} ${InstallFlag} ${FortranOffFlag} ${NDATK_Flags} ${MonteRayFlags} $ENV{PWD} in directory ${buildDir}" )
execute_process( COMMAND ${CMAKE_COMMAND} ${DebugOption} ${BatchFlag} ${UnityBuildFlag} ${VersionFlag} ${StandaloneFlag} ${CudaFlag} ${InstallFlag} ${FortranOffFlag} ${NDATK_Flags} ${MonteRayFlags} $ENV{PWD}
                 WORKING_DIRECTORY ${buildDir}
                 RESULT_VARIABLE result_var
                 )

#execute_process( 
#                 COMMAND chmod -R ug+rwX ${buildDir}
#                 COMMAND chmod -R g+s    ${buildDir}
#                 COMMAND chmod -R o-rwX  ${buildDir}
#                 COMMAND chmod ug+rwX ${buildDir}/..
#                 COMMAND chmod g+s    ${buildDir}/..
#                 COMMAND chmod o-rwX  ${buildDir}/..
#                 RESULT_VARIABLE result_var
#               )

#if( NOT result_var EQUAL 0 )
#    message( FATAL_ERROR "Builder.cmake:: Unable to set permissions on ${buildDir}/.." )
#endif() 

if( NOT Standalone )             
#  execute_process( 
#                   COMMAND chgrp -R mcatk  ${buildDir}
#                   COMMAND chgrp mcatk  ${buildDir}/.. 
#                   RESULT_VARIABLE chgrp_result_var
#                 )
                       
#  if( NOT chgrp_result_var EQUAL 0 )
#      message( FATAL_ERROR "Builder.cmake:: Unable to set permissions on ${buildDir}/.." )
#  endif()
endif()
