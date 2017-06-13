include( GeneralFunctions )
checkBuildDirConsistency()

if( GNU_VER )
  set( CompilerExtension ${GNU_VER} CACHE STRING "Extension to target specific gnu variants." )
endif()

#set( CMAKE_C_COMPILER "gcc" )
#set( CMAKE_CXX_COMPILER "g++" )
#set( CMAKE_Fortran_COMPILER "gfortran" )

#project( MonteRay CXX )

include( VersionInfo )

find_path( isSubversionDir .svn ${mcatk_SOURCE_DIR} )

if( isSubversionDir )
    find_package( Subversion )
    if( NOT Subversion_FOUND )
        message( FATAL_ERROR "Subversion components were found but a svn executable was not." )
    endif()
    Subversion_WC_INFO( ${mcatk_SOURCE_DIR} mcatk )
    message( "Last changed revision is ${mcatk_WC_REVISION}" )
    set( RevisionName "r${mcatk_WC_REVISION}" CACHE STRING "Root directory name for release products" FORCE )
else()
    set( RevisionName "rUNKNOWN" CACHE STRING "Root directory name for release products" )
endif()

include( LocatePackagesDir )
if( OverwritingPublicRelease )
    message( FATAL_ERROR "**!!!Release ${ReleaseName} of version ${ToolkitVersion} already exists.  Please update the version number in main.cmake." )
endif()

if( needVerboseVar )
 include( Echo_CMake_Variables )
 echo_all_cmake_variable_values()
endif()

include( ToolChain_${CMAKE_CXX_COMPILER_ID} )
message( "*** Building on ${CMAKE_SYSTEM_NAME} with ${CMAKE_CXX_COMPILER_ID}'s toolchain" )

# Determine linker vendor
determine_linker_vendor()

# Check which C++11 features are available for this toolset
include( CheckCXX11Features )

# This next bit is necessary because Loki originally used auto_ptr
if( HAS_CXX11_UNIQUE_PTR )
    add_definitions( -DHAS_CXX11_UNIQUE_PTR )
endif()

# Set this until we're ready to move to signals2. NOTE: Intel WON'T support it til 15!
# intel issue -- https://software.intel.com/en-us/forums/topic/515966
add_definitions( -DBOOST_SIGNALS_NO_DEPRECATION_WARNING )

if( NOT CMAKE_BUILD_TYPE )
    set( CMAKE_BUILD_TYPE RelWithDebInfo 
         CACHE 
         STRING 
         "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." 
         FORCE )
endif()
if( WITH_CODE_COVERAGE )
    set( CMAKE_BUILD_TYPE Debug 
         CACHE 
         STRING 
         "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." 
         FORCE )
         
    set( CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${CodeCoverage_${CMAKE_CXX_COMPILER_ID}}" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CodeCoverage_${CMAKE_CXX_COMPILER_ID}}" )
    set( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS}  ${CodeCoverage_${CMAKE_CXX_COMPILER_ID}}" )
endif()

include( package_release )
include( documentation )
generate_doc_rule( Toolkit Doxyfile.in     Docs )

concatenateInstallSubdirs()

if( CMAKE_BUILD_TYPE STREQUAL "Debug" )
  add_definitions( -DBOOST_ENABLE_ASSERT_HANDLER -DDEBUG)
  message( "Build Type = ${CMAKE_BUILD_TYPE}; Enabling BOOST_ASSERT; Enabling Debugging" )
else()
  if( CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" ) 
    add_definitions( -DBOOST_ENABLE_ASSERT_HANDLER -DNDEBUG )
    message( "Build Type = ${CMAKE_BUILD_TYPE}; Enabling BOOST_ASSERT; Disabling Debugging" )
   else()
     add_definitions( -DBOOST_DISABLE_ASSERTS -DNDEBUG )
     message( "Build Type = ${CMAKE_BUILD_TYPE}; Disabling BOOST_ASSERT; Disabling Debugging" )
  endif()
endif()

# Special Build Flags
# <><><><><><><><><><><><><><><><>
# Try to address situations where the distances to event are nearly (or exactly) identical
# Files Affected: MinCombiner.hh stream_test.cc
# add_definitions( -DHANDLE_NEARLY_SIMULTANEOUS_EVENTS ) 

############################################
#  Enable testing with ctest
include( CTest )
enable_testing()

if( ENABLE_FT )
  include( ftmpi )
endif()

#include( NDATK )
#include( Boost )
#include( tbb )
#include( loki )
include( unittest )

############################################
#  List the directory names that only contain testing source
set( TestDirNames unit_test punit_test pnightly nightly fi_test pfi_test)

if( UNIX )
#    install( CODE "execute_process( COMMAND chgrp -R mcatk ${CMAKE_INSTALL_PREFIX} )" )
#    install( CODE "execute_process( COMMAND chmod -R g+rwX ${CMAKE_INSTALL_PREFIX} )" )
#    install( CODE "execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink ${Boost_INCLUDE_DIRS}/boost ${CMAKE_INSTALL_PREFIX}/include/boost )" )
#    install( CODE "execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink ${Loki_INCLUDE_DIRS}/loki ${CMAKE_INSTALL_PREFIX}/include/loki )" )
endif()

# Get the compile flags for mcatk-config
get_directory_property(compile_defs COMPILE_DEFINITIONS)
set(_build_type ${CMAKE_BUILD_TYPE})
string(TOUPPER "${_build_type}" _build_type)
set(MCATK_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_build_type}}")
set(MCATK_DEFS "${compile_defs}")
if(MCATK_FLAGS)
    list(REMOVE_DUPLICATES MCATK_FLAGS)
endif()
if(MCATK_DEFS)
    list(REMOVE_DUPLICATES MCATK_DEFS)
endif()
unset(_build_type)

