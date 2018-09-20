################################################
#
# Common cmake stuff for the unit tests
#
################################################
if( NOT UnitTest_LIBRARIES )
message( "Locating UnitTest++ -- unittest.cmake" )

set( ut_PathSuffixes ${Platform}/${compiler_install_prefix}
                     ${compiler_install_prefix}
                     ${CMAKE_CXX_COMPILER_ID}/${CMAKE_SYSTEM_NAME}
                     ${CMAKE_CXX_COMPILER_ID} 
                     UnitTest++
   )

set( _orig_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES} )

if(DEFINED ENV{UNITTEST_ROOT}) 
    set( UNITTEST_ROOT $ENV{UNITTEST_ROOT} )
    message( "-- unittest.cmake -- UnitTest++ root directoy set by environment variable. UNITTEST_ROOT = [ ${UNITTEST_ROOT} ]" )
else()
    #TODO: TRA
    set( UNITTEST_ROOT ${package_dir} )
    message( "--xx unittest.cmake -- UnitTest++ root directoy set to by default to package_dir. UNITTEST_ROOT = [ ${UNITTEST_ROOT} ]" )
endif()

if( UNIX )
    # This changes the preferred extension to make .a's the first match
    set( CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES} )
    find_library( UnitTest_LIBRARIES 
                  NAMES UnitTest++ 
                  PATHS ${UNITTEST_ROOT}/lib
                  PATH_SUFFIXES ${ut_PathSuffixes}
                  DOC "The UnitTest++ library for System: ${CMAKE_SYSTEM_NAME} Compiler: ${CMAKE_CXX_COMPILER}"
                  NO_DEFAULT_PATH )
    # If this isn't erased, it will still appear in the link line as -L.. and in the rpath!!!
    unset( UnitTest_LIBRARY_DIRS )
endif()
if( WIN32 )
    set( CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES} )
    find_library( UnitTest_LIBRARIES libUnitTest++ PATHS ${UnitTest_LIBRARY_DIRS}/UnitTest++/lib/${CMAKE_CXX_COMPILER_ID} )
endif()

if( NOT UnitTest_LIBRARIES )
    if(NOT DEFINED ENV{UNITTEST_ROOT}) 
        message( STATUS "-- unittest.cmake -- ERROR:  UnitTest++ library not found.  Consider setting the UNITTEST_ROOT environment variable." )
    endif()
    message( FATAL_ERROR "-- unittest.cmake -- ERROR: UnitTest++ library not found. Searched in: ${UNITTEST_ROOT}/lib under ${ut_PathSuffixes}" )
else()
    
endif()

# Restore the original cmake list of library suffixs that will be searched
set( CMAKE_FIND_LIBRARY_SUFFIXES ${_orig_CMAKE_FIND_LIBRARY_SUFFIXES} )

set( ut_include_PathSuffixes UnitTest++ )

find_path( UnitTest_INCLUDE_DIRS UnitTest++.h 
           PATHS ${UNITTEST_ROOT}/include
           PATH_SUFFIXES ${ut_include_PathSuffixes} 
         )

if( NOT UnitTest_INCLUDE_DIRS )
    if(NOT DEFINED ENV{UNITTEST_ROOT}) 
        message( STATUS "-- unittest.cmake -- ERROR: UnitTest++ include path not found.  Consider setting the UNITTEST_ROOT environment variable." )
    endif()
    message( FATAL_ERROR "-- unittest.cmake -- ERROR: UnitTest++ include path not found. Searched in: ${UNITTEST_ROOT}/include under ${ut_include_PathSuffixes}" )
endif()    

message( "-- unittest.cmake -- Found UnitTest++ libraries in : ${UnitTest_LIBRARIES}" )
message( "-- unittest.cmake -- Found UnitTest++ include headers in : ${UnitTest_INCLUDE_DIRS}" )

endif()
