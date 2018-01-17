################################################
#
# Common cmake stuff for the unit tests
#
################################################
if( NOT UnitTest_LIBRARIES )

set( ut_PathSuffixes ${Platform}/${compiler_install_prefix}
                     ${compiler_install_prefix}
                     ${CMAKE_CXX_COMPILER_ID}/${CMAKE_SYSTEM_NAME}
                     ${CMAKE_CXX_COMPILER_ID} )

set( _orig_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES} )
if( UNIX )
    # This changes the preferred extension to make .a's the first match
    set( CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES} )
    find_library( UnitTest_LIBRARIES 
                  NAMES UnitTest++ 
                  PATHS ${package_dir}/lib
                        $ENV{UNITTEST_ROOT}/lib
                        $ENV{UNITTEST_ROOT}/lib/UnitTest++
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
    message( FATAL_ERROR "ERROR: UnitTest++ library not found.  In: ${package_dir} under ${ut_PathSuffixes}" )
endif()

# Restore the original cmake list of library suffixs that will be searched
set( CMAKE_FIND_LIBRARY_SUFFIXES ${_orig_CMAKE_FIND_LIBRARY_SUFFIXES} )

find_path( UnitTest_INCLUDE_DIRS UnitTest++.h 
           PATHS ${package_dir}
                 $ENV{UNITTEST_ROOT}
           PATH_SUFFIXES include/UnitTest++ )

message( "Found UnitTest++ libraries in : ${UnitTest_LIBRARIES}" )
message( "Found UnitTest++ headers   in : ${UnitTest_INCLUDE_DIRS}" )

endif()
