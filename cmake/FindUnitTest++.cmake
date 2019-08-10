#
# exports:
#
#   UnitTest++_FOUND
#   UnitTest++_INCLUDE_DIRS
#   UnitTest++_LIBRARIES
#

include(FindPkgConfig)

pkg_check_modules(PC_UnitTest++ QUIET unittest++)
find_path(UnitTest++_INCLUDE_DIR
  NAMES UnitTest++.h
  HINTS
    ${PC_UnitTest++_INCLUDE_DIRS}
    ${PC_UnitTest++_INCLUDEDIR}
    ${UnitTestPP_ROOT}
    $ENV{UnitTestPP_ROOT}
  PATH_SUFFIXES
    include
    include/unittest++
    include/UnitTest++)

find_library(UnitTest++_LIBRARY
  NAMES unittest++ UnitTest++
  HINTS
    ${PC_UnitTest++_LIBRARY_DIRS}
    ${PC_UnitTest++_LIBDIR}
    ${UnitTestPP_ROOT}
    $ENV{UnitTestPP_ROOT}
  PATH_SUFFIXES
    lib
    lib64
    lib/unittest++
    lib64/unittest++
    lib/UnitTest++
    lib64/UnitTest++)

find_package_handle_standard_args(UnitTest++
  DEFAULT_MSG
  UnitTest++_LIBRARY
  UnitTest++_INCLUDE_DIR)
  
mark_as_advanced(UnitTest++_INCLUDE_DIR UnitTest++_LIBRARY )

set(UnitTest++_LIBRARIES ${UnitTest++_LIBRARY})
set(UnitTest++_INCLUDE_DIRS ${UnitTest++_INCLUDE_DIR})
  

add_library(UnitTest++ INTERFACE IMPORTED)
set_property(TARGET UnitTest++ PROPERTY
  INTERFACE_LINK_LIBRARIES ${UnitTest++_LIBRARIES})
set_property(TARGET UnitTest++ PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${UnitTest++_INCLUDE_DIRS})
