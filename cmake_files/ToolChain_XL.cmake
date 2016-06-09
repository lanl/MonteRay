# xlC is Visual Age compiler from IBM

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qlanglvl=extended0x -qpic -qnostaticlink -qmaxmem=-1" )

#==============================================================================
# Get XL Version
#
execute_process( COMMAND ${CMAKE_CXX_COMPILER} -qversion
                 OUTPUT_VARIABLE XL_CXX_VERSION
                 OUTPUT_STRIP_TRAILING_WHITESPACE )
                 
string( REGEX MATCH "[0-9]+\\.[0-9]+" XL_CXX_VERSION ${XL_CXX_VERSION} )

string( REGEX REPLACE "\\." ";" XL_CXX_VERSION ${XL_CXX_VERSION} )

list( GET XL_CXX_VERSION 0 XL_MAJOR_VERSION )
list( GET XL_CXX_VERSION 1 XL_MINOR_VERSION )
#list( GET XL_CXX_VERSION 2 XL_REVISION_VERSION )
message( STATUS "XL Version is [ ${XL_MAJOR_VERSION}.${XL_MINOR_VERSION} ]" )

#Needed for install releases. Has to be done in each indivdual toolchain.
set(compiler_install_prefix "xl-${XL_MAJOR_VERSION}.${XL_MINOR_VERSION}" CACHE STRING "Compiler Name Prefix used in naming the install directory")
message(STATUS "Compiler Install Prefix is [ ${compiler_install_prefix} ]" )

set( Boost_COMPILER "-xlc" )
# This makes 1.50 the preferred version regardless of others present
set( Boost_ADDITIONAL_VERSIONS "1.50.0" "1.50" )

add_definitions( -DENUM_PROBLEMS )

set( CXX_TOOL XLC )

# To look at all compiler flags set...
# xlc++   -qshowmacros -E /dev/null
find_library( mcatk_COMPILER_LIBRARY
              NAMES ibmc++
              PATHS /opt/ibmcmp/vacpp/bg/12.1/lib64 
              NO_DEFAULT_PATH )
if( NOT mcatk_COMPILER_LIBRARY )
    message( FATAL_ERROR "Unable to locate IBM's c++ library." )
endif()
get_filename_component( compiler_libdir ${mcatk_COMPILER_LIBRARY} PATH )
get_filename_component( mcatk_COMPILER_LIBRARY_DIR ${compiler_libdir} ABSOLUTE CACHE )

if( Platform STREQUAL "BlueGeneQ" )
    set( CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,muldefs" )
endif()

# Suppress spurious warnings
#  0724 - integer type wrapping in boost/type_traits
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qsuppress=1540-0724" )

find_program( MPI_C_COMPILER   mpixlc_r )
find_program( MPI_CXX_COMPILER mpixlcxx_r )
