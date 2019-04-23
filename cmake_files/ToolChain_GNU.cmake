
execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
    OUTPUT_VARIABLE CXX_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)
string( REGEX REPLACE "\\." ";" CXX_VERSION ${CXX_VERSION} ) 
list( GET CXX_VERSION 0 GNU_MAJOR_VERSION ) 
list( GET CXX_VERSION 1 GNU_MINOR_VERSION ) 
list( GET CXX_VERSION 2 GNU_REVISION_VERSION ) 
message(STATUS "GCC version is ${GNU_MAJOR_VERSION}.${GNU_MINOR_VERSION}.${GNU_REVISION_VERSION}" )

#Needed for install releases. Has to be done in each indivdual toolchain.
set(compiler_install_prefix "gnu-${GNU_MAJOR_VERSION}.${GNU_MINOR_VERSION}" CACHE STRING "Compiler Name Prefix used in naming the install directory")
message(STATUS "Compiler Install Prefix is [ ${compiler_install_prefix} ]" )

set( CodeCoverage_GNU "-fprofile-arcs -ftest-coverage" )

if( CMAKE_SYSTEM_NAME STREQUAL "Darwin" )
    set( GNU_DYNAMIC_LIB libstdc++.dylib )
else()
    set( GNU_DYNAMIC_LIB libstdc++.so )
endif()
message( "Searching for gnu library : [ ${GNU_DYNAMIC_LIB} ]" )

execute_process( COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=${GNU_DYNAMIC_LIB}
                 OUTPUT_VARIABLE WhereIsStd
                 OUTPUT_STRIP_TRAILING_WHITESPACE )
get_filename_component( WhereIsStd ${WhereIsStd} PATH )
get_filename_component( mcatk_COMPILER_LIBRARY_DIR ${WhereIsStd} ABSOLUTE CACHE )
    
message( STATUS "Found libstdc++.so in [ ${mcatk_COMPILER_LIBRARY_DIR} ]" )

#string( FIND ENV{LD_LIBRARY_PATH} ${mcatk_COMPILER_LIBRARY_DIR} isLDOK )
#message( "Compiler library path was found in LD_LIBRARY_PATH at position [ ${isLDOK} ] among [ $ENV{LD_LIBRARY_PATH} ]" )

# To look at all compiler flags set...
# g++ -dM -E -x c++ /dev/null

if( GNU_MAJOR_VERSION EQUAL 4 )
    if( GNU_MINOR_VERSION LESS 7 )
        message( FATAL_ERROR "MCATK requires a more recent version of compiler. g++ 4.7 or greater required." )
    endif()
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wno-terminate" )
endif()

set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og" CACHE STRING "newly available feature from gcc 4.8.x for faster debug runs" FORCE )

# GCC 5.X
if( GNU_MAJOR_VERSION EQUAL 5 )
    # This guarantees that MCATK will still link with other c++ libraries that might NOT have been built with gcc 5.0
    add_definitions( -D_GLIBCXX_USE_CXX11_ABI=1 )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++14 -Wno-terminate -Wno-deprecated-declarations" )
endif()

# GCC 6.X
if( GNU_MAJOR_VERSION EQUAL 6 )
    add_definitions( -D_GLIBCXX_USE_CXX11_ABI=1 )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-terminate -fpic -fPIC -std=c++14 -Wno-deprecated-declarations -Wno-placement-new" )
endif()

if( Platform STREQUAL "BlueGeneQ" )
#    set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--eh-frame-hdr" )
    set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -dynamic" )
endif()

set( COMPILER_CPP_FILE_FLAG "-x c++" )
