# Portland Group Toolset

if( PGI_VERSION )
    return()
endif()

set( CMAKE_CXX_COMPILER pgCC )
set( CMAKE_C_COMPILER pgcc )
set( CMAKE_FORTRAN_COMPILER pgfortran )

execute_process(COMMAND ${CMAKE_CXX_COMPILER} -V
    OUTPUT_VARIABLE PGI_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)
string( REGEX MATCH "([0-9]+).([0-9]+).([0-9]+)" PGI_VERSION ${PGI_VERSION} ) 
string( REGEX REPLACE "[\\.\\-]" ";" PGI_VERSION ${PGI_VERSION} )
list( GET PGI_VERSION 0 PGI_MAJOR_VERSION ) 
list( GET PGI_VERSION 1 PGI_MINOR_VERSION ) 
list( GET PGI_VERSION 2 PGI_REVISION_VERSION ) 

message(STATUS "CompilerID [ ${CMAKE_CXX_COMPILER_ID} ]  Version [ ${PathScale_MAJOR_VERSION}.${PathScale_MINOR_VERSION}-${PathScale_REVISION_VERSION} ]" )

#Needed for install releases. Has to be done in each indivdual toolchain.
set(compiler_install_prefix "pgi-${PGI_MAJOR_VERSION}.${PGI_MINOR_VERSION}" CACHE STRING "Compiler Name Prefix used in naming the install directory")
message(STATUS "Compiler Install Prefix is [ ${compiler_install_prefix} ]" )

set( Boost_COMPILER "-pgi" )
set( Boost_SUBDIR "/pgi" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --c++14" )

