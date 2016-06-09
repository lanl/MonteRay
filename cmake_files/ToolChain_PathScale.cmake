# EkoPath PathScale compiler toolset
# Defines: __PATHSCALE__, __PATHCC__

if( PathScale_VERSION )
    return()
endif()

set( CMAKE_CXX_COMPILER pathCC )
set( CMAKE_C_COMPILER pathcc )
set( CMAKE_FORTRAN_COMPILER pathf95 )

execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version
    ERROR_VARIABLE PathScale_VERSION
    ERROR_STRIP_TRAILING_WHITESPACE)
string( REGEX MATCH "([0-9]+).([0-9]+).([0-9]+).([0-9]+)" PathScale_VERSION ${PathScale_VERSION} ) 
string( REGEX REPLACE "[\\.\\-]" ";" PathScale_VERSION ${PathScale_VERSION} )
list( GET PathScale_VERSION 0 PathScale_MAJOR_VERSION ) 
list( GET PathScale_VERSION 1 PathScale_MINOR_VERSION ) 
list( GET PathScale_VERSION 2 PathScale_REVISION_VERSION ) 

message(STATUS "CompilerID [ ${CMAKE_CXX_COMPILER_ID} ]  Version [ ${PathScale_MAJOR_VERSION}.${PathScale_MINOR_VERSION}-${PathScale_REVISION_VERSION} ]" )

#Needed for install releases. Has to be done in each indivdual toolchain.
set(compiler_install_prefix "pathscale-${PathScale_MAJOR_VERSION}.${PathScale_MINOR_VERSION}" CACHE STRING "Compiler Name Prefix used in naming the install directory")
message(STATUS "Compiler Install Prefix is [ ${compiler_install_prefix} ]" )

set( Boost_COMPILER "-pathscale" )
set( Boost_SUBDIR "/pathscale" )

# Flags unique to PathScale
# set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --c++11" )

