set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x" )

execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
    OUTPUT_VARIABLE CXX_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)
string( REGEX REPLACE "\\." ";" CXX_VERSION ${CXX_VERSION} ) 
list( GET CXX_VERSION 0 Intel_MAJOR_VERSION ) 
list( GET CXX_VERSION 1 Intel_MINOR_VERSION ) 
#list( GET CXX_VERSION 2 Intel_REVISION_VERSION ) 
message(STATUS "Intel version is ${Intel_MAJOR_VERSION}.${Intel_MINOR_VERSION}" )

#Needed for install releases. Has to be done in each indivdual toolchain.
set(compiler_install_prefix "intel-${Intel_MAJOR_VERSION}.${Intel_MINOR_VERSION}" CACHE STRING "Compiler Name Prefix used in naming the install directory")
message(STATUS "Compiler Install Prefix is [ ${compiler_install_prefix} ]" )

set( CodeCoverage_Intel "-prof-gen=srcpos -prof-dir=${CMAKE_BINARY_DIR}" )

# Remark
# 8290,8291 - non-recommended relationship between format width and requested number of digits
# 3373 nonstandard use of "auto"
# Warning
# 5462 - Global name too long
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fp-model source -diag-disable 279,2928,3373 -we1101" )
set( CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fp-model source -diag-disable 8290,8291,5462" )
#list( APPEND CMAKE_CXX_FLAGS "-fp-model precise" ) # numerically reproducible

#set( CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -stand f03" )

execute_process( COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so
                 OUTPUT_VARIABLE WhereIsStd
                 OUTPUT_STRIP_TRAILING_WHITESPACE )
get_filename_component( WhereIsStd ${WhereIsStd} PATH )
get_filename_component( mcatk_COMPILER_LIBRARY_DIR ${WhereIsStd} ABSOLUTE CACHE )
    
message( STATUS "Found libstdc++.so in [ ${mcatk_COMPILER_LIBRARY_DIR} ]" )

# To look at all compiler flags set...
# icpc -dM -E -x c++ /dev/null
