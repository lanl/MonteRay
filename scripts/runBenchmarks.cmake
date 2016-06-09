#######################################################################
# Build toolkit and run benchmark problems
#================================================================================

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR} )
include( PlatformInfo )

#######################################################################
# Parse argument list

ParseBuildArgs( ${CTEST_SCRIPT_ARG} )

#######################################################################
# Collect system info

PlatformInfo()

#######################################################################
# General Configuration
configureCTest( "Bench" )

# 1 hrs -  3600
# 2 hrs -  7200
# 3 hrs - 10800
# 4 hrs - 14400
# 8 hrs - 28800
set( CTEST_TEST_TIMEOUT "14400" )

set( CTEST_PROJECT_SUBPROJECTS ToolkitLib Benchmarks )

##############################################
# Run benchmark problems
#--------------------------------
set( CTEST_PROJECT_NAME Benchmarks )

set_property( GLOBAL PROPERTY SubProject Benchmarks )
set_property( GLOBAL PROPERTY Label      Benchmarks )

set( CTEST_SOURCE_DIRECTORY "${toolkitSrcDir}/Benchmarks")
set( CTEST_BINARY_DIRECTORY "${toolkitBuildDir}/Benchmarks")

#######################################################################
# START
#--------------------------------
ctest_start( ${Model} APPEND )

file  ( READ "${CTEST_BINARY_DIRECTORY}/Testing/TAG" tag_file )
string( REGEX MATCH "[^\n]*" BuildTag ${tag_file} )
set   ( TEST_OUTPUT_DIR "${CTEST_BINARY_DIRECTORY}/Testing/${BuildTag}" )
set   ( TEST_TEMP_DIR "${TEST_OUTPUT_DIR}/temp" )
file  ( MAKE_DIRECTORY ${TEST_TEMP_DIR} )

#######################################################################
# Test
# ---------------------------------------------------------------
ctest_test( APPEND INCLUDE_LABEL ${ProblemTag} )

configure_file( ${TEST_OUTPUT_DIR}/Test.xml ${TEST_TEMP_DIR}/Test${ProblemTag}.xml COPYONLY )
