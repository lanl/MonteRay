#================================================================================
# Function: BuildAndTest( SubProjectName [directory location] )
function( BuildAndTest SubProjectName )

    set_property( GLOBAL PROPERTY SubProject ${SubProjectName} )
    set_property( GLOBAL PROPERTY Label      ${SubProjectName} )

    if( ${ARGC} EQUAL 2 )
        set( CTEST_SOURCE_DIRECTORY "${toolkitSrcDir}/${ARGV1}")
        set( CTEST_BINARY_DIRECTORY "${toolkitBuildDir}/${ARGV1}")
    else()
        set( mainProject true )
        set( CTEST_SOURCE_DIRECTORY "${toolkitSrcDir}")
        set( CTEST_BINARY_DIRECTORY "${toolkitBuildDir}")
    endif()

    #######################################################################
    # START
    #--------------------------------
    message( STATUS "ctestrun.cmake::ctest_start() Model = [${Model}]" )
    ctest_start( ${Model} )

    file  ( READ "${CTEST_BINARY_DIRECTORY}/Testing/TAG" tag_file )
    string( REGEX MATCH "[^\n]*" BuildTag ${tag_file} )
    set   ( TEST_OUTPUT_DIR "${CTEST_BINARY_DIRECTORY}/Testing/${BuildTag}" )
    set   ( TEST_TEMP_DIR "${TEST_OUTPUT_DIR}/temp" )
    file  ( MAKE_DIRECTORY ${TEST_TEMP_DIR} )

    ##############################################
    # UPDATE
    #--------------------------------
    # only want 1 update file for now since everything points to same repository
    if( mainProject )
       if( NOT (CIType STREQUAL PullRequestCI) )
          ctest_update( RETURN_VALUE update_result)
          if( update_result EQUAL -1 )
             if( CIType STREQUAL HandCI )
                message( WARNING "ctestrun.cmake::ctest_update() - Repository update saw some possible issues. Look at (git pull) message output right before this message, or Server not responding" )
             else()
                message( FATAL_ERROR "Repository update error: Server not responding" )
             endif()
          endif()
       endif()
    endif()

    # option setting of UnityBuild in PlatformInfo.cmake
    if(UnityBuild)
       set(UnityBuildString "-DUnityBuild:BOOL=ON" )
    endif()

    set( config_options 
          -DBatchMode:BOOL=ON
          ${UnityBuildString}
          -DCMAKE_BUILD_TYPE:STRING=${CTEST_BUILD_CONFIGURATION}
    )    
    message(STATUS "ctestrun.cmake:BuildAndTest has config_options = [${config_options}]")
    ctest_configure( OPTIONS "${config_options}" )

    if( CoverageToggle )
        startBullsEye()
    endif()

    # ---------------------------------------------------------------
    ctest_build( TARGET "install" )

    # ---------------------------------------------------------------
    # Test the serial Quick tests using 8 way concurrent
    ctest_test( INCLUDE_LABEL "Quick" PARALLEL_LEVEL ${nHostProcs} SCHEDULE_RANDOM on )
    configure_file( ${TEST_OUTPUT_DIR}/Test.xml ${TEST_TEMP_DIR}/TestQuick.xml COPYONLY )

    # ===============================================================
    # Optional Submissions
    # The following are nice, but not necessary so we have added guards to
    # limit the run time so that we have a chance at completion
    if( ${CTEST_ELAPSED_TIME} GREATER ${MaxSafeTestTime} )
        set( CTEST_TEST_TIMEOUT "60" )
    endif()

    # ---------------------------------------------------------------
    # Long running nightlies
    if( Model STREQUAL "Nightly" OR Model STREQUAL "ContinuousNightly" OR TestAll )
        message( "NIGHTLY : Begin nightly tests.  Current Elapsed:  ${CTEST_ELAPSED_TIME}" )

        ctest_test( INCLUDE_LABEL "Nightly" PARALLEL_LEVEL ${nHostProcs} SCHEDULE_RANDOM on )
        configure_file( ${TEST_OUTPUT_DIR}/Test.xml ${TEST_TEMP_DIR}/TestSerialNightly.xml COPYONLY )
        
    endif()

    # ---------------------------------------------------------------
    # Coverage Analysis
    # Tool: BullsEye
    if( CoverageToggle )
        message( "Coverage : Current Elapsed:  ${CTEST_ELAPSED_TIME}" )
        # Compute the coverage results.  This MUST be run AFTER the tests
        ctest_coverage( LABELS ${SubProjectName}Src )
  
        stopBullsEye()
    endif()

    # ---------------------------------------------------------------
    # Memory Checking
    # Tool: valgrind
    if( WITH_MEMCHECK AND CTEST_MEMORYCHECK_COMMAND )
        message( "Memory Check : Begin valgrind tests.  Current Elapsed:  ${CTEST_ELAPSED_TIME}" )
    
        # Valgrind the serial Quick tests using 8 way concurrent
        ctest_memcheck( EXCLUDE_LABEL "(Nightly|Parallel)" PARALLEL_LEVEL ${nHostProcs} SCHEDULE_RANDOM on )

        # Test the parallel (mpirun) consecutively
        ctest_memcheck( EXCLUDE_LABEL "(Nightly|Serial)" PARALLEL_LEVEL 2 )

    endif()

endfunction()

#######################################################################
# Main Section
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
configureCTest()

# Guarantee the binary directory is deleted before building
ctest_empty_binary_directory( ${toolkitBuildDir} )

set( TenHours 36000 ) # 10 hr max build/test
set( TenMinutes 600 )
math( EXPR MaxSafeTestTime "${TenHours} - ${TenMinutes}" )

#######################################################################
# Repository
#--------------------------------
if( NOT (CIType STREQUAL PullRequestCI) ) 
  include( UseGit )
  initializeRepository()

  if( CTEST_CHECKOUT_COMMAND AND CIType )
     message( STATUS "ctestrun.cmake - CTEST_CHECKOUT_COMMAND  = [${CTEST_CHECKOUT_COMMAND}]" )
  endif()
  if( CTEST_GIT_UPDATE_CUSTOM AND CIType )
     message( STATUS "ctestrun.cmake - CTEST_GIT_UPDATE_CUSTOM = [${CTEST_GIT_UPDATE_CUSTOM}]" )
  endif()
endif()

#######################################################################
# Memory Checking: valgrind
#--------------------------------
if( WITH_MEMCHECK )
    initializeMemoryChecking()
endif()

#######################################################################
# Code Coverage: BullsEye
#--------------------------------
if( WITH_COVERAGE )
    setupCodeCoverage()
endif()

find_program( Builder gmake )
find_program( Builder make )
if( NOT Builder )
    message( FATAL_ERROR "Unable to find a typical build tool (i.e. gmake, make, etc)" )
endif()

# Before doing a parallel build, determine how many processors we have
if( BatchSystem STREQUAL slurm AND HostDomain STREQUAL llnl )
    set( CTEST_BUILD_COMMAND "${Builder} -j 8 install" )
    set( nHostProcs 64 )
else()
    determineProcessorCount()
    if( nHostProcs GREATER 16 )
        #For ADX and HPC machines.....
        math( EXPR nBuildProcs "${nHostProcs}/2" )
        message( WARNING "Limiting build procs to [ ${nBuildProcs} ]" )
    elseif( nHostProcs GREATER 4 )
        # MCATK ADX machines .......
        math( EXPR nBuildProcs "${nHostProcs}-2" )
        message( WARNING "Limiting build procs to [ ${nBuildProcs} ]" )
    else()
        set( nBuildProcs ${nHostProcs} )
    endif()
    
    # if the environment variable is GNUMAKE_J is set use that value
    if( DEFINED ENV{GNUMAKE_J} )
        set( GNUMAKE_J $ENV{GNUMAKE_J} )
    else()
        set( GNUMAKE_J ${nBuildProcs} )
    endif()
    set( CTEST_BUILD_COMMAND "${Builder} -j ${GNUMAKE_J} install" )

endif()

set( CTEST_TEST_TIMEOUT "7200" )

set( CTEST_PROJECT_SUBPROJECTS ToolkitLib FlatAPI QtAPI )

##############################################
# Toolkit
#--------------------------------
BuildAndTest( ToolkitLib )

#disable further checkouts
unset( CTEST_CHECKOUT_COMMAND )

##############################################
# Trolltech Qt-4.0 API
#--------------------------------
#BuildAndTest( QtAPI "API/Qt4" )

if( HostDomain STREQUAL xdiv OR BatchSystem STREQUAL slurm )
    CollateAndPostResults()
endif()
