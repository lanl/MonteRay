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
        ctest_update( RETURN_VALUE update_result )
        if( update_result EQUAL -1 )
            message( FATAL_ERROR "Repository update error: Server not responding" )
        endif()
    endif()
    
    set( config_options 
             -DBatchMode:BOOL=ON
             -DCMAKE_BUILD_TYPE:STRING=${CTEST_BUILD_CONFIGURATION}
        )

    ctest_configure( OPTIONS "${config_options}"
                    )

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
        message( "NIGHTLY : Begin serial tests.  Current Elapsed:  ${CTEST_ELAPSED_TIME}" )

        ctest_test( INCLUDE_LABEL "SerialNightly" PARALLEL_LEVEL ${nHostProcs} SCHEDULE_RANDOM on )
        configure_file( ${TEST_OUTPUT_DIR}/Test.xml ${TEST_TEMP_DIR}/TestSerialNightly.xml COPYONLY )
        
        if( ${CTEST_ELAPSED_TIME} GREATER ${MaxSafeTestTime} )
            set( CTEST_TEST_TIMEOUT "60" )
        endif()
            
        set( nParallelTaskOptions 9 8 4 3 2 1 )
        foreach( nParallelTasks ${nParallelTaskOptions} )
            
            ctest_test( INCLUDE_LABEL "Nt${nParallelTasks}" PARALLEL_LEVEL ${nHostProcs} SCHEDULE_RANDOM on )
            configure_file( ${TEST_OUTPUT_DIR}/Test.xml ${TEST_TEMP_DIR}/TestNt${nParallelTasks}.xml COPYONLY )

            if( ${CTEST_ELAPSED_TIME} GREATER ${MaxSafeTestTime} )
                set( CTEST_TEST_TIMEOUT "60" )
            endif()

        endforeach()
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
# Repository: Subversion
#--------------------------------
initializeSVN()

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

# Before doing a parallel build, determine how many processors we have
if( BatchSystem STREQUAL slurm )
    set( CTEST_BUILD_COMMAND "gmake -j 8 install" )
    set( nHostProcs 64 )
else()
    determineProcessorCount()
    if( nHostProcs GREATER 16 )
        set( nHostProcs 16 )
        message( WARNING "Limiting procs to [ ${nHostProcs} ]" )
    endif()
    set( CTEST_BUILD_COMMAND "gmake -j ${nHostProcs} install" )
endif()

set( CTEST_TEST_TIMEOUT "7200" )

set( CTEST_PROJECT_SUBPROJECTS ToolkitLib FlatAPI QtAPI )

##############################################
# Toolkit
#--------------------------------
BuildAndTest( ToolkitLib )

##############################################
# FORTRAN/C API
#--------------------------------
BuildAndTest( FlatAPI "API/Flat" )

##############################################
# Trolltech Qt-4.0 API
#--------------------------------
BuildAndTest( QtAPI "API/Qt4" )

if( HostDomain STREQUAL xdiv OR BatchSystem STREQUAL slurm )
    CollateAndPostResults()
endif()
