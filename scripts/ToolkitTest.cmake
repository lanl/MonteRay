#================================================================================
# Function: TestProject( SubProjectName [directory location] )
function( TestProject SubProjectName )

    set_property( GLOBAL PROPERTY SubProject ${SubProjectName} )
    set_property( GLOBAL PROPERTY Label      ${SubProjectName} )

    if( ${ARGC} EQUAL 2 )
        set( CTEST_SOURCE_DIRECTORY "${toolkitSrcDir}/${ARGV1}")
        set( CTEST_BINARY_DIRECTORY "${toolkitBuildDir}/${ARGV1}")
    else()
        set( CTEST_SOURCE_DIRECTORY "${toolkitSrcDir}")
        set( CTEST_BINARY_DIRECTORY "${toolkitBuildDir}")
    endif()

    #######################################################################
    # START
    #--------------------------------
    ctest_start( ${Model} APPEND )

    file  ( READ "${CTEST_BINARY_DIRECTORY}/Testing/TAG" tag_file )
    message( "CTEST BinDir : [ ${CTEST_BINARY_DIRECTORY} ]" )
    string( REGEX MATCH "[^\n]*" BuildTag ${tag_file} )
    set   ( TEST_OUTPUT_DIR "${CTEST_BINARY_DIRECTORY}/Testing/${BuildTag}" )
    set   ( TEST_TEMP_DIR "${TEST_OUTPUT_DIR}/temp" )
    file  ( MAKE_DIRECTORY ${TEST_TEMP_DIR} )


    # turn on coverage (Bullseye if its available) for building. If the Bullseye binary directory is the
    # first in the path, then it's version of the compiler calls are already in place.  This just specifies
    # whether they are called directly or after Bullseye modifies the calling args.
    if( CoverageToggle )
        set(RES 1)
        execute_process(COMMAND ${CoverageToggle} -1 RESULT_VARIABLE RES)
        if(RES)
            message(FATAL_ERROR "Failed to enable Bullseye coverage system.  Could not run cov01 -1")
        endif()
    endif()

    # ---------------------------------------------------------------
    # Test the Quick tests
    ctest_test( APPEND INCLUDE_LABEL "SerialQuick" PARALLEL_LEVEL ${Nodes} SCHEDULE_RANDOM on )
    configure_file( ${TEST_OUTPUT_DIR}/Test.xml ${TEST_TEMP_DIR}/TestSerialQuick.xml COPYONLY )
    
    foreach( nProcs 1 2 3 4 )
        math( EXPR ParallelJobs "${Nodes} * ${nProcs}" )
        ctest_test( APPEND INCLUDE_LABEL "Qk${nProcs}" PARALLEL_LEVEL ${ParallelJobs} SCHEDULE_RANDOM on )
        configure_file( ${TEST_OUTPUT_DIR}/Test.xml ${TEST_TEMP_DIR}/TestQk${nProcs}.xml COPYONLY )
    endforeach()

    # ===============================================================
    # Optional Submissions
    # The following are nice, but not necessary so we have added guards to
    # limit the run time so that we have a chance at completion
    if( ${CTEST_ELAPSED_TIME} GREATER ${MaxSafeTestTime} )
        set( CTEST_TEST_TIMEOUT "60" )
    endif()

    # ---------------------------------------------------------------
    # Long running nightlies
    if( Model STREQUAL "Nightly" )
        message( "NIGHTLY : Begin serial tests.  Current Elapsed:  ${CTEST_ELAPSED_TIME}" )

        if( ${CTEST_ELAPSED_TIME} GREATER ${MaxSafeTestTime} )
            set( CTEST_TEST_TIMEOUT "60" )
        endif()
        
        math( EXPR ParallelJobs "${Nodes} * 8" )
        ctest_test( APPEND INCLUDE_LABEL "Nt8" PARALLEL_LEVEL ${ParallelJobs} SCHEDULE_RANDOM on )
        configure_file( ${TEST_OUTPUT_DIR}/Test.xml ${TEST_TEMP_DIR}/TestNt8.xml COPYONLY )

        if( ${CTEST_ELAPSED_TIME} GREATER ${MaxSafeTestTime} )
            set( CTEST_TEST_TIMEOUT "60" )
        endif()

        ctest_test( APPEND INCLUDE_LABEL "SerialNightly" PARALLEL_LEVEL ${Nodes} SCHEDULE_RANDOM on )
        configure_file( ${TEST_OUTPUT_DIR}/Test.xml ${TEST_TEMP_DIR}/TestSerialNightly.xml COPYONLY )
        
    endif()

    # ---------------------------------------------------------------
    # Coverage Analysis
    # Tool: BullsEye
    if( CoverageToggle )
        message( "Coverage : Current Elapsed:  ${CTEST_ELAPSED_TIME}" )
        # Compute the coverage results.  This MUST be run AFTER the tests
        ctest_coverage( APPEND LABELS ${SubProjectName}Src )
  
        # Turn off the Bullseye coverage analysis system.
        execute_process(COMMAND ${CoverageToggle} -0 RESULT_VARIABLE RES)
    endif()

    # ---------------------------------------------------------------
    # Memory Checking
    # Tool: valgrind
    if( WITH_MEMCHECK AND CTEST_MEMORYCHECK_COMMAND )
        message( "Memory Check : Begin valgrind tests.  Current Elapsed:  ${CTEST_ELAPSED_TIME}" )
    
        if( ${CTEST_ELAPSED_TIME} GREATER ${MaxSafeTestTime} )
            message( "Memory Check : Skipping -> All !!! Current Elapsed:  ${CTEST_ELAPSED_TIME}" )
            return()
        endif()
        
        # Valgrind the serial Quick tests using 8 way concurrent
        ctest_memcheck( APPEND EXCLUDE_LABEL "(Nightly|Parallel)" PARALLEL_LEVEL ${Nodes} SCHEDULE_RANDOM on )

        if( ${CTEST_ELAPSED_TIME} GREATER ${MaxSafeTestTime} )
            message( "Memory Check : Skipping -> MPI !!! Current Elapsed:  ${CTEST_ELAPSED_TIME}" )
            return()
        endif()
        
        # Test the parallel (mpirun) consecutively
        ctest_memcheck( APPEND EXCLUDE_LABEL "(Nightly|Serial)" PARALLEL_LEVEL ${Nodes} )

    endif()

endfunction()

#################################################################################
# Main Section
#================================================================================
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR} )
include( PlatformInfo )
PlatformInfo()

ParseBuildArgs()

configureCTest()

if( Build STREQUAL Debug AND Model STREQUAL Experimental )
    if( BatchSystem )
        # Work with valgrind on the big iron machines
        set( WITH_MEMCHECK true )
    else()
        # use Bullseye on trr or candycorn
        set( WITH_COVERAGE true )
    endif()
endif()

set( TenHours 36000 ) # 10 hr max build/test
set( TenMinutes 600 )
math( EXPR MaxSafeTestTime "${TenHours} - ${TenMinutes}" )

#######################################################################
# Memory Checking: VALGRIND
#--------------------------------
find_program( CTEST_MEMORYCHECK_COMMAND NAMES valgrind)
set( CTEST_MEMORYCHECK_COMMAND_OPTIONS   "${CTEST_MEMORYCHECK_COMMAND_OPTIONS} --trace-children=yes --track-origins=yes" )
set( CTEST_MEMORYCHECK_SUPPRESSIONS_FILE "${CTEST_SOURCE_DIRECTORY}/cmake_files/valgrind_suppressions.txt" )


# Before doing a parallel build, determine how many processors we have
determineProcessorCount()

set( Nodes 1 )
if( ENV{PBS_NUM_NODES} )
    set( Nodes $ENV{PBS_NUM_NODES} )
endif()

set( CTEST_TEST_TIMEOUT "7200" )

set( CTEST_PROJECT_SUBPROJECTS ToolkitLib FlatAPI QtAPI )

##############################################
# Toolkit
#--------------------------------
TestProject( ToolkitLib )

##############################################
# FORTRAN/C API
#--------------------------------
TestProject( FlatAPI "API/Flat" )

##############################################
# Trolltech Qt-4.0 API
#--------------------------------
TestProject( QtAPI "API/Qt4" )
