#================================================================================
# Function: BuildProject( SubProjectName [directory location] )
function( BuildProject SubProjectName )

    set_property( GLOBAL PROPERTY SubProject ${SubProjectName} )
    set_property( GLOBAL PROPERTY Label      ${SubProjectName} )

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
    message( "CTEST BinDir : [ ${CTEST_BINARY_DIRECTORY} ]" )
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

    ctest_configure( OPTIONS "${config_options}" )

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
    ctest_build( TARGET "install" )

    # ---------------------------------------------------------------
    # Coverage Analysis
    # Tool: BullsEye
    if( CoverageToggle )
        message( "Coverage : Current Elapsed:  ${CTEST_ELAPSED_TIME}" )
        # Turn off the Bullseye coverage analysis system.
        execute_process(COMMAND ${CoverageToggle} -0 RESULT_VARIABLE RES)
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

if( Build STREQUAL Debug AND Model STREQUAL Experimental )
    if( BatchSystem )
        # Work with valgrind on the big iron machines
        set( WITH_MEMCHECK true )
    else()
        # use Bullseye on trr or candycorn
        set( WITH_COVERAGE true )
    endif()
endif()

#######################################################################
# Repository: Subversion
#--------------------------------
initializeSVN()

# Before doing a parallel build, determine how many processors we have
determineProcessorCount()

if( nHostProcs GREATER 8 )
    set( nHostProcs 8 )
endif()

set( CTEST_BUILD_COMMAND "gmake -j ${nHostProcs} install" )

set( CTEST_TEST_TIMEOUT "7200" )

set( CTEST_PROJECT_SUBPROJECTS ToolkitLib FlatAPI QtAPI )

##############################################
# Toolkit
#--------------------------------
BuildProject( ToolkitLib )

##############################################
# FORTRAN/C API
#--------------------------------
BuildProject( FlatAPI "API/Flat" )

##############################################
# Trolltech Qt-4.0 API
#--------------------------------
BuildProject( QtAPI "API/Qt4" )

##############################################
# Testing
#   Since the testing requires more resources than may exists on the compilation nodes,
#   submit the testing through the batching queueing system.
#--------------------------------

find_file( TestScript SubmitTest.cmake 
           PATHS ${CMAKE_CURRENT_LIST_DIR} )
if( TestScript )
    execute_process( COMMAND ${CMAKE_COMMAND} 
                             -DModel=${Model}
                             -DTool=${Tool}
                             -DBuild=${Build}
                             -DBranchName=${BranchName}
                             -DtoolkitSrcDir=${toolkitSrcDir}
                             -DtoolkitBuildDir=${toolkitBuildDir}
                             -P ${TestScript} 
                     RESULT_VARIABLE ResultSubmitTest )
else()
    message( FATAL_ERROR "Unable to submit for testing." )
endif()
