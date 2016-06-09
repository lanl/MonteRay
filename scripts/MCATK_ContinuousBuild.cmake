#================================================================================
# Function: BuildAndTest( SubProjectName [directory location] )
function( BuildAndTest SubProjectName )

    set_property( GLOBAL PROPERTY SubProject ${SubProjectName} )
    set_property( GLOBAL PROPERTY Label      ${SubProjectName} )

    if( ${ARGC} EQUAL 2 )
        set(CTEST_SOURCE_DIRECTORY "${toolkitSrcDir}/${ARGV1}")
        set(CTEST_BINARY_DIRECTORY "${toolkitBuildDir}/${ARGV1}")
        
        ctest_start( Continuous )
    endif()

    file  ( READ "${CTEST_BINARY_DIRECTORY}/Testing/TAG" tag_file )
    string( REGEX MATCH "[^\n]*" BuildTag ${tag_file} )
    set   ( TEST_OUTPUT_DIR "${CTEST_BINARY_DIRECTORY}/Testing/${BuildTag}" )

    set( config_options 
             -DBatchMode:BOOL=ON
             -DCMAKE_BUILD_TYPE:STRING=${CTEST_BUILD_CONFIGURATION}
        )

    ctest_configure( OPTIONS "${config_options}" )

    ctest_build( TARGET "install" )

    # Test the Quick tests using 8 way concurrent
    ctest_test( INCLUDE_LABEL "Quick" PARALLEL_LEVEL 8 SCHEDULE_RANDOM on )
    
    # Process the Test.xml through our filter to coalesce and remove non-utf characters
    file( RENAME ${TEST_OUTPUT_DIR}/Test.xml ${TEST_OUTPUT_DIR}/TestQuick.xml )
    execute_process( COMMAND ${MergeScript} WORKING_DIRECTORY ${TEST_OUTPUT_DIR} )
    
    # Post the results to the dashboard server
    ctest_submit()
    
endfunction()

#######################################################################
# Main Section
#======================================================================
# Load any required functions
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR} )
include( PlatformInfo )
include( SyncRepository )
include( pushChanges )

#######################################################################
# Parse argument list
# at MOST the user can specify whether this happens on a branch
ParseBuildArgs( "Continuous gnu Release ${CTEST_SCRIPT_ARG}" )

#######################################################################
# Collect system info

PlatformInfo()

#######################################################################
# General Configuration
configureCTest()

# Guarantee the binary directory is deleted before building
ctest_empty_binary_directory( ${CTEST_BINARY_DIRECTORY} )

#######################################################################
# Repository: Subversion
#--------------------------------
initializeSVN()

set( SyncClusterID candycorn CACHE STRING "The sole system responsible for insuring all mirrors are consistent." )
if( ClusterID STREQUAL SyncClusterID )
    set( SVN ${Subversion_SVN_EXECUTABLE} )
    set( repoRoot ${MCATK_Repository} )
    set( mirrorRoot /usr/projects/mcatk/mirror )
    
    initializeMirrorSync()
    initializePush()
else()
    message( STATUS "Cluster [ ${ClusterID} ] will NOT attempt repository synchronization." )
endif()

#######################################################################
#  Specific CTest Options
set( CTEST_BUILD_COMMAND "gmake -j 8 install" )

# Limit is short since only the quick tests are a part of a continuous build
set( CTEST_TEST_TIMEOUT "60" )

set( CTEST_PROJECT_SUBPROJECTS ToolkitLib FlatAPI QtAPI )

set( SleepTime_sec 50 )
set( TenHours 36000 )
find_file( MergeScript 
           MergeTestResults.py 
           PATHS ${CMAKE_CURRENT_LIST_DIR}
            )

#######################################################################
#  **** Run Continuous for 10 hrs ****
while (${CTEST_ELAPSED_TIME} LESS ${TenHours} )

    #################################
    # START
    #--------------------------------
    ctest_start( Continuous )

    file  ( READ "${CTEST_BINARY_DIRECTORY}/Testing/TAG" tag_file )
    string( REGEX MATCH "[^\n]*" BuildTag ${tag_file} )
    set   ( TEST_OUTPUT_DIR "${CTEST_BINARY_DIRECTORY}/Testing/${BuildTag}" )
    
    #################################
    # UPDATE
    #--------------------------------
    ctest_update( RETURN_VALUE update_result )

    # Prevent any more checkouts
    unset( CTEST_CHECKOUT_COMMAND )

    # if update fails we're done (for now)
    if( update_result EQUAL -1 )
        message( FATAL_ERROR "Repository update error: Server not responding" )
        break()
    endif()

    if( ${update_result} EQUAL 0 )
        # Remove this uninteresting build
        file( REMOVE_RECURSE ${TEST_OUTPUT_DIR} )
    
        # No files changed so sleep 'til something happens
        ctest_sleep( ${SleepTime_sec} )

    else()
        Subversion_WC_INFO( ${CTEST_SOURCE_DIRECTORY} mcatk )
        message( "Revision [ ${mcatk_WC_REVISION} ] contains [ ${update_result} ] changed file(s).  Starting continuous build..." )
        
        #################################
        # update mirror
        syncMirror()
          
        #################################
        # Synchronize with red
        pushChanges()
        
        # Begin a continous build submission
        ctest_submit( PARTS Update )
    
        #################################
        # Toolkit
        #--------------------------------
        BuildAndTest( ToolkitLib )
        
        #################################
        # FORTRAN/C API
        #--------------------------------
        BuildAndTest( FlatAPI "API/Flat" )
    
        #################################
        # Trolltech Qt-4.0 API
        #--------------------------------
        BuildAndTest( QtAPI "API/Qt4" )

    endif()

endwhile()
