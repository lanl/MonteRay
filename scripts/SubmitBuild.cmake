list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR} )
include( PlatformInfo )
PlatformInfo()

if( NOT BatchSystem )
    message( FATAL_ERROR "This script is only meant for submitting automated builds" )
endif()

ParseBuildArgs( ${OPTS} )

set( JobName        "${Model}-${Tool}-${Build}" )
set( OutputFileNameRoot "${ClusterID}_${JobName}-$ENV{USER}" )

set( JobOutputDir  ${AutoBuildRoot}/Results )
set( JobOutputFile "${JobOutputDir}/${OutputFileNameRoot}_UCBT.out" )
# remove the current output file
execute_process( COMMAND ${CMAKE_COMMAND} -E remove -f ${JobOutputFile} )

find_file( AutoBuildScript 
           ctestrun.cmake
           PATHS ${CMAKE_CURRENT_LIST_DIR} )

getDateStamp( dateStamp )

set( ShellScriptName /tmp/mcatkSubmit/$ENV{USER}/${JobName}_${ClusterID}-${dateStamp}.tcsh )

if( BatchSystem STREQUAL MOAB )
    set( CTestScriptName ${AutoBuildScript} )
    if( NOT Nodes )
        set( Nodes 1 )
    endif()
    configure_file( ${CMAKE_CURRENT_LIST_DIR}/SubmitBuild_msub.tcsh.in ${ShellScriptName} @ONLY )
    message( "*** Building [ ${OPTS} ] : Submitting through msub" )
else()
    message( FATAL_ERROR "Only moab currently recognized." )
endif()

execute_process( COMMAND ${SubmitCmd} ${ShellScriptName} 
                 RESULT submitResult 
                 OUTPUT_VARIABLE myJobID
                 ERROR_VARIABLE moabError )

if( submitResult AND NOT submitResult EQUAL 0 )
    message( "Result : ${submitResult}" )
endif()
if( myJobID )
    string( REPLACE "\n" "" myJobID ${myJobID} )
    set( myJobIDOut "${JobOutputDir}/${OutputFileNameRoot}.info" )
    message( "*** Dispatch [ ${OPTS} ] : Successful.  Job ID: [ ${myJobID} ]\n    TTY: [ ${JobOutputFile} ]\n    Info: [ ${myJobIDOut} ]" )
    file( WRITE ${myJobIDOut} 
          "Job ID : [ ${myJobID} ]\n" 
          "To follow: tail -f ${JobOutputFile}\n" 
          "To cancel: mjobctl -c ${myJobID}\n" )
else()
    message( FATAL_ERROR "Unable to submit: [ ${moabError} ]" )
endif()

# -----------------------------------------------
# Wait for the build/test job to finish
#
include( MOABJobWait )
    
MOABJobWait( ${myJobID} )

# -----------------------------------------------
# Commit the results (from the front-end)
#
CollateAndPostResults()
