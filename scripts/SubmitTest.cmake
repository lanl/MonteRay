# Collect system info
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR} )
include( PlatformInfo )
PlatformInfo()

if( NOT BatchSystem )
    message( FATAL_ERROR "This script is only meant for submitting automated builds" )
endif()

getDateStamp( dateStamp )

set( JobName        "${Model}-${Tool}-${Build}" )
set( Nodes 2 )
set( OutputFileNameRoot "${ClusterID}_${JobName}-$ENV{USER}" )

set( JobOutputDir  ${AutoBuildRoot}/Results )
set( JobOutputFile "${JobOutputDir}/${OutputFileNameRoot}_TEST.out" )

# remove the current output file
execute_process( COMMAND ${CMAKE_COMMAND} -E remove -f ${JobOutputFile} )

set( AutoTestScriptName ToolkitTest.cmake )
find_file( TestingScript
           ${AutoTestScriptName} 
           PATHS ${CMAKE_CURRENT_LIST_DIR} )
set( CTestScriptName ${TestingScript} )

# set the options into a single argument
set( ScriptDefines "-DModel=${Model} -DTool=${Tool} -DBuild=${Build} -DBranchName=${BranchName}" )

set( ShellScriptName /tmp/mcatkSubmit/$ENV{USER}/${JobName}_${ClusterID}-${dateStamp}.tcsh )
message( "*** Dispatch [ TESTING ] : Submitting through msub" )
configure_file( ${CMAKE_CURRENT_LIST_DIR}/SubmitBuild_msub.tcsh.in ${ShellScriptName} @ONLY )

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
    message( "*** Dispatch [ TESTING ] : Successful.  Job ID: [ ${myJobID} ]\n    TTY: [ ${JobOutputFile} ]\n    Info: [ ${myJobIDOut} ]" )
    file( WRITE ${myJobIDOut} 
          "Job ID : [ ${myJobID} ]\n" 
          "To follow: tail -f ${JobOutputFile}\n" 
          "To cancel: mjobctl -c ${myJobID}\n" )
else()
    message( "Unable to submit: [ ${moabError} ]" )
endif()

include( MOABJobWait )
MOABJobWait( ${myJobID} )

# -----------------------------------------------
# Commit the results (from the front-end)
#
CollateAndPostResults()
