#=================================================================================
#  __  __  ____          ____       _       _ __          __   _ _   
# |  \/  |/ __ \   /\   |  _ \     | |     | |\ \        / /  (_) |  
# | \  / | |  | | /  \  | |_) |    | | ___ | |_\ \  /\  / /_ _ _| |_ 
# | |\/| | |  | |/ /\ \ |  _ < _   | |/ _ \| '_ \ \/  \/ / _` | | __|
# | |  | | |__| / ____ \| |_) | |__| | (_) | |_) \  /\  / (_| | | |_ 
# |_|  |_|\____/_/    \_\____/ \____/ \___/|_.__/ \/  \/ \__,_|_|\__|
#                                                                    
#                                                                  
#=================================================================================
# find summary of all past jobs
# host> showq -c -w user=sdnolen

function( JobWait JobID )

    find_program( SLEEP sleep )

    set( waitTime 300 )

    if( ${ARGC} GREATER 1 )
        set( waitTime ${ARGV1} )
    endif()

    # 72000 = 20hr  But we want to be done before the next day's commits start.
    math( EXPR NCalls "72000 / ${waitTime}" )

    find_program( CHECKJOB checkjob )

    set( StatusFile /tmp/JobStatus-${JobID} )

    message( "Waiting on MOAB job ID [ ${JobID} ]" )

    foreach( i RANGE ${NCalls} )

        execute_process( COMMAND ${CHECKJOB} ${JobID} OUTPUT_FILE ${StatusFile} )
        file( STRINGS ${StatusFile} Status REGEX "State: .*" )
        if( Status )
            string( REGEX REPLACE "State: ([A-Za-z]+) " "\\1" STATE ${Status} ) 
        else()
            # checkjob at LLNL can return nothing :(
            set( STATE "Indeterminate" )
        endif()
        # Check if job has been ended successfully.  'Removed' means it just exceeded its
        # walltime which may mean the results are still worth reporting
        string( REGEX MATCH "(Completed|Removed)" jobSuccess ${STATE} )
        if( jobSuccess )
            return()
        endif()

        # Check if job has been ended by a system failure
        string( REGEX MATCH "(Suspended|Vacated)" jobFailed ${STATE} )
        if( jobFailed )
            message( FATAL_ERROR "Job [ ${JobID} ] was unexpectedly terminated." )
        endif()

        # Job state is either running or indeterminate so check again in waitTime [sec]
        execute_process( COMMAND ${SLEEP} ${waitTime} )

    endforeach()

    message( FATAL_ERROR "Problem with Job [ ${JobID} ]. Last reported state [ ${STATE} ]." )

endfunction()

#######################################################################
#  __  __  ____          ____       _       _     _____       _               _ _   
# |  \/  |/ __ \   /\   |  _ \     | |     | |   / ____|     | |             (_) |  
# | \  / | |  | | /  \  | |_) |    | | ___ | |__| (___  _   _| |__  _ __ ___  _| |_ 
# | |\/| | |  | |/ /\ \ |  _ < _   | |/ _ \| '_ \\___ \| | | | '_ \| '_ ` _ \| | __|
# | |  | | |__| / ____ \| |_) | |__| | (_) | |_) |___) | |_| | |_) | | | | | | | |_ 
# |_|  |_|\____/_/    \_\____/ \____/ \___/|_.__/_____/ \__,_|_.__/|_| |_| |_|_|\__|
#                                                                                   
# ---------------------------------------------------------------
function( JobSubmit Tag ScriptName )

    getDateStamp( dateStamp )
    
    set( JobName        "${Model}-${Tool}-${Build}" )
    set( OutputFileNameRoot "${ClusterID}${Tag}_${JobName}-$ENV{USER}" )
    
    set( JobOutputDir  ${AutoBuildRoot}/Results )
    set( JobOutputFile "${JobOutputDir}/${OutputFileNameRoot}.out" )
    
    # remove the current output file
    execute_process( COMMAND ${CMAKE_COMMAND} -E remove -f ${JobOutputFile} )
    
    find_file( TestingScript
               ${ScriptName} 
               PATHS ${CMAKE_CURRENT_LIST_DIR} )
    if( NOT TestingScript )
        message( FATAL_ERROR "Unable to locate script [${ScriptName} ]" )
    endif()
    set( CTestScriptName ${TestingScript} )
    
    set( ShellScriptName /tmp/mcatkSubmit/$ENV{USER}/${JobName}_${ClusterID}${Tag}-${dateStamp}.tcsh )
    message( "*** Dispatch [ ${Tag} ] : Submitting through msub" )
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
        message( "*** Dispatch [ ${Tag} ] : Successful.  Job ID: [ ${myJobID} ]\n    TTY: [ ${JobOutputFile} ]\n    Info: [ ${myJobIDOut} ]" )
        file( WRITE ${myJobIDOut} 
              "Job ID : [ ${myJobID} ]\n" 
              "To follow: tail -f ${JobOutputFile}\n" 
              "To cancel: mjobctl -c ${myJobID}\n" )
        if( ActiveJobs )
            set( ActiveJobs "${ActiveJobs};${myJobID}" PARENT_SCOPE )
        else()
            set( ActiveJobs ${myJobID} PARENT_SCOPE )
        endif()
    else()
        message( "Unable to submit: [ ${moabError} ]" )
    endif()
        
endfunction()

#######################################################################
#  __  __  ____          ______          __   _ _            _ _ 
# |  \/  |/ __ \   /\   |  _ \ \        / /  (_) |     /\   | | |
# | \  / | |  | | /  \  | |_) \ \  /\  / /_ _ _| |_   /  \  | | |
# | |\/| | |  | |/ /\ \ |  _ < \ \/  \/ / _` | | __| / /\ \ | | |
# | |  | | |__| / ____ \| |_) | \  /\  / (_| | | |_ / ____ \| | |
# |_|  |_|\____/_/    \_\____/   \/  \/ \__,_|_|\__/_/    \_\_|_|
#                                                                
# ---------------------------------------------------------------
function( WaitAll )
    foreach( jobID ${ActiveJobs} )
        JobWait( ${jobID} )
    endforeach()
endfunction()

