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

function( MOABJobWait JobID )

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
