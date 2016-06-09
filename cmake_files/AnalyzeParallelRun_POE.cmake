################################################################################
# Analyze results of unit test runs

################################################################################
get_filename_component( TestDirName ${CMAKE_CURRENT_BINARY_DIR} NAME )

set( CmdTag "Testing ${UNIT}[ ${TestDirName}.MPI${NPROCS} ]..." )
set( FailTag "${CmdTag} FAIL" )

#execute_process( COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --blue --bold ${CmdTag} )
set( ENV{MP_RMPOOL} pdebug )
set( ENV{MP_PROCS} ${NPROCS} )

if( NPROCS GREATER 8 )
    set( ENV{MP_NODES}  2 )
else()
    set( ENV{MP_NODES}  1 )
endif()

execute_process( COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${APP} 
                 TIMEOUT 60
                 RESULT_VARIABLE result
                 OUTPUT_VARIABLE warnings
                 ERROR_VARIABLE  errors
                 ERROR_FILE      ${APP}.MPI${NPROCS}.err
                 OUTPUT_FILE     ${APP}.MPI${NPROCS}.out
               )

file( STRINGS ${APP}.MPI${NPROCS}.err errFile )
if( errFile )
    execute_process( COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --red --bold ${FailTag} )
    message( SEND_ERROR "ERROR:  See details in ${CMAKE_CURRENT_BINARY_DIR}/${APP}.MPI${NPROCS}.err" )
endif()


file( STRINGS ${APP}.MPI${NPROCS}.out output )

# If the file ran, but didn't produce output, something strange has happened and should be analyzed
if( NOT output ) 
    execute_process( COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --red --bold ${FailTag} )
    message( SEND_ERROR "${APP}.MPI${NPROCS} [ UNKNOWN ]" )
endif()


foreach( elem ${output} )
    string( REGEX MATCHALL "Failure" errorLine ${elem} )
    if( errorLine )
        message( ${elem} )
    endif()

    string( REGEX MATCHALL "[Dd]ebug" debugLine ${elem} )
    if( debugLine )
        message( ${elem} )
    endif()
endforeach()

# Check for success
string( REGEX MATCH "Success" wasSuccess ${output} )
if( wasSuccess )
    return()
    
else()
    execute_process( COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --red --bold ${FailTag} )
    message( SEND_ERROR "ERROR:  See details in ${CMAKE_CURRENT_BINARY_DIR}/${APP}.MPI${NPROCS}.out" )
endif()
