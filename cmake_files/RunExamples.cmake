################################################################################
# Analyze results of unit test runs

################################################################################
#get_filename_component( TestDirName ${CMAKE_CURRENT_BINARY_DIR} NAME )

#set( CmdTag "Testing ${UNIT}[ ${TestDirName} ]..." )
#set( FailTag "${CmdTag} FAIL" )
set( FailTag "${TAG}  FAIL" )

file( GLOB oldCSVs "*.csv" "*.txt" )
if( oldCSVs )
    execute_process( COMMAND ${CMAKE_COMMAND} -E remove ${oldCSVs} )
endif()

execute_process( COMMAND ${JOBCTRL} ${APP} ${args}
                 TIMEOUT 7200
                 RESULT_VARIABLE result
                 OUTPUT_VARIABLE warnings
                 ERROR_VARIABLE  errors
                 ERROR_FILE      ${NAME}.err
                 OUTPUT_FILE     ${NAME}.out
               )

if( NOT ${result} EQUAL 0 ) 
  if( ${result} MATCHES "^ *[a-zA-Z]+" )
    message(STATUS "result= ${result}")
  endif()
endif()
 
if( ${warnings} MATCHES "^ *[a-zA-Z]+" )           
  message(STATUS "warnings= ${warnings}")
endif()

if( ${errors} MATCHES "^ *[a-zA-Z]+" )       
  message(STATUS "errors= ${errors}")
endif()
    
# Collect any information from the test execution phase
file( STRINGS ${NAME}.out output )
file( STRINGS ${NAME}.err errFile )

# If the file ran, but didn't produce output, something strange has happened and should be analyzed
if( NOT output )
    execute_process( COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --red --bold ${FailTag} )
    if( errFile )
        message( SEND_ERROR "${NAME} ERROR:[ ${errFile} ]" )
    else()
        message( SEND_ERROR "${NAME} [ UNKNOWN ]" )
    endif()
endif()

if( NOT ${result} EQUAL 0 ) 
    # Report non-Zero return value
    execute_process( COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --red --bold ${FailTag} )
    
    if( errFile )   
        message( SEND_ERROR  "Problem running ${NAME}.  Error: ${result}.  See details in ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.err" )
    else()
        execute_process( COMMAND ${CMAKE_COMMAND} -E remove ${NAME}.err )
        message( SEND_ERROR  "Problem running ${NAME}.  Error: ${result}.  File: ${NAME}.err is empty and will be removed." )
    endif()
endif()

# If the file ran, but didn't produce errors, remove the error file
if( NOT errFile )
    execute_process( COMMAND ${CMAKE_COMMAND} -E remove ${NAME}.err )
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

set( DIFF_IGNORE_COMMAND "^\\(Reported:\\)\\|\\(Debug:\\)\\|\\(DEBUG:\\)\\|\\(Application\\)\\|\\(Timing:\\)" )
execute_process( COMMAND diff -w -I ${DIFF_IGNORE_COMMAND} --brief ${NAME}.out Baseline
                 OUTPUT_FILE DiffMessage
                 RESULT_VARIABLE diffresult
               )     

# This is essential if it detects adjacent newline characters
cmake_policy( SET CMP0007 NEW )

# Check for failure
if( NOT diffresult EQUAL 0 ) 
    execute_process( COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --red --bold ${FailTag} )

    message( "\t\t\t\t\t\t\t\t\t\t     BASELINE <<<<<< | >>>>>> ${NAME}.out" )
    execute_process( COMMAND diff -t --suppress-common-lines -by -W 198 Baseline ${NAME}.out 
                     OUTPUT_FILE diffOutputFile
                    )

    # Convert output file to list of strings
    file( STRINGS diffOutputFile diffOut ) 
    
    list( LENGTH diffOut NumDiff )
    set( FirstLines 10 )
    if( NumDiff LESS FirstLines )
        math( EXPR FirstLines "${NumDiff}-1" )
        set( DONE true )
    endif()

    # Print the 1st 10 lines
    foreach( lineID RANGE 0 ${FirstLines} )
        list( GET diffOut ${lineID} outLine )
        message( "${outLine}" )
    endforeach()
    
    if( DONE )
        message( SEND_ERROR "ERROR: *** Baselines Differ ***" )
    endif()

    math( EXPR Offset "${NumDiff} - 40" )

    if( Offset LESS FirstLines )
        set( Offset ${FirstLines} )
    else()
        # Indicate that we're trimming the output
        message( "  <SNIP>" )
    endif()

    math( EXPR NumDiff "${NumDiff} - 1" )

    # Print out the last lines of the file
    foreach( lineID RANGE ${Offset} ${NumDiff} )
        list( GET diffOut ${lineID} outLine )
        message( "${outLine}" )
    endforeach()
    message( SEND_ERROR "ERROR: *** Baselines Differ ***" )
endif()
