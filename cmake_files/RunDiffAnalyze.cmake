################################################################################
# Analyze results of unit test runs

################################################################################
#get_filename_component( TestDirName ${CMAKE_CURRENT_BINARY_DIR} NAME )

#set( CmdTag "Testing ${UNIT}[ ${TestDirName} ]..." )
#set( FailTag "${CmdTag} FAIL" )
set( FailTag "${TAG}  FAIL" )

string( REGEX REPLACE " " ";" args ${OPTION1} )

execute_process( COMMAND ${MAKE_PROGRAM} ${SETUP_DEPENDS}
                 TIMEOUT 7200
                 RESULT_VARIABLE result_setup
                 OUTPUT_VARIABLE warnings_setup
                 ERROR_VARIABLE  errors_setup
                 ERROR_FILE      ${NAME}_setup.err
                 OUTPUT_FILE     ${NAME}_setup.out
               )

execute_process( COMMAND echo ${TARGET} ${args} ${INPUT}
                 TIMEOUT 7200
                 RESULT_VARIABLE result_echo
                 OUTPUT_VARIABLE warnings_echo
                 ERROR_VARIABLE  errors_echo
                 ERROR_FILE      ${NAME}_echo.err
                 OUTPUT_FILE     ${NAME}_echo.out
               )
               
execute_process( COMMAND ${TARGET} ${args} ${INPUT}
                 TIMEOUT 14400
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
    
set( DIFF_IGNORE_COMMAND "^\\(Reported:\\)\\|\\(Debug:\\)\\|\\(DEBUG:\\)" )
execute_process( COMMAND diff -w -I ${DIFF_IGNORE_COMMAND} --brief ${NAME}.out ${BASELINE}
                 RESULT_VARIABLE diffresult
                 OUTPUT_VARIABLE diffwarnings
                 ERROR_VARIABLE  differrors
                 ERROR_FILE      ${NAME}_diff.err
                 OUTPUT_FILE     ${NAME}_diff.out
               )     

if( NOT ${result} EQUAL 0 ) 
    file( STRINGS ${NAME}.err errFile )
    if( errFile )   
      execute_process( COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --red --bold ${FailTag} )
      message( SEND_ERROR  "Problem running ${NAME}.  Error: ${result}.  See details in ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.err" )
    endif()
endif()

file( STRINGS ${NAME}.out output )
# If the file ran, but didn't produce output, something strange has happened and should be analyzed
if( NOT output )
    execute_process( COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --red --bold ${FailTag} )
    message( SEND_ERROR "${NAME} [ UNKNOWN ]" )
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
if( diffresult ) 
    execute_process( COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --red --bold ${FailTag} )
    message( SEND_ERROR "ERROR: *** ${diffresult} ***" )
else()
    return()
endif()
