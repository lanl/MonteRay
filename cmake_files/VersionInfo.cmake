# Setup up versioning information

# Versioning
# See also http://en.wikipedia.org/wiki/Software_versioning
set( Toolkit_Major_Version 1 )
set( Toolkit_Minor_Version 0 )
set( Toolkit_Minor_Revision "1beta" )
set( ToolkitVersion ${Toolkit_Major_Version}.${Toolkit_Minor_Version}.${Toolkit_Minor_Revision} )
message( "Toolkit Version is [ ${ToolkitVersion} ]" )

MACRO (TODAY RESULT)
    IF (WIN32)
        EXECUTE_PROCESS(COMMAND "cmd" " /C date /T" OUTPUT_VARIABLE ${RESULT})
        string(REGEX REPLACE "(..)/(..)/..(..).*" "\\1/\\2/\\3" ${RESULT} ${${RESULT}})
    ELSEIF(UNIX)
        EXECUTE_PROCESS(COMMAND "date" "+%m/%d/%Y" OUTPUT_VARIABLE ${RESULT})
        string(REGEX REPLACE "(..)/(..)/..(..).*" "\\1/\\2/\\3" ${RESULT} ${${RESULT}})
    ELSE (WIN32)
        MESSAGE(SEND_ERROR "date not implemented")
        SET(${RESULT} 000000)
    ENDIF (WIN32)
ENDMACRO (TODAY)
TODAY(CompilationDate)
message( "Toolkit compilation date is [ ${CompilationDate} ]" )
