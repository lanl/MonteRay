# Function: CheckTransferCredentials
# Args    : <none>
#
# CheckCredentials can be called to insure the kerberos credentials
# are valid.  If they are not, the function attempts to initialize
# them from a user supplied keytab file.  A failure to renew for
# any reason will invoke a fatal error.
#
# To setup a transfer recognized keytab see transfer.lanl.gov
#
function( checkUsingTransfer )
    find_path( YELLOW mcatk PATHS /home/xshares/PROJECTS DOC "Test to determine transfer default." NO_DEFAULT_PATH )
    if( YELLOW )
        set( AvailableRemoteSystem yellow@transfer.lanl.gov )
    else()
        set( AvailableRemoteSystem red@transfer.lanl.gov )
    endif()
    set( RemoteCommand myfiles )
    find_program( SSH ssh )
    set( Cmd ${SSH} -oBatchMode=yes ${AvailableRemoteSystem} ${RemoteCommand} )
    #message( WARNING "Transfer Check Cmd: [${Cmd}]" )
    execute_process( COMMAND ${Cmd}
                     RESULT_VARIABLE ResultOfRemoteCmd
                     OUTPUT_VARIABLE myfilesOutput
                     OUTPUT_STRIP_TRAILING_WHITESPACE 
                     ERROR_VARIABLE myfilesErr 
                     ERROR_STRIP_TRAILING_WHITESPACE
                 )
    if( ResultOfRemoteCmd EQUAL 0 )
        set( transferOK TRUE PARENT_SCOPE )
        return()
    endif()
    
    # Some successful connections to transfer result in non-zero error codes!
    string( REGEX MATCH "No files found" noFiles ${myfilesErr} )
    if( noFiles )
        set( transferOK TRUE PARENT_SCOPE )
    else()
        set( transferOK FALSE PARENT_SCOPE )
    endif()
endfunction()


function( CheckTransferCredentials )

   set( ExpectedDirs /usr/kerberos/bin /usr/bin /usr/local/bin )
   find_program( KINIT 
                 NAME kinit
                 PATHS ${ExpectedDirs}
                       ENV PATH
                 DOC "Kerberos app for renewing credentials"
                 NO_DEFAULT_PATH )
   if( NOT KINIT )
       message( FATAL_ERROR "**TCRED: Unable to locate kerberos executable - kinit." )
   endif()
   
   checkUsingTransfer()
   if( transferOK )
       return()
   endif()
   
   message( "**TCRED: Attempting credentials renewal for transfer..." )
   
   # Make sure the user environment defines HOME and USER
   if( NOT DEFINED ENV{HOME} )
       message( FATAL_ERROR "**TCRED: ERROR: Must specify the HOME environment variable" )
   endif()
   if( NOT DEFINED ENV{USER} )
       message( FATAL_ERROR "**TCRED: ERROR: Must specify the USER environment variable" )
   endif()
   
   # Locate a keytab file from which new credentials can be generate
   find_file( TransferKeyFile transfer.keytab
              PATHS ENV HOME
              PATH_SUFFIXES .ssh
              DOC "Transfer specific keytab file for renewing credentials" )
         
   if( NOT TransferKeyFile )
       message( FATAL_ERROR "**TCRED: ERROR: Unable to locate transfer.keytab for user!" )
   endif()
   
   set( principal transfer/sdnba@lanl.gov )
   set( renewCredCmd ${KINIT} -f -l 8h -kt ${TransferKeyFile} ${principal} )
   execute_process( COMMAND ${renewCredCmd}
                    RESULT_VARIABLE InitStatus
                    OUTPUT_VARIABLE InitOut
                    ERROR_VARIABLE  InitErr
                   )
   string( REPLACE ";" " " cmdtemp "${renewCredCmd}" )
   
   if( NOT InitStatus EQUAL 0 )
       message( FATAL_ERROR "**TCRED: FAILURE: Unable to renew transfer credentials. Tried: [ ${cmdtemp} ]" )
   endif()
   
   checkUsingTransfer()
   if( NOT transferOK )
       message( FATAL_ERROR "**TCRED: FAILURE: Unable to renew credentials. Connection to transfer failed. Tried: [ ${cmdtemp} ]" )
   endif()
   
   message( "**TCRED: SUCCESS: Credentials for [ ${principal} ] have been renewed." )
endfunction()
