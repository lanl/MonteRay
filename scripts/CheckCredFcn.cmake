# Function: CheckCredentials
# Args    : <none>
#
# CheckCredentials can be called to insure the kerberos credentials
# are valid.  If they are not, the function attempts to initialize
# them from a user supplied keytab file.  A failure to renew for
# any reason will invoke a fatal error.
#
# To setup a keytab
#
# 1) Create the keytab on a XLAN machine (xray/candycorn,etc.) NOTE: Name of file doesn't really matter
#    host> setenv KEYTABFILE $HOME/.ssh/cron.keytab
#    host> kadmin -p $USER@lanl.gov
#    kadmin> ktadd -k $KEYTABFILE -p $USER@lanl.gov
#    kadmin> quit
# 2) Ensure permissions are for you to read/write ONLY ( -rw------- !)
# 2) place a copy (DON'T regen!) where the other machines can see it.
#    host> scp $HOME/.ssh/cron.keytab tu-fe1:~/.ssh/
# 3) Copy wherever else it may be needed (i.e. other non-cross mounted homespaces on turquoise)
function( checkUsingSSH )
    set( AvailableRemoteSystem xlogin.lanl.gov )
    set( RemoteCommand cd )
    find_program( SSH ssh )
    set( Cmd ${SSH} -oBatchMode=yes ${AvailableRemoteSystem} ${RemoteCommand} )
    #message( WARNING "SSH Check Cmd: [${Cmd}]" )
    execute_process( COMMAND ${Cmd}
                     RESULT_VARIABLE ResultOfRemoteCmd
                     OUTPUT_QUIET ERROR_QUIET
                 )
    if( ResultOfRemoteCmd EQUAL 0 )
        set( sshOK TRUE PARENT_SCOPE )
    else()
        set( sshOK FALSE PARENT_SCOPE )
    endif()
endfunction()


function( CheckCredentials )

   set( ExpectedDirs /usr/kerberos/bin /usr/bin /usr/local/bin )
   find_program( KINIT 
                 NAME kinit
                 PATHS ${ExpectedDirs}
                       ENV PATH
                 DOC "Kerberos app for renewing credentials"
                 NO_DEFAULT_PATH )
   if( NOT KINIT )
       message( FATAL_ERROR "**CRED: Unable to locate kerberos executable - kinit." )
   endif()
   
   checkUsingSSH()
   if( sshOK )
       return()
   endif()
   
   message( "**CRED: Attempting credentials renewal..." )
   
   # Make sure the user environment defines HOME and USER
   if( NOT DEFINED ENV{HOME} )
       message( FATAL_ERROR "**CRED: ERROR: Must specify the HOME environment variable" )
   endif()
   if( NOT DEFINED ENV{USER} )
       message( FATAL_ERROR "**CRED: ERROR: Must specify the USER environment variable" )
   endif()
   
   # Locate a keytab file from which new credentials can be generate
   find_file( KeyFile cron.keytab
              PATHS ENV HOME
              PATH_SUFFIXES .ssh
              DOC "Keytab file for renewing credentials" )
         
   if( NOT KeyFile )
       message( FATAL_ERROR "**CRED: ERROR: Unable to locate cron.keytab for user!" )
   endif()
   
   set( renewCredCmd ${KINIT} -f -l 8h -kt ${KeyFile} $ENV{USER}@lanl.gov )
   execute_process( COMMAND ${renewCredCmd}
                    RESULT_VARIABLE InitStatus
                    OUTPUT_VARIABLE InitOut
                    ERROR_VARIABLE  InitErr
                   )
   string( REPLACE ";" " " cmdtemp "${renewCredCmd}" )
   
   if( NOT InitStatus EQUAL 0 )
       message( FATAL_ERROR "**CRED: FAILURE: Unable to renew credentials. Tried: [ ${cmdtemp} ]" )
   endif()
   
   checkUsingSSH()
   if( NOT sshOK )
       message( FATAL_ERROR "**CRED: FAILURE: Unable to renew credentials. SSH connection failed. Tried: [ ${cmdtemp} ]" )
   endif()
   
   message( "**CRED: SUCCESS: Credentials for [ $ENV{USER}@lanl.gov ] have been renewed." )
endfunction()
