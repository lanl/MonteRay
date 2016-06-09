# Function : pushChanges
# 
# Provides the means for keeping a mirror repository current across network partition
#
include( CheckCredFcn )
include( CheckTransferCred )

function( CheckInitializePushConditions )
    foreach( requiredName SVN repoRoot )
        if( NOT DEFINED ${requiredName} )
            set( missingNames "${missingNames} ${requiredName}" )
        endif()
    endforeach()
    if( missingNames )
        message( FATAL_ERROR "**REPO PUSH: ERROR: initializePush called without values for : [ ${missingNames} ]" )
    endif()
endfunction()

#=====================================================================
# Function: initializePush
# 
# Sets up arguments, locates executables and insures existence of output directory
function( initializePush )
    if( ARGC LESS 2 )
        CheckInitializePushConditions()
    else()
        set( SVN        ${ARGV1} )
        set( repoRoot   ${ARGV2} )
    endif()
    
    # Setup the necessary commands for querying the local repository
    get_filename_component( SVNBin ${SVN} PATH )

    find_path( SVNPATH NAMES svnadmin svnlook
               PATH ${SVNBin}
               DOC "Path to the required subversion tools." )
    if( NOT SVNPATH )
        message( FATAL_ERROR "**REPO PUSH: ERROR: Unable to locate svnadmin and svnlook in same directory as svn" )
    endif()

    set( SYNCdump ${SVNPATH}/svnadmin dump --incremental )
    set( RPUSH_dump ${SYNCdump} PARENT_SCOPE )
    
    set( getLatestRev ${SVNPATH}/svnlook youngest ${repoRoot} )
    set( RPUSH_getLatest ${getLatestRev} PARENT_SCOPE )
    
    
    # This directory contains the result of the archived changes PLUS a file denoting the expected
    # remote head version once these changes have been loaded.
    set( pushdir ${repoRoot}/../RedSync )
    find_path( PushDir RedHeadRev
               PATHS ${pushdir} )
    if( NOT PushDir )
        message( FATAL_ERROR "**REPO PUSH: ERROR: Unable to locate directory [ ${pushdir} ] for storing result of push." )
    endif()
    set( RPUSH_dir ${PushDir} PARENT_SCOPE )
    
    # Transfer specific configuration
    # Setup ssh command.
    find_program( SSH ssh
                  PATH /usr/bin
                  DOC "Secure shell network protocol." )
    if( NOT SSH )
        message( FATAL_ERROR "**REPO SSH: ERROR: Unable to locate a version of ssh." )
    endif()
    # Setup ssh command.
    find_program( SCP scp
                  PATH /usr/bin
                  DOC "Secure network protocal." )
    if( NOT SCP )
        message( FATAL_ERROR "**REPO SCP: ERROR: Unable to locate a version of scp." )
    endif()

    set( Recipient drnuke@lanl.gov )
    set( TransferDestination red@transfer.lanl.gov CACHE STRING "Indicates that the transfer is from yellow to red" )
    set( TransferInitCmd ${SSH} ${TransferDestination} init --quiet --recipients=${Recipient} CACHE STRING "Transfer command that initiates a yellow->red transfer" ) 
    set( TransferSubmitCmd ${SSH} yellow@transfer.lanl.gov submit CACHE STRING "Finalize transfer command that performs the actual push." )
endfunction()

#=====================================================================
# Function: pushChanges
# 
# Generates a list of the required changes and pushes them to mercury
function( pushChanges )

    if( NOT DEFINED RPUSH_dir )
        return()
    endif()

    file( READ ${RPUSH_dir}/RedHeadRev redHeadRev )
    execute_process( COMMAND ${RPUSH_getLatest}
                     OUTPUT_VARIABLE RepoLatestVer
                     ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
                     
    if( NOT RepoLatestVer GREATER redHeadRev )
        message( WARNING "**REPO PUSH:  Push directory appears to be up to date!?" )
        return()
    endif()
    
    #------------------------------
    # Generate repository updates...
    math( EXPR startRev "${redHeadRev}+1" )
    set( dumpName ${RPUSH_dir}/updates_r${redHeadRev}.dump )
    
    message( "**REPO PUSH:  Dumping rev[ ${startRev} ] to rev[ ${RepoLatestVer} ] into file: [ ${dumpName} ]" )
    execute_process( COMMAND ${RPUSH_dump} -r${startRev}:${RepoLatestVer} ${repoRoot}
                     OUTPUT_FILE ${dumpName} )
                     
    #------------------------------
    # TRANSFER Based
    #
    CheckTransferCredentials()
     
    execute_process( COMMAND ${TransferInitCmd} 
                     RESULT_VARIABLE ResultOfInit
                     OUTPUT_VARIABLE TransferID OUTPUT_STRIP_TRAILING_WHITESPACE
                     ERROR_VARIABLE InitErrMsg ERROR_STRIP_TRAILING_WHITESPACE )
    if( ResultOfInit EQUAL 0 AND TransferID )
        execute_process( COMMAND ${SCP} ${dumpName} ${TransferDestination}:${TransferID} 
                         RESULT_VARIABLE ResultOfCopy )
        message( "Commit ID [ ${TransferID} ] Copy Result: [${ResultOfCopy}]" )
        execute_process( COMMAND ${TransferSubmitCmd} ${TransferID} 
                         RESULT_VARIABLE ResultOfSubmit )
        message( "Commit ID [ ${TransferID} ] Result: [${ResultOfSubmit}]" )
        if( ResultOfSubmit EQUAL 0 )
            file( WRITE ${RPUSH_dir}/RedHeadRev "${RepoLatestVer}" )
        endif()
    else()
        message( WARNING "TRANSFER ERROR: While executing [ ${TransferInitCmd} ]" )
        message( WARNING "Unable to initialize transfer of files. ID: [${TransferID}] Error: [${InitErrMsg}]" )
    endif()
    
    # either way, remove the dumpfile
    file( REMOVE ${dumpName} )                
endfunction()
