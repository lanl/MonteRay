# Function: SyncMirror
# Args    : Host 
#

include( CheckCredFcn )

function( checkMirrorSyncRequirements )
    foreach( requiredName SVN repoRoot mirrorRoot )
        if( NOT DEFINED ${requiredName} )
            set( missingNames "${missingNames} ${requiredName}" )
        endif()
    endforeach()
    if( missingNames )
        message( FATAL_ERROR "**SYNC: ERROR: initializeMirrorSync called without values for : [ ${missingNames} ]" )
    endif()
endfunction()

function( initializeMirrorSync )
    if( ARGC LESS 3 )
        checkMirrorSyncRequirements()
    else()
        set( SVN        ${ARGV1} )
        set( repoRoot   ${ARGV2} )
        set( mirrorRoot ${ARGV3} )
    endif()

    # Will need credentials before pinging the potential hosts
    CheckCredentials()

    # use "svn pg ..." to see which of the big iron machines is available for syncing
    # the repository's mirror hosted on the nfs mounted HPC project space
    set( PotentialRemoteHosts ml-fey tu-fe1 lo-fe3 pi-fey ct-fe1 ty-fe1 hu-fe1 ci-fe1 )
    set( tempRev 0 )

    foreach( rhost ${PotentialRemoteHosts} )
        set( tempURL "svn+ssh://${rhost}${mirrorRoot}" )

        set( getMirrorRev ${SVN} propget svn:sync-last-merged-rev --revprop -r0 ${tempURL} )

        execute_process( COMMAND ${getMirrorRev}
                         OUTPUT_VARIABLE tempRev
                         ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

        if( tempRev GREATER 0 )
            set( MirrorHost ${rhost} )
            break()
        endif()
        
    endforeach()

    if( tempRev EQUAL 0 )
        message( WARNING "**SYNC: Unable to establish link with remote mirror." )
        return()
    else()
        message( "**SYNC: Mirror initial revision [ ${tempRev} ]" )
    endif()
    
    set( MirrorURL  ${tempURL} )

    set( getMirrorTarget ${SVN} propget svn:sync-from-url --revprop -r0 ${MirrorURL} )
    execute_process( COMMAND ${getMirrorTarget}
                     OUTPUT_VARIABLE MirrorTarget
                     ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    
    if( NOT ${MirrorTarget} STREQUAL "file://${repoRoot}" )
        message( FATAL_ERROR "**SYNC: ERROR: Mirror URL [ ${MirrorURL} ] reflects [ ${MirrorTarget} ] NOT [ file://${repoRoot} ]" )
    endif()

    find_program( SCP scp
                  PATHS /usr/bin 
                  DOC "Securely copy files across the network." )
    if( NOT SCP )
        message( FATAL_ERROR "**SYNC: ERROR: Unable to locate a version of scp." )
    endif()

    find_path( HookDir NAMES svnAllowAll svnAllowNone
               PATHS ${CMAKE_MODULE_PATH}
               DOC "Location of the two files used to replace the mirror's start-commit hook." )
    if( NOT HookDir )
        message( FATAL_ERROR "**SYNC: ERROR: Unable to locate svnAllowAll and svnAllowNone." )
    endif()

    set( unlockMirror ${SCP} ${HookDir}/svnAllowAll ${MirrorHost}:${mirrorRoot}/hooks/start-commit )

    set( lockMirror ${SCP} ${HookDir}/svnAllowNone ${MirrorHost}:${mirrorRoot}/hooks/start-commit )

    get_filename_component( SVNBin ${SVN} PATH )

    find_path( SVNPATH NAMES svnsync svnadmin svnlook
               PATHS ${SVNBin}
               DOC "Path to the required subversion tools." )
    if( NOT SVNPATH )
        message( FATAL_ERROR "**SYNC: ERROR: Unable to locate svnsync in same directory as svn" )
    endif()

    set( syncMirror ${SVNPATH}/svnsync sync --non-interactive ${MirrorURL} )
    set( recoverMirror ${SVN} propdel svn:sync-lock --non-interactive --revprop -r0 ${MirrorURL} )

    # Copy commands into Parent/global scope
    foreach( phase unlock sync lock recover )
#        message( "${phase}   : ${${phase}Mirror}" )
        set( SYNC_${phase}Mirror "${${phase}Mirror}" PARENT_SCOPE )
    endforeach()

endfunction()

function( attemptMirrorRecover )
    # should check for lock, but...
    execute_process( COMMAND ${SYNC_recoverMirror} )
endfunction()

function( syncMirror )
    if( NOT DEFINED SYNC_unlockMirror )
        return()
    endif()

    # Will need valid credentials
    CheckCredentials()
    foreach( phase unlock sync lock )
        execute_process( COMMAND ${SYNC_${phase}Mirror}
                         RESULT_VARIABLE Result
                         OUTPUT_VARIABLE Out
                         ERROR_VARIABLE  Err
                        )
        if( NOT Result EQUAL 0 ) 
            set( ErrMsg "${ErrMsg}**SYNC: During [ ${phase} ] encountered [ ${Err} ]  " )
        endif()
    endforeach()
    if( NOT ErrMsg )
        message( "**SYNC: Mirror syncing successful." )
    else()
        message( "**SYNC: Attempting recover." )
        attemptMirrorRecover()
    endif()
endfunction()

