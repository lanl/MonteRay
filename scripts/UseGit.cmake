include( CheckTransferCred )

#######################################################################
# MCATK via git 
#--------------------------------
function( initializeRepository )
    if( NOT CTEST_SOURCE_DIRECTORY )
        message( FATAL_ERROR "user must set CTEST_SOURCE_DIRECTORY prior to invocation." )
    endif()
    
    # insure we can find the 'git' executable
    find_package( Git REQUIRED )

    # Need to tell CTest
    set( CTEST_GIT_COMMAND ${GIT_EXECUTABLE} PARENT_SCOPE )

    if( BranchName )
        set( BranchRequest "--branch ${BranchName}" )
    else()
        set( BranchName "NOBRANCH" )
        unset( BranchRequest )
    endif()

    set( StashRepo "ssh://git@xcp-stash.lanl.gov:7999/mr/monteray" )

    execute_process( COMMAND ${GIT_EXECUTABLE} ls-remote ${StashRepo} 
                     TIMEOUT 5 
                     OUTPUT_VARIABLE gitBranches
                     RESULT_VARIABLE useMirror )
    if( useMirror )
        find_path( LocalCloneDir FETCH_HEAD
                   PATHS ${MCATKDir} ${MCATKDir}/../mcatk_autobuild
                   PATH_SUFFIXES mirror/mcatk.git )
        if( NOT LocalCloneDir )
            message( FATAL_ERROR "-- GIT: Unable to locate a *locally* mirrored repository. Looked under : [${MCATKDir}]" )
        endif()
        set( repo "file://${LocalCloneDir}" )
        message( "-- GIT: Using mirror: [${repo}] for ${BranchRequest}" )
    else()
        set( repo ${StashRepo} )
        message( "-- GIT: Using bitbucket: [${repo}] for ${BranchRequest}" )
    endif()

    # Needs to be checked everytime ..... before the below where there is a return() before then next CIType check.
    # FYI CIType is setup in PlatformInfo.cmake if it needs to be set at all.
    if( CIType STREQUAL HandCI )
      find_file( gitMCATKBashScript gitMONTERAY.bash
             PATHS ${CMAKE_MODULE_PATH} 
                   ${CTEST_SCRIPT_DIRECTORY} 
                   ${AutoBuildRoot} 
             PATH_SUFFIXES scripts
      )
      set( CTEST_GIT_UPDATE_CUSTOM "bash ${gitMCATKBashScript} UPDATE" PARENT_SCOPE)
    endif()
    
    
    # Can't clone to an existing directory! if its there assume it's ok. update function gets called later
    if( EXISTS ${CTEST_SOURCE_DIRECTORY} AND EXISTS ${CTEST_SOURCE_DIRECTORY}/src )
        return()
    endif()
    
    # Directory either doesn't exist or is corrupted (lacks the src dir)
    if( EXISTS ${CTEST_SOURCE_DIRECTORY} )
        execute_process( COMMAND ${CMAKE_COMMAND} -E remove_directory ${CTEST_SOURCE_DIRECTORY} )
    endif()


    if( CIType STREQUAL HandCI )
      set( CTEST_CHECKOUT_COMMAND  "bash ${gitMCATKBashScript} ${GIT_EXECUTABLE} ${BranchName} ${repo} ${CTEST_SOURCE_DIRECTORY}"  PARENT_SCOPE)
    else()
      set( CTEST_CHECKOUT_COMMAND "${GIT_EXECUTABLE} clone ${BranchRequest} ${repo} ${CTEST_SOURCE_DIRECTORY}" PARENT_SCOPE )   
    endif()

    
endfunction()

#=====================================================================
# Function: initializePush
# Called From: MCATK_ContinuousBuild.cmake
# 
# Sets up arguments, locates executables and insures existence of output directory
function( initializePush Repo )
    if( DEFINED BatchSystem )
        message( FATAL_ERROR "Repository transactions should only happen on non-HPC platforms" )
    endif()
     
    # insure we can find the 'git' executable
    find_package( Git REQUIRED )
    find_program( TAR tar )
    
    set( repo "ssh://git@xcp-stash.lanl.gov:7999/MCATK/${Repo}" )

    execute_process( COMMAND ${GIT_EXECUTABLE} ls-remote ${repo} 
                     TIMEOUT 5 
                     OUTPUT_VARIABLE repoExists
                     RESULT_VARIABLE repoNotFound )

    if( repoNotFound )
        message( FATAL_ERROR "Repo [${Repo}] Does not appear to exist." )
    endif()

    set( MIRROR_ROOT /local/mcatk/users/$ENV{USER}/MIRROR )
    set( MIRROR      ${MIRROR_ROOT}/${Repo}               )
    set( MirrorRoot  ${MIRROR_ROOT}      PARENT_SCOPE     )
    set( MirrorName  ${Repo}             PARENT_SCOPE     )
    set( GitMirror   ${MIRROR}           PARENT_SCOPE     )
    set( GetGitTag   ${GET_TAG_CMD}      PARENT_SCOPE     )

    if( NOT EXISTS ${MIRROR} )
        execute_process( COMMAND ${GIT_EXECUTABLE} clone --mirror ${repo} ${MIRROR} )
    endif()
    
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

    if( NOT DEFINED GitMirror )
        message( FATAL_ERROR "**REPO pushChanges: ERROR: User must call initializePush prior to this function." )
    endif()
    
    set( UpdateMirror ${GIT_EXECUTABLE} -C ${GitMirror} remote update  )
    set( GetGitTag    ${GIT_EXECUTABLE} -C ${GitMirror} rev-parse --short HEAD )
    
    execute_process( COMMAND ${GetGitTag} OUTPUT_VARIABLE OLD_TAG ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE )
    execute_process( COMMAND ${UpdateMirror} )
    execute_process( COMMAND ${GetGitTag} OUTPUT_VARIABLE NEW_TAG ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
                     
    if( OLD_TAG EQUAL NEW_TAG )
        message( WARNING "**REPO PUSH:  Push directory appears to be up to date." )
        if( force )
            message( WARNING "**REPO PUSH:  Forcing..." )
        else()
            message( WARNING "**REPO PUSH:  Done." )
            return()
        endif()
    endif()
    
    #-----------------------------------------------------------
    # Compress mirror
    #
    # the extension specified on the following line will tell 'tar' which compression algorithm to use. (see man 'tar -a ...' )
    set( PackageName ${MirrorName}_${NEW_TAG}_Repo.tar.xz )
    execute_process( COMMAND ${TAR} acf ${PackageName} ${MirrorName} 
                     WORKING_DIRECTORY ${MirrorRoot} )
    set( PackedMirror ${MirrorRoot}/${PackageName} )
    
    if( NOT EXISTS ${PackedMirror} )
        message( FATAL_ERROR "Packing of repository failed. In [${MirrorRoot} ], no file [ ${PackageName} ]." )
    endif()
    message( "**REPO PUSH:  Old: [ ${OLD_TAG} ] New: [ ${NEW_TAG} ]" )
                     
    #------------------------------
    # TRANSFER Based
    #
    # The LANL Transfer system requires a different type of credentials
    CheckTransferCredentials()
     
    # 1. Initialize transfer channel
    execute_process( COMMAND ${TransferInitCmd} 
                     RESULT_VARIABLE ResultOfInit
                     OUTPUT_VARIABLE TransferID OUTPUT_STRIP_TRAILING_WHITESPACE
                     ERROR_VARIABLE InitErrMsg ERROR_STRIP_TRAILING_WHITESPACE )
    # 2. Check channel is open
    if( ResultOfInit EQUAL 0 AND TransferID )
    
        # 3. Start copying files to server
        execute_process( COMMAND ${SCP} ${PackedMirror} ${TransferDestination}:${TransferID} 
                         RESULT_VARIABLE ResultOfCopy )
        message( "Commit ID [ ${TransferID} ] Copy Result: [${ResultOfCopy}]" )
        
        # 4. Initiate transfer
        execute_process( COMMAND ${TransferSubmitCmd} ${TransferID} 
                         RESULT_VARIABLE ResultOfSubmit )
        message( "Commit ID [ ${TransferID} ] Result: [${ResultOfSubmit}]" )

    else()
        message( WARNING "TRANSFER ERROR: While executing [ ${TransferInitCmd} ]" )
        message( WARNING "Unable to initialize transfer of files. ID: [${TransferID}] Error: [${InitErrMsg}]" )
    endif()
    
    # either way, remove the dumpfile
    file( REMOVE ${PackedMirror} )
endfunction()
