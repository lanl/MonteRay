list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR} )
include( PlatformInfo )

PlatformInfo()
message( "HostDomain : [${HostDomain}]" )
if( NOT ${HostDomain} STREQUAL xdiv )
    message( "Repository synchronization should only be initiated from machines on the xcp lan." )
endif()

include( SyncRepository )
include( pushChanges )

initializeSVN()

set( SVN ${CTEST_SVN_COMMAND} )
set( repoRoot ${MCATK_Repository} )
set( mirrorRoot "/usr/projects/mcatk/mirror" )

message( "Attempting sync from [${repoRoot}] to  [${mirrorRoot}]" )
initializeMirrorSync()
initializePush()

#################################
# Synchronize with yellow
syncMirror()
        
#################################
# Synchronize with red
pushChanges()
        
