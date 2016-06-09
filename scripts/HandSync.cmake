#invoke -- ctest -S <scriptname>


find_file( MergeScript 
           MergeTestResults.py 
           PATHS ${CTEST_SCRIPT_DIRECTORY}
            )

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CTEST_SCRIPT_DIRECTORY} )
include( CheckCredFcn )
include( SyncRepository )
include( pushChanges )

find_package( Subversion REQUIRED )

find_path( ProjectDir
           NAMES AutoBuilds packages
           PATHS /home/xshares/PROJECTS /usr/aprojects /usr/projects /usr/gapps 
           PATH_SUFFIXES mcatk )
if( NOT ProjectDir )
    message( FATAL_ERROR "Could not locate MCATK project directory on this system." )
endif()

find_path( RepositoryRoot
           NAMES db hooks
           PATHS  ${ProjectDir}
           PATH_SUFFIXES svn/repo mirror )
if( NOT RepositoryRoot )
    message( FATAL_ERROR "Could not locate MCATK's repository on this system." )
endif()

set( SVN ${Subversion_SVN_EXECUTABLE} )
set( repoRoot ${RepositoryRoot} )
set( mirrorRoot /usr/projects/mcatk/mirror )
    
initializeMirrorSync()
initializePush()

#################################
# update mirror
syncMirror()
  
#################################
# Synchronize with red
pushChanges()
        
