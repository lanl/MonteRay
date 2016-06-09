#  ___    _            _   _  __         ____  _       _    __                      
# |_ _|__| | ___ _ __ | |_(_)/ _|_   _  |  _ \| | __ _| |_ / _| ___  _ __ _ __ ___  
#  | |/ _` |/ _ \ '_ \| __| | |_| | | | | |_) | |/ _` | __| |_ / _ \| '__| '_ ` _ \ 
#  | | (_| |  __/ | | | |_| |  _| |_| | |  __/| | (_| | |_|  _| (_) | |  | | | | | |
# |___\__,_|\___|_| |_|\__|_|_|  \__, | |_|   |_|\__,_|\__|_|  \___/|_|  |_| |_| |_|
#                                |___/                                              
#
# ====================================================================
include(PlatformInfo)
PlatformInfo()

#  _____         _   ____            _         
# |___ / _ __ __| | |  _ \ __ _ _ __| |_ _   _ 
#   |_ \| '__/ _` | | |_) / _` | '__| __| | | |
#  ___) | | | (_| | |  __/ (_| | |  | |_| |_| |
# |____/|_|  \__,_| |_|   \__,_|_|   \__|\__, |
#                                        |___/ 
# ====================================================================
find_path( package_dir 
           NAMES lib include
           PATHS /local
                 /home/xshares/PROJECTS
                 /home/xshares
                 /usr/projects 
                 /usr/gapps 
                 /opt/local/bin
           PATH_SUFFIXES mcatk/packages/${Platform} mcatk/packages/${CMAKE_SYSTEM_NAME}
           DOC "Location of third party packages required by the toolkit" 
           NO_DEFAULT_PATH )

if( NOT EXISTS ${package_dir} )
  # try PACKAGEDIR environment variable
  find_path( package_dir 
           NAMES lib include
           PATHS $ENV{PACKAGEDIR}
           DOC "Location of third party packages required by the toolkit" 
           NO_DEFAULT_PATH )
endif()

if( NOT EXISTS ${package_dir} OR NOT IS_DIRECTORY ${package_dir} )
  message( FATAL_ERROR "Unable to locate required package dir." )
else()
    message( "Found toolkit packages in [ ${package_dir} ]" )
endif()

#  ___           _        _ _       _   _             
# |_ _|_ __  ___| |_ __ _| | | __ _| |_(_) ___  _ __  
#  | || '_ \/ __| __/ _` | | |/ _` | __| |/ _ \| '_ \ 
#  | || | | \__ \ || (_| | | | (_| | |_| | (_) | | | |
# |___|_| |_|___/\__\__,_|_|_|\__,_|\__|_|\___/|_| |_|
# ====================================================================

find_path( install_dir 
           NAMES release
           PATHS /local
                 /home/xshares/PROJECTS 
                 /home/xshares 
                 /usr/projects 
                 /usr/gapps 
           PATH_SUFFIXES mcatk
           DOC "Default location to install toolkit" 
           NO_DEFAULT_PATH )
           
if( NOT EXISTS ${install_dir} )
  # try INSTALLDIR environment variable
  if( DEFINED ENV{INSTALLDIR} ) 
    set( install_dir $ENV{INSTALLDIR} )
  endif()
endif()

if( NOT EXISTS ${install_dir} OR NOT IS_DIRECTORY ${install_dir} )
    message( FATAL_ERROR "Unable to locate root directory for installation." )
    
else()

    if( ReleaseName )
        # Find subdirectory structure for releases.
        set( tempDirType developer )
        if( isProdRelease )
            set( tempDirType release )
        endif()
        
        set( MCATK_ReleaseDir "${install_dir}/${tempDirType}/${ReleaseName}" CACHE PATH "Location of toolkit release files" )
        if( EXISTS ${MCATK_ReleaseDir} )
            set( OverwritingPublicRelease false )
        endif()
    else()
        set( DefaultReleaseName "SandboxRelease" )
        # Most builds will be installed in the binary directory
        set( MCATK_ReleaseDir "${CMAKE_BINARY_DIR}/${DefaultReleaseName}" CACHE PATH "Location of toolkit release files" )
        # remove the installation directory when doing 'make clean'
        set_property( DIRECTORY APPEND PROPERTY
                      ADDITIONAL_MAKE_CLEAN_FILES "${MCATK_ReleaseDir}" )
    endif()

    set( CMAKE_INSTALL_PREFIX "${MCATK_ReleaseDir}" CACHE INTERNAL "Prefix prepended to install directories" FORCE )

endif()

