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
                 $ENV{UNITTEST_DIR}
           DOC "Location of third party packages required by the toolkit" 
           NO_DEFAULT_PATH )
endif()

if( NOT EXISTS ${package_dir} OR NOT IS_DIRECTORY ${package_dir} )
#  message( FATAL_ERROR "Unable to locate required package dir." )
else()
#    message( "Found toolkit packages in [ ${package_dir} ]" )
endif()

#  ___           _        _ _       _   _             
# |_ _|_ __  ___| |_ __ _| | | __ _| |_(_) ___  _ __  
#  | || '_ \/ __| __/ _` | | |/ _` | __| |/ _ \| '_ \ 
#  | || | | \__ \ || (_| | | | (_| | |_| | (_) | | | |
# |___|_| |_|___/\__\__,_|_|_|\__,_|\__|_|\___/|_| |_|
# ====================================================================

    unset( MCATK_ReleaseDir CACHE ) 
    if( InstallDir )
        message( "LocatePackagesDir.cmake: -- InstallDir is set to [ ${InstallDir} ]." )
                
        set( MCATK_ReleaseDir "${InstallDir}" CACHE PATH "Location of toolkit release files" )
        if( EXISTS ${MCATK_ReleaseDir} )
            set( OverwritingPublicRelease false )
        endif()
    else()
        message( "LocatePackagesDir.cmake: -- InstallDir is NOT set." )
        set( DefaultReleaseName "SandboxRelease" )
        # Most builds will be installed in the binary directory
        set( MCATK_ReleaseDir "${CMAKE_BINARY_DIR}/${DefaultReleaseName}" CACHE PATH "Location of toolkit release files" )
        # remove the installation directory when doing 'make clean'
        set_property( DIRECTORY APPEND PROPERTY
                      ADDITIONAL_MAKE_CLEAN_FILES "${MCATK_ReleaseDir}" )
    endif()

    message( "LocatePackagesDir.cmake: -- Installing MCATK to [ ${MCATK_ReleaseDir} ]" )
    set( CMAKE_INSTALL_PREFIX "${MCATK_ReleaseDir}" CACHE INTERNAL "Prefix prepended to install directories" FORCE )
    
#  _       _    _____     _       _   
# | |_ __ | | _|___ /  __| |_ __ | |_ 
# | | '_ \| |/ / |_ \ / _` | '_ \| __|
# | | | | |   < ___) | (_| | | | | |_ 
# |_|_| |_|_|\_\____/ \__,_|_| |_|\__|
# 
# ====================================================================
find_path( lnk3dnt_location 
           NAMES godiva.lnk3dnt
           PATHS /local
                 /home/xshares/PROJECTS
                 /home/xshares
                 /usr/projects 
                 /usr/gapps 
                 ENV MCATK_LNK3DNT
           PATH_SUFFIXES mcatk/lnk3dnt
           DOC "Location of numerous geometry link files." 
           NO_DEFAULT_PATH )

if( NOT EXISTS ${lnk3dnt_location} )
  # try LNK3DNTDIR environment variable
  find_path( lnk3dnt_location 
           NAMES godiva.lnk3dnt
           PATHS $ENV{LNK3DNTDIR}
           DOC "Location of numerous geometry link files." 
           NO_DEFAULT_PATH )
endif()

if( lnk3dnt_location )
    message( "Found link files at [ ${lnk3dnt_location} ]" )
#    add_definitions( -DMCATK_LINK_DIR="${lnk3dnt_location}" )
else()
    message( FATAL_ERROR "Could not locate link files *** ${lnk3dnt_location}" )
endif()


# MontRayTestFiles
# ====================================================================
find_path( MonteRayTestFiles_location 
           NAMES 1001-70c_MonteRayCrossSection.bin
           PATHS /local
                 /home/xshares/PROJECTS
                 /home/xshares
                 /usr/projects 
                 /usr/gapps 
                 ENV MONTERAY_TESTFILES_DIR
           PATH_SUFFIXES mcatk/MonteRayTestFiles
           DOC "Location of MonteRay Test files." 
           NO_DEFAULT_PATH )

if( NOT EXISTS ${MonteRayTestFiles_location} )
  # try MONTERAY_TESTFILES_DIR environment variable
  find_path( MonteRayTestFiles_location 
           NAMES 1001-70c_MonteRayCrossSection.bin
           PATHS $ENV{MONTERAY_TESTFILES_DIR}
           DOC "Location of MonteRay Test files." 
           NO_DEFAULT_PATH )
endif()

if( MonteRayTestFiles_location )
    message( "Found MonteRay test files at [ ${MonteRayTestFiles_location} ]" )
#    add_definitions( -DMONTERAY_TESTFILES_DIR="${MonteRayTestFiles_location}" )
else()
    message( FATAL_ERROR "Could not locate MonteRay test files *** ${MonteRayTestFiles_location}" )
endif()

