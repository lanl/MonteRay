message( "-- FindMonteRayTestFiles.cmake -- Looking for geometry test files using the following ENVIRONMENT VARIABLES:"  )
message( "-- FindMonteRayTestFiles.cmake --     MCATK_LNK3DNT " )
if(DEFINED ENV{MCATK_LNK3DNT}) 
    find_path( lnk3dnt_location 
               NAMES godiva.lnk3dnt
               PATHS ENV MCATK_LNK3DNT
             )
    if( lnk3dnt_location )
       message( "-- FindMonteRayTestFiles.cmake -- MCATK LNK3DNT test directory set by environment variable. MCATK_LNK3DNT = [ $ENV{MCATK_LNK3DNT} ]" )
    endif()
endif()

if( NOT EXISTS ${lnk3dnt_location} )
    message( "-- FindMonteRayTestFiles.cmake --     LNK3DNTDIR  " )
    if(DEFINED ENV{LNK3DNTDIR}) 
        find_path( lnk3dnt_location 
                   NAMES godiva.lnk3dnt
                   PATHS ENV LNK3DNTDIR
                 )
        if( lnk3dnt_location )
            message( "-- FindMonteRayTestFiles.cmake -- MCATK LNK3DNT test directory set by environment variable. LNK3DNTDIR = [ $ENV{LNK3DNTDIR} ]" )
        endif()
   endif()
endif()

if( NOT EXISTS ${lnk3dnt_location} )
    message( "-- FindMonteRayTestFiles.cmake --     MONTERAY_TESTFILES_DIR  " )
    if(DEFINED ENV{MONTERAY_TESTFILES_DIR}) 
        find_path( lnk3dnt_location 
                   NAMES godiva.lnk3dnt
                   PATHS ENV MONTERAY_TESTFILES_DIR
                 )
        if( lnk3dnt_location )
            message( "-- FindMonteRayTestFiles.cmake -- MCATK LNK3DNT test directory set by environment variable. MONTERAY_TESTFILES_DIR = [ $ENV{MONTERAY_TESTFILES_DIR} ]" )
        endif()
   endif()
endif()

if( NOT EXISTS ${lnk3dnt_location} )
    message( "-- FindMonteRayTestFiles.cmake -- Looking for geometry test files not found using environment variables. " )
    message( "-- FindMonteRayTestFiles.cmake -- Looking for geometry test files in standard locations. " )
    find_path( lnk3dnt_location 
           NAMES godiva.lnk3dnt
           PATHS /local
                 /home/xshares/PROJECTS
                 /home/xshares
                 /usr/projects 
                 /usr/gapps                 
           PATH_SUFFIXES mcatk/lnk3dnt
           DOC "Location of numerous geometry link files." 
           NO_DEFAULT_PATH )
endif()

if( lnk3dnt_location )
    message( "-- FindMonteRayTestFiles.cmake -- Found geometry test files at [ ${lnk3dnt_location} ]" )
#    add_definitions( -DMCATK_LINK_DIR="${lnk3dnt_location}" )
else()
    message( FATAL_ERROR "-- FindMonteRayTestFiles.cmake -- Could not locate geometry test files.  Consider setting the LNK3DNTDIR environment variable." )
endif()


# MontRayTestFiles
# ====================================================================
message( "-- FindMonteRayTestFiles.cmake -- Looking for MonteRay test files (cross-sections, collisions files, etc.) using the following ENVIRONMENT VARIABLES:"  )
message( "-- FindMonteRayTestFiles.cmake --     MONTERAY_TESTFILES_DIR " )
if(DEFINED ENV{MONTERAY_TESTFILES_DIR}) 
    find_path( MonteRayTestFiles_location 
               NAMES 1001-70c_MonteRayCrossSection.bin
               PATHS ENV MONTERAY_TESTFILES_DIR
             )
    if( MonteRayTestFiles_location )
       message( "-- FindMonteRayTestFiles.cmake -- MonteRay test file directory set by environment variable. MONTERAY_TESTFILES_DIR = [ $ENV{MONTERAY_TESTFILES_DIR} ]" )
    endif()
endif()

if( NOT EXISTS ${MonteRayTestFiles_location} )
    message( "-- FindMonteRayTestFiles.cmake -- MonteRay test file directory not found using environment variables. " )
    message( "-- FindMonteRayTestFiles.cmake -- Looking for MonteRay test file directory in standard locations. " )
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
endif()

if( MonteRayTestFiles_location )
    message( "-- FindMonteRayTestFiles.cmake -- Found MonteRay test files at [ ${MonteRayTestFiles_location} ]" )
#    add_definitions( -DMONTERAY_TESTFILES_DIR="${MonteRayTestFiles_location}" )
else()
    message( FATAL_ERROR "-- FindMonteRayTestFiles.cmake -- Could not locate MonteRay test files." )
endif()

macro(create_lnk3dnt_symlink target)
  add_custom_command( TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${lnk3dnt_location} lnk3dnt)
endmacro()

macro(create_MonteRayTestFiles_symlink target)
  add_custom_command( TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${MonteRayTestFiles_location} MonteRayTestFiles)
endmacro()

