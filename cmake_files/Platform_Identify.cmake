function(Platform_Identify)

#######################################################################
# Determine host name
#--------------------------------

find_path( MCATK_PROJECT_ROOT mcatk
           PATHS /home/xshares
                 /usr/projects
                 /usr/gapps
                 /projects
                 $ENV{HOME}
           PATH_SUFFIXES PROJECTS
           DOC "MCATK project directory containing user files, repository and pre-built libraries." )
if( NOT MCATK_PROJECT_ROOT )
    message( FATAL_ERROR "Unable to locate MCATK project directory on this system." )
endif()

string( REGEX MATCH "xshares" isXDiv ${MCATK_PROJECT_ROOT} )
if( isXDiv AND DEFINED ENV{PlatformOS} )
    set( Platform $ENV{PlatformOS} CACHE INTERNAL "Platform on which this was configured." )
    return()
endif()

if( DEFINED ENV{SYS_TYPE} ) # Check for LLNL HPC system
    if( $ENV{SYS_TYPE} STREQUAL "bgqos_0" )
        set( Platform BlueGeneQ CACHE INTERNAL "LLNL big iron machine" )
        find_program( SRUN srun DOC "Path to the SLURM srun executable" )
        set( SRUN_SERIAL ${SRUN} --partition=pdebug -n 1 -t 1:30:00 CACHE DOC "Required invocation line on BlueGeneQ" )
    else()
        set( Platform $ENV{SYS_TYPE} CACHE INTERNAL "LLNL TLCC machine." )
        set( DefaultQ drlanl )
    endif()
elseif( NOT isXDiv AND DEFINED ENV{MODULEPATH} ) # Check for LANL HPC system
    set( BigIronNames "(pinto|lobo|wolf|rry|RRZ|typhoon|cielito|cielo|moonlight|luna|trinitite|trinity)" )
    string( REGEX MATCH ${BigIronNames} BigIron $ENV{MODULEPATH} )
    if( DEFINED BigIron )
      set( Platform ${BigIron} CACHE  INTERNAL "Holds Platform we are configuring on.")
      message( STATUS "FOUND: PLATFORM being used is [ ${Platform} ]" )
    else()
      message( FATAL_ERROR "Unable to determine host machine from MODULEPATH.  Expected one of ${BigIronNames}" )
    endif()
else()
    if( NOT CMAKE_SYSTEM_NAME )
      SITE_NAME( LocalPlatform_temp )
    else()
      set( LocalPlatform_temp ${CMAKE_SYSTEM_NAME} )
    endif()

    string( REGEX REPLACE "(.*)\\.(.*)\\.(.*)" "\\1" LocalPlatform ${LocalPlatform_temp} )
    set( Platform ${LocalPlatform} CACHE INTERNAL "Holds Platform we are configuring on.")
    message( STATUS "FOUND: PLATFORM being used is [ ${Platform} ]." )
endif()

endfunction(Platform_Identify)
