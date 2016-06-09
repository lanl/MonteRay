#
# Try to find MCATK include dirs and libraries via mcatk-config
# Usage is as follows:
#
#   - In below, 1.0 is the minimum version number.
#   - COMPONENTS ALL links the shared mcatk library that contains all the components
#   - COMPONENTS <...specific libraries...> searches for specific libraries
#           (generally not recommended yet since this introduces link order dependencies)
#   - If no components are specified, only not libraries are searched for
#     (i.e. only the include path is searched for)
#
#   find_package(MCATK 1.0 COMPONENTS <ALL, Particle Utilities
#                                           ContinuousNeutronXS FreeGas
#                                           ENDF_Laws Reader_ACE Bank ProbInfo
#                                           Alpha TimeDependent StaticK
#                                           OutputControls Geometry lnk3dnt
#                                           GeometryInfo Math_Functions Source
#                                           MaterialSetup Isotope Restart
#                                           Tallies ContinuousPhotonXS IPComm >)
#
#   if(MCATK_FOUND)
#       include_directories(${MCATK_INCLUDE_DIRS} <${MCATK_EXTERNAL_INCLUDE_DIRS}>)
#       add_executable(foo foo.c)
#       target_link_libraries(foo ${MCATK_LIBRARIES} <${MCATK_EXTERNAL_LIBRARIES}>)
#   endif()
#
#
#
#   MCATK uses several external libraries: Boost, MPI, ndatk, Loki, and (optionally) TBB
#   The include directories and libraries used for the MCATK installation can be
#   called directly (i.e. MCATK_EXTERNAL_INCLUDE_DIRS and MCATK_EXTERNAL_LIBRARIES)
#   or the packages can be found by the user if this fails to be satisfactory
#

#
# Optional environment Variables/CMake Variable (can be either)
# to assist in locating MCATK:
#
#   MCATK_ROOT or MCATK_DIR     Preferred installation prefix for MCATK
#
#   MCATK_INCLUDEDIR            Set this to include directory of MCATK
#
#   MCATK_LIBRARYDIR            Set this to lib directory of MCATK
#
#   MCATK_SYSTEM_NAME           Set this to system of installation (e.g. Linux)
#
#   MCATK_COMPILER_NAME         Set this to compiler of installation (e.g. gnu-4.7)
#
#   MCATK_CONFIG_FILEPATH       Set this to the full path of mcatk-config
#

#
# Variables defined by this module:
#
#   MCATK_ROOT_DIR              The root directory of MCATK installation (cached)
#
#   MCATK_FOUND                 Means MCATK include directory was found
#
#   MCATK_CONFIG_FILE           The mcatk-config executable (full path, cached)
#
#   MCATK_INCLUDE_DIRS          MCATK include directories (not cached)
#
#   MCATK_INCLUDE_DIR           MCATK include directories (cached)
#
#   MCATK_LIBRARY               Same as MCATK_LIBRARIES (cached)
#
#   MCATK_LIBRARIES             Link to these MCATK libraries that are specified (not cached)
#
#   MCATK_LIBRARY_DIRS          The path to where the MCATK libraries are (not cached)
#
#   MCATK_VERSION               The version number of the MCATK package found
#
#   MCATK_MAJOR_VERSION         The major version number of the MCATK package found
#   MCATK_MINOR_VERSION         The minor version number of the MCATK package found
#   MCATK_PATCH_VERSION         The patch version number of the MCATK package found
#
#   MCATK_SHARED_LIBRARY        The shared library (cached)
#
#   MCATK_SHARED_LIBRARIES      The shared library (not cached)
#
#
#
# For each component specified in find_package(), the following variables are set:
#
#   MCATK_${COMPONENT}_FOUND    True if MCATK "component"
#
#   MCATK_${COMPONENT}_LIBRARY  Contains the libraries for the specified MCATK
#                               "component"
#
# 
# EXTERNALS - The following flags are also set for external libraries and includes
#
#
#   MCATK_EXTERNAL_INCLUDE_DIR  MCATK include directories of external packages (cached)
#
#   MCATK_EXTERNAL_INCLUDE_DIRS MCATK include directories of external packages (not cached)
#     
#   MCATK_EXTERNAL_LIBRARY      Link to these MCATK mcatk libraries (cached)
#
#   MCATK_EXTERNAL_LIBRARIES    Link to these MCATK mcatk libraries (not cached)
#
#
#
#
#   Author(s):
#      Jonathan R. Madsen (Los Alamos National Laboratory) - June 2014
#
#

include(SelectLibraryConfigurations)
include(FindPackageHandleStandardArgs)

#message(STATUS "")
#message(STATUS "")

# Default recommendation if a FATAL_ERROR occurs
set(DEFAULT_RECOMMEND_MSG "Please set/update (via -D<variable>=... in CMake or environment -- former has precedence over latter): MCATK_ROOT, MCATK_SYSTEM_NAME, and MCATK_COMPILER_NAME or point to mcatk-config via MCATK_CONFIG_FILEPATH or putting mcatk-config's folder in the evnironment PATH")

#------------------------------------------------------------------------------#
# The primary (non-component) variables that are cached
#------------------------------------------------------------------------------#
set(VARS_TO_CACHE
        MCATK_ROOT_DIR
        MCATK_CONFIG_FILE
        MCATK_INCLUDE_DIR
        MCATK_LIBRARY
)

#------------------------------------------------------------------------------#
# Macro to unset cached variables taking optional arguments to exclude from
# being unset
#------------------------------------------------------------------------------#
macro(RemoveCachedVariables)
    
    set(_explicit )
    # Variables to exclude
    foreach(_arg ${ARGN})
        list(APPEND _explicit ${_arg})
    endforeach()
        
    foreach(_var ${VARS_TO_CACHE})
        list(FIND _explicit ${_var} _index)
        if(NOT "${_index}" STREQUAL "-1")
            unset(${_var} CACHE)
        endif()
    endforeach()
        
endmacro()


#------------------------------------------------------------------------------#
# Function for check if a variable is defined in CMake command line or
# if defined as an environment variable
#------------------------------------------------------------------------------#
function(CheckForSpecifiedVariables _define _variable)
    # CMake variables have priority
    if(DEFINED ${_variable})
        if(NOT MCATK_FIND_QUIETLY)
            message(STATUS "Variable ${_variable} was defined via CMake [ -D${_variable}=... ]")
        endif()
        set(${_define} ${${_variable}} PARENT_SCOPE)
    # Check for environment variable if not defined as CMake variable
    elseif(DEFINED ENV{${_variable}})
        if(NOT MCATK_FIND_QUIETLY)
            message(STATUS "Variable ${_variable} was defined via environment")
        endif()
        set(${_define} $ENV{${_variable}} PARENT_SCOPE)
    endif()
    # if neither of above was satified the _define variable isn't altered

endfunction()

#------------------------------------------------------------------------------#
# Macro for setting cached variables if defined
#------------------------------------------------------------------------------#
macro(CacheIfDefined _cached_name _variable _cache_type _doc_string _force)
    if(DEFINED ${_variable})
        set(${_cached_name} ${${_variable}} CACHE ${_cache_type} "${_doc_string}" ${_force})
    endif()
endmacro()

#------------------------------------------------------------------------------#
# Macro for unsetting cached variables
#------------------------------------------------------------------------------#
macro(UnsetIfSet _variable)
    if(DEFINED ${_variable})
        unset(${_variable} CACHE)
    endif()
endmacro()

#------------------------------------------------------------------------------#
# Macro for checking if file exists
#------------------------------------------------------------------------------#
macro(CheckFileExists _id _name _path)
    set(_file_path "${${_path}}/${${_name}}")
    if(EXISTS "${_file_path}")
        set(${_id}_EXISTS TRUE)
    else()
        set(${_id}_EXISTS FALSE)
    endif()
    unset(_file_path)
endmacro()

#------------------------------------------------------------------------------#
# Function for doing for loop
# Must use operating system commands, not CMake functions/macros
#------------------------------------------------------------------------------#
function(forloop _start _end _increment _output _function _argument)
    set(i ${_start})
    #message(STATUS "FORLOOP : ${_start} ${_end} ${_increment}")
    
    set(_arg ${_argument})
    #message(STATUS "ARG : ${_arg}")
    
    # if the increment is positive
    if(${_increment} GREATER 0)
    
       while(${i} LESS ${_end})
            execute_process(COMMAND ${_function} ${_arg}
                            OUTPUT_VARIABLE _arg
                            ERROR_VARIABLE _arg
            )
            string(REPLACE "\n" "" _arg ${_arg})
            math(EXPR i "${i} + ${_increment}")
        endwhile()
        
    # if the increment is negative    
    elseif(${_increment} LESS 0)
    
        while(${i} GREATER ${_end} OR ${i} EQUAL ${_end})
            execute_process(COMMAND ${_function} ${_arg}
                            OUTPUT_VARIABLE _arg
                            ERROR_VARIABLE _arg
                            OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            math(EXPR i "${i} ${_increment}")
        endwhile()
    # if increment is zero --> infinite loop
    else()
        message(FATAL_ERROR "For loop function cannot have in increment of zero")    
    endif()
    # Set output in PARENT_SCOPE
    set(${_output} ${_arg} PARENT_SCOPE)

endfunction()

#------------------------------------------------------------------------------#
# Extract root dir, system, and compiler
#------------------------------------------------------------------------------#
function(GET_ROOT_SYS_COMPILER_FROM_CONFIG _root_output _sys_output _compiler_output _config)
    set(_config_file_root ${SPECIFIC_MCATK_CONFIG_FILE})
    # Move up to the root
    forloop(0 4 1 _config_file_root dirname ${_config} ${ARGN})
    # Move up one folder
    forloop(0 1 1 _config_file_compiler dirname ${_config} ${ARGN})
    # Move up two folders
    forloop(0 2 1 _config_file_system dirname ${_config} ${ARGN})
    # Get the name of the folder
    get_filename_component(_config_file_compiler "${_config_file_compiler}" NAME)
    # Get the name of the folder
    get_filename_component(_config_file_system "${_config_file_system}" NAME)
    # set variables in PARENT_SCOPE
    set(${_root_output} ${_config_file_root} PARENT_SCOPE)
    set(${_sys_output} ${_config_file_system} PARENT_SCOPE)  
    set(${_compiler_output} ${_config_file_compiler} PARENT_SCOPE)
endfunction()

#------------------------------------------------------------------------------#
# Check if variable changed
#------------------------------------------------------------------------------#
function(CheckIfChanged _prefix _expected _variable)
    # If ${_expected} (cached value) or ${_variable} (temporary) is not defined
    # then we can't tell if it changed

    if(NOT DEFINED ${_expected} OR "${${_expected}}" STREQUAL "")
        set(${_prefix}_CHANGED TRUE PARENT_SCOPE)
    # If ${_variable} (temporary) is not defined
    elseif(NOT DEFINED ${_variable} OR "${${_variable}}" STREQUAL "")
        set(${_prefix}_CHANGED FALSE PARENT_SCOPE)
    # If both defined but the same
    elseif("${${_expected}}" STREQUAL "${${_variable}}")
        set(${_prefix}_CHANGED FALSE PARENT_SCOPE)
    # If both defined but different
    else()
        set(${_prefix}_CHANGED TRUE PARENT_SCOPE)
    endif()

    #if(NOT DEFINED ${_expected} OR "${${_expected}}" STREQUAL "" OR NOT DEFINED ${_variable} OR "${${_variable}}" STREQUAL "")
    #    set(${_prefix}_CHANGED FALSE PARENT_SCOPE)
    
    
endfunction()


#------------------------------------------------------------------------------#
# temporaries set to nothing
#------------------------------------------------------------------------------#
set(SPECIFIC_MCATK_ROOT )
set(SPECIFIC_MCATK_INCLUDEDIR )
set(SPECIFIC_MCATK_LIBRARYDIR )
set(SPECIFIC_MCATK_SYSTEM )
set(SPECIFIC_MCATK_COMPILER )
set(SPECIFIC_MCATK_CONFIG_FILE )

#------------------------------------------------------------------------------#
# Check for CMake/Environment Variables
#------------------------------------------------------------------------------#
# check for MCATK_ROOT first
CheckForSpecifiedVariables(SPECIFIC_MCATK_ROOT MCATK_ROOT)
# check MCATK_DIR if MCATK_ROOT wasn't specified
if(NOT DEFINED SPECIFIC_MCATK_ROOT)
    CheckForSpecifiedVariables(SPECIFIC_MCATK_ROOT MCATK_DIR)
endif()
# check for MCATK_INCLUDEDIR
CheckForSpecifiedVariables(SPECIFIC_MCATK_INCLUDEDIR MCATK_INCLUDEDIR)
# check for MCATK_LIBRARYDIR
CheckForSpecifiedVariables(SPECIFIC_MCATK_LIBRARYDIR MCATK_LIBRARYDIR)
# check for MCATK_SYSTEM
CheckForSpecifiedVariables(SPECIFIC_MCATK_SYSTEM MCATK_SYSTEM_NAME)
# check for MCATK_COMPILER
CheckForSpecifiedVariables(SPECIFIC_MCATK_COMPILER MCATK_COMPILER_NAME)
# check for MCATK_CONFIG_FILE
CheckForSpecifiedVariables(SPECIFIC_MCATK_CONFIG_FILE MCATK_CONFIG_FILEPATH)

#message(STATUS "SPECIFIC MCATK ROOT DIR    : ${SPECIFIC_MCATK_ROOT}")
#message(STATUS "SPECIFIC MCATK INCLUDEDIR  : ${SPECIFIC_MCATK_INCLUDEDIR}")
#message(STATUS "SPECIFIC MCATK LIBRARYDIR  : ${SPECIFIC_MCATK_LIBRARYDIR}")
#message(STATUS "SPECIFIC MCATK SYSTEM      : ${SPECIFIC_MCATK_SYSTEM}")
#message(STATUS "SPECIFIC MCATK COMPILER    : ${SPECIFIC_MCATK_COMPILER}")
#message(STATUS "SPECIFIC MCATK CONFIG FILE : ${SPECIFIC_MCATK_CONFIG_FILE}")

CheckIfChanged(config_file MCATK_CONFIG_FILE SPECIFIC_MCATK_CONFIG_FILE)

# Let the user know if the config file changed but only if MCATK_CONFIG_FILE was already defined and different from one specified
if(NOT MCATK_FIND_QUIETLY AND config_file_CHANGED AND DEFINED MCATK_CONFIG_FILE AND NOT "${SPECIFIC_MCATK_CONFIG_FILE}" STREQUAL "${MCATK_CONFIG_FILE}")
    message(STATUS "Filepath of mcatk-config has changed. Attempting to update FindMCATK variables accordingly...")
endif()

CacheIfDefined(MCATK_ROOT_DIR       SPECIFIC_MCATK_ROOT         PATH     "MCATK Installation Root Directory" FORCE)
CacheIfDefined(MCATK_INCLUDE_DIR    SPECIFIC_MCATK_INCLUDEDIR   PATH     "MCATK Installation Include Directory" FORCE)
CacheIfDefined(MCATK_SYSTEM         SPECIFIC_MCATK_SYSTEM       STRING   "MCATK Platform" FORCE)
CacheIfDefined(MCATK_COMPILER       SPECIFIC_MCATK_COMPILER     STRING   "MCATK Compiler and Compiler version (e.g. gnu-4.6, Clang-2.9)" FORCE)
CacheIfDefined(MCATK_CONFIG_FILE    SPECIFIC_MCATK_CONFIG_FILE  FILEPATH "Full exe path of mcatk-config" FORCE)


#------------------------------------------------------------------------------#
# Extract ROOT_DIR, SYSTEM, and COMPILER if changed
#------------------------------------------------------------------------------#
set(RESET_COMPONENT_FOUND_CACHE FALSE)
set(_extract_root_sys_compiler TRUE)
set(_compose_config_from_root_sys_compiler FALSE)


if(DEFINED MCATK_CONFIG_FILE AND DEFINED MCATK_ROOT_DIR AND DEFINED MCATK_SYSTEM AND DEFINED MCATK_COMPILER AND NOT config_file_CHANGED)
    if(NOT "${MCATK_ROOT_DIR}" STREQUAL "" AND NOT "${MCATK_SYSTEM}" STREQUAL "" AND NOT "${MCATK_COMPILER}" STREQUAL "") 
        set(_extract_root_sys_compiler FALSE)
    endif()
endif()

if(NOT DEFINED MCATK_CONFIG_FILE OR "${MCATK_CONFIG_FILE}" STREQUAL "" OR "${MCATK_CONFIG_FILE}" STREQUAL "MCATK_CONFIG_FILE-NOTFOUND")
    if(DEFINED MCATK_ROOT_DIR AND DEFINED MCATK_SYSTEM AND DEFINED MCATK_COMPILER AND NOT "${MCATK_ROOT_DIR}" STREQUAL "" AND NOT "${MCATK_SYSTEM}" STREQUAL "" AND NOT "${MCATK_COMPILER}" STREQUAL "")
        set(_compose_config_from_root_sys_compiler TRUE)
    endif()
endif()

if(config_file_CHANGED)
    set(_extract_root_sys_compiler TRUE)
    set(RESET_COMPONENT_FOUND_CACHE TRUE)
endif()

#message(STATUS "--HERE-- ${_extract_root_sys_compiler}  ${_compose_config_from_root_sys_compiler}")
# Set root, sys, and compiler via mcatk-config path even if user tried specifying them
if(_extract_root_sys_compiler AND DEFINED MCATK_CONFIG_FILE)
    if(NOT MCATK_FIND_QUIETLY)
        message(STATUS "FindMCATK is extracting MCATK_ROOT, MCATK_SYSTEM, and MCATK_COMPILER FROM mcatk-config filepath (overrides any explict setting of these variables)")
    endif()
    set(_root )
    GET_ROOT_SYS_COMPILER_FROM_CONFIG(_root _sys _compiler ${MCATK_CONFIG_FILE})
    set(MCATK_ROOT_DIR ${_root} CACHE PATH "MCATK Installation Root Directory" FORCE)
    set(MCATK_SYSTEM ${_sys} CACHE STRING "MCATK Platform" FORCE)
    set(MCATK_COMPILER ${_compiler} CACHE STRING "MCATK Compiler and Version" FORCE)
    set(RESET_COMPONENT_FOUND_CACHE TRUE)
    RemoveCachedVariables(MCATK_LIBRARY)
elseif(_compose_config_from_root_sys_compiler)
    if(NOT MCATK_FIND_QUIETLY)
        message(STATUS "FindMCATK is extracting filepath of mcatk-config from MCATK_ROOT_DIR, MCATK_SYSTEM, and MCATK_COMPILER")
    endif()
    set(_config_path "${MCATK_ROOT_DIR}/bin/${MCATK_SYSTEM}/${MCATK_COMPILER}")
    set(MCATK_CONFIG_FILE ${_config_fpath} CACHE FILEPATH "Full exe path of mcatk-config" FORCE)
    set(RESET_COMPONENT_FOUND_CACHE TRUE)
    RemoveCachedVariables(MCATK_LIBRARY)
    CheckFileExists(_config_exe mcatk-config _config_path)
    if(_config_exe_EXISTS)
        set(MCATK_CONFIG_FILE "${MCATK_ROOT_DIR}/bin/${MCATK_SYSTEM}/${MCATK_COMPILER}/mcatk-config" CACHE FILEPATH "Full exe path of mcatk-config" FORCE)
    else()
        message(FATAL_ERROR "mcatk-config does not exist in path ${_config_path}. ${DEFAULT_RECOMMEND_MSG}")
    endif()
endif()

unset(_extract_root_sys_compiler)
unset(_compose_config_from_root_sys_compiler)



#------------------------------------------------------------------------------#
# MCATK Config File
# Here, we check if we need to find it and if so try to find it.
# If we need to find it because the root directory changed, we unset some
# cache variables
#
# BOOLEAN VARIABLES:
#       FIND_MCATK_CONFIG
#
#------------------------------------------------------------------------------#
# If we need to search for mcatk-config
set(FIND_MCATK_CONFIG_FILE TRUE)
if(DEFINED MCATK_CONFIG_FILE)
    set(FIND_MCATK_CONFIG_FILE FALSE)
endif()


if(FIND_MCATK_CONFIG_FILE)
    #--------------------------------------------------------------------------#
    # The MCATK installs use the following structure for the lib and bin directories
    #   - bin/
    #       - {SystemName}/
    #           - {compiler}-{compiler_major_version}.{compiler_minor_version}/
    #               - {EXECUTABLES}
    #   - lib/
    #       - {SystemName}/
    #           - {compiler}-{compiler_major_version}.{compiler_minor_version}/
    #               - {LIBS}
    #   - include/
    #       - mcatk/
    #           - {INCLUDE FILES}
    #
    #
    #   So we need to make the
    #   {SystemName}/{compiler}-{compiler_major_version}.{compiler_minor_version}/
    #   path
    #
    #--------------------------------------------------------------------------#

    #--------------------------------------------------------------------------#
    # Try to discern systems/available systems
    #--------------------------------------------------------------------------#
    function(MakeSystemsList)
        if(NOT DEFINED MCATK_SYSTEM)
            set(_SystemList )
            set(_cmake_sys_name ${CMAKE_SYSTEM_NAME})
            string( REGEX REPLACE "(.*)\\.(.*)\\.(.*)" "\\1" _cmake_sys_name ${_cmake_sys_name} )

            if(DEFINED ENV{SYS_TYPE})
                if( $ENV{SYS_TYPE} STREQUAL "bgqos_0" )
                    list(APPEND _SystemList BlueGeneQ)
                else()
                    list(APPEND _SystemList $ENV{SYS_TYPE})
                endif()
            elseif( DEFINED ENV{MODULEPATH} ) # Check for LANL HPC system
                set( BigIronNames "(pinto|lobo|roadrunner|rry|RRZ|typhoon|cielito|cielo|moonlight|luna|trinity|trinitite)" )
                string( REGEX MATCH ${BigIronNames} BigIron $ENV{MODULEPATH} )
                if( DEFINED BigIron )
                    list(APPEND _SystemList ${BigIron})
                else()
                    if(DEFINED ENV{SYS_TYPE})
                        list(APPEND _SystemList $ENV{SYS_TYPE})
                    endif()
                    list(APPEND _SystemList ${_cmake_sys_name})
                endif()
            else()
                list(APPEND _SystemList ${_cmake_sys_name})
            endif()
        else()
            set(_SystemList ${MCATK_SYSTEM})
        endif()

        set(SystemList ${_SystemList} PARENT_SCOPE)
    endfunction()

    #--------------------------------------------------------------------------#
    # Make available systems list
    #--------------------------------------------------------------------------#
    if(NOT DEFINED MCATK_SYSTEM OR "${MCATK_SYSTEM}" STREQUAL "")
        MakeSystemsList()
        set(MCATK_POSSIBLE_SYSTEMS ${SystemList})
        if(NOT DEFINED MCATK_POSSIBLE_SYSTEMS)
            list(APPEND MCATK_POSSIBLE_SYSTEMS Linux Apple)
        endif()
        unset(SystemList)
    endif()
    
    #--------------------------------------------------------------------------#
    # Compiler and version
    #--------------------------------------------------------------------------#
    if(NOT DEFINED MCATK_COMPILER OR "${MCATK_COMPILER}" STREQUAL "")
        # make CMAKE_CXX_COMPILER_ID first in list
        set(_compiler ${CMAKE_CXX_COMPILER_ID})
        # All compiler folders are lowercase except for Clang
        if(NOT "${_compiler}" STREQUAL "Clang")
            string(TOLOWER "${_compiler}" _compiler)
        endif()

        execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
                        OUTPUT_VARIABLE CXX_VERSION
                        OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        string( REGEX REPLACE "\\." ";" CXX_VERSION ${CXX_VERSION} )
        list( GET CXX_VERSION 0 _COMPILER_MAJOR_VERSION )
        list( GET CXX_VERSION 1 _COMPILER_MINOR_VERSION )
        set(MCATK_COMPILER_SUFFIXES "${_compiler}-${_COMPILER_MAJOR_VERSION}.${_COMPILER_MINOR_VERSION}")
        unset(_compiler)
        unset(CXX_VERSION)
        unset(_COMPILER_MAJOR_VERSION)
        unset(_COMPILER_MINOR_VERSION)
    else()
        set(MCATK_COMPILER_SUFFIXES ${MCATK_COMPILER})
    endif()

    # Maybe write this later to look in bin/lib directories of MCATK_ROOT_DIR
    # but finding the closest system, compiler, and compiler version match
    # is not particularly easy so not doing it right now unless it proves to
    # be valuable later
    #------------------------------------------------------------------------------#
    # Find closest match
    #------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------#
    #
    #------------------------------------------------------------------------------#


    #------------------------------------------------------------------------------#
    # Make Path suffix hints
    #------------------------------------------------------------------------------#
    set(MCATK_PATH_SUFFIX_HINTS )
    foreach(_system ${MCATK_POSSIBLE_SYSTEMS})
        list(APPEND MCATK_PATH_SUFFIXES "bin/${_system}/${MCATK_COMPILER_SUFFIXES}")
    endforeach()

    #------------------------------------------------------------------------------#
    # Find mcatk-config
    #------------------------------------------------------------------------------#
    find_program(MCATK_CONFIG_FILE
                 mcatk-config
                 HINTS
                    ${MCATK_ROOT_DIR}
                 PATHS
                    ENV PATH
                 PATH_SUFFIXES
                    ${MCATK_PATH_SUFFIXES}
                 DOC
                    "mcatk-config is used to extract include dirs and libraries"
    )
    # if mcatk-config is not found, don't attempt to use it... obviously
    if("${MCATK_CONFIG_FILE}" STREQUAL "MCATK_CONFIG_FILE-NOTFOUND" OR NOT EXISTS "${MCATK_CONFIG_FILE}")
        message(FATAL_ERROR "Unable to locate mcatk-config. ${DEFAULT_RECOMMEND_MSG}")
    else()
        GET_ROOT_SYS_COMPILER_FROM_CONFIG(_root _sys _compiler ${MCATK_CONFIG_FILE})
        CacheIfDefined(MCATK_ROOT_DIR _root PATH "MCATK Installation Root Directory" FORCE)
        CacheIfDefined(MCATK_SYSTEM _sys STRING "MCATK Platform" FORCE)
        CacheIfDefined(MCATK_COMPILER _compiler STRING "MCATK Compiler and Compiler version (e.g. gnu-4.6, Clang-2.9)" FORCE)
        set(RESET_COMPONENT_FOUND_CACHE TRUE)
        RemoveCachedVariables(MCATK_LIBRARY)   
    endif()

else()
    # Get PATH of specified mcatk-config
    get_filename_component(mcatk_config_folder ${MCATK_CONFIG_FILE} PATH)
    # Get NAME of specified mcatk-config (should always be mcatk-config, but do this to be generic)
    get_filename_component(mcatk_config_fname ${MCATK_CONFIG_FILE} NAME)
    # Check if mcatk-config exists, if it does mcatk_config_EXISTS = TRUE
    CheckFileExists(mcatk_config mcatk_config_fname mcatk_config_folder)
    # If it exists, add to cache, if not this is FATAL
    if(NOT mcatk_config_EXISTS)
        message(STATUS "MCATK_CONFIG_FNAME : ${mcatk_config_fname}")
        message(STATUS "MCATK_CONFIG_PATH : ${mcatk_config_folder}")
        message(STATUS "MCATK ROOT DIR      : ${MCATK_ROOT_DIR}")
        message(STATUS "MCATK SYSTEM        : ${MCATK_SYSTEM}")
        message(STATUS "MCATK COMPILER      : ${MCATK_COMPILER}")
        message(FATAL_ERROR "Specified mcatk-config file does not exist in path ${mcatk_config_folder}. ${DEFAULT_RECOMMEND_MSG}")
    endif()
    # unset variables that are temporary
    unset(mcatk_config_folder)
    unset(mcatk_config_fname)
    unset(mcatk_config_EXISTS)
endif()

message(STATUS "MCATK Config file: ${MCATK_CONFIG_FILE}")

#------------------------------------------------------------------------------#
# MACROS
#------------------------------------------------------------------------------#
# convert space separated string to list
macro(CONVERT_TO_LIST _string)
    string(REPLACE " " ";" ${_string} ${${_string}})
endmacro()
#------------------------------------------------------------------------------#
# execute mcatk-config
macro(EXECUTE_MCATK_CONFIG _argument _variable)
    execute_process(COMMAND ${MCATK_CONFIG_FILE} ${_argument}
                    OUTPUT_VARIABLE ${_variable}
                    ERROR_VARIABLE ${_variable}
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_STRIP_TRAILING_WHITESPACE
    )
endmacro()



#------------------------------------------------------------------------------#
# USE THE MCATK-CONFIG FILE
#------------------------------------------------------------------------------#
if(DEFINED MCATK_CONFIG_FILE AND EXISTS "${MCATK_CONFIG_FILE}")

    # MCATK ROOT DIR
    if(NOT DEFINED MCATK_ROOT_DIR OR "${MCATK_ROOT_DIR}" STREQUAL "")
        execute_mcatk_config("--prefix" MCATK_CONFIG_ROOT_DIR)
        set(MCATK_ROOT_DIR ${MCATK_CONFIG_ROOT_DIR} CACHE PATH "The root installation directory of MCATK")
    endif()

    # MCATK VERSION
    if(NOT DEFINED MCATK_VERSION OR "${MCATK_VERSION}" STREQUAL "")
        execute_mcatk_config("--version" MCATK_CONFIG_VERSION)
        string( REGEX REPLACE "\\." ";" MCATK_CONFIG_VERSION_AS_LIST ${MCATK_CONFIG_VERSION} )
        list( GET MCATK_CONFIG_VERSION_AS_LIST 0 MCATK_CONFIG_MAJOR_VERSION )
        list( GET MCATK_CONFIG_VERSION_AS_LIST 1 MCATK_CONFIG_MINOR_VERSION )
        list( GET MCATK_CONFIG_VERSION_AS_LIST 2 MCATK_CONFIG_PATCH_VERSION )

        set(MCATK_VERSION ${MCATK_CONFIG_VERSION} CACHE INTERNAL "MCATK Version")
        set(MCATK_MAJOR_VERSION ${MCATK_CONFIG_MAJOR_VERSION} CACHE INTERNAL "MCATK Major Version")
        set(MCATK_MINOR_VERSION ${MCATK_CONFIG_MINOR_VERSION} CACHE INTERNAL "MCATK Minor Version")
        set(MCATK_PATCH_VERSION ${MCATK_CONFIG_PATCH_VERSION} CACHE INTERNAL "MCATK Patch Version")

        unset(MCATK_CONFIG_VERSION)
        unset(MCATK_CONFIG_MAJOR_VERSION)
        unset(MCATK_CONFIG_MINOR_VERSION)
        unset(MCATK_CONFIG_PATCH_VERSION)
    endif()

    # Unset/set to NOTFOUND the component libraries
    if(RESET_COMPONENT_FOUND_CACHE)
        # Get list of libraries and convert to CMake list
        execute_mcatk_config("--list-libs" _lib_list)
        convert_to_list(_lib_list)
        # Loop over components and set to NOTFOUND
        foreach(_lib ${_lib_list})
            set(MCATK_${_lib}_LIBRARY "MCATK_${_lib}_LIBRARY-NOTFOUND" CACHE STRING "The ${_lib} library of MCATK" FORCE)
        endforeach()
        # Set shared library to not found
        set(MCATK_SHARED_LIBRARY "MCATK_SHARED_LIBRARY-NOTFOUND" CACHE STRING "The MCATK shared library containing all the components" FORCE)
        # unset temporary
        unset(_lib_list)
        # unset CACHE variables
        set(MCATK_LIBRARY )
        unset(MCATK_LIBRARY CACHE)
        unset(MCATK_FOUND_COMPONENTS CACHE)
    endif()
    
    # MCATK LIBRARIES
    if(NOT DEFINED MCATK_LIBRARY OR "${MCATK_LIBRARY}" STREQUAL "" OR RESET_COMPONENT_FOUND_CACHE)

        # Temporary for MCATK_LIBRARY
        set(_mcatk_library )
        # Get libs from mcatk-config
        execute_mcatk_config("--libs" MCATK_CONFIG_LIBS)
        # convert to list
        convert_to_list(MCATK_CONFIG_LIBS)

        set(_components )

        set(_find_all FALSE)
        foreach(_comp ${MCATK_FIND_COMPONENTS})
            if("${_comp}" STREQUAL "ALL")
                set(_find_all TRUE)
                execute_mcatk_config("--list-libs" MCATK_CONFIG_LIB_LIST)
                convert_to_list(MCATK_CONFIG_LIB_LIST)
                list(APPEND _components ${MCATK_CONFIG_LIB_LIST})
                # Remove all from components
                #list(REMOVE_ITEM MCATK_FIND_COMPONENTS ${_comp})
                # Add full list to MCATK_FIND_COMPONENTS
                #list(APPEND MCATK_FIND_COMPONENTS ${MCATK_CONFIG_LIB_LIST})
            else()
                list(APPEND _components ${_comp})
            endif()
        endforeach()

        if(DEFINED _components)
            list(REMOVE_DUPLICATES _components)
        endif()

        # Extract the library prefix(s)
        set(_lib_prefix )
        foreach(_lib ${MCATK_CONFIG_LIBS})
            string(REGEX MATCH "-L" _libdir_flag ${_lib})
            if(NOT "${_libdir_flag}" STREQUAL "")
                string(REPLACE "-L" "" _tmp_lib_prefix ${_lib})
                list(APPEND _lib_prefix ${_tmp_lib_prefix})
            endif()
        endforeach()

        find_library(MCATK_SHARED_LIBRARY
                     mcatk
                     HINTS
                        ${_lib_prefix}
                     PATHS
                        ${_lib_prefix}
                        ENV LD_LIBRARY_PATH
                        ENV DYLD_LIBRARY_PATH
                        ENV LIBRARY_PATH
                     DOC
                        "MCATK Shared library")
                
        if(NOT "${MCATK_SHARED_LIBRARY}" STREQUAL "MCATK_SHARED_LIBRARY-NOTFOUND")
            set(MCATK_LIBRARY ${MCATK_SHARED_LIBRARY} CACHE STRING "MCATK Shared library with all components")
            if(_find_all)
                message(STATUS "MCATK Shared library (contains all components) : ${MCATK_SHARED_LIBRARY}")
            endif()
        elseif(NOT _find_all)
            # Find the components
            foreach(_component ${_components})
    
                find_library(MCATK_${_component}_LIBRARY
                             ${_component}
                             HINTS
                                ${_lib_prefix}
                             PATHS
                                ${_lib_prefix}
                                ENV LD_LIBRARY_PATH
                                ENV DYLD_LIBRARY_PATH
                                ENV LIBRARY_PATH
                             DOC
                                "MCATK ${_component} Component library"
                    )
                    
                mark_as_advanced(MCATK_${_component}_LIBRARY)
                
                if(NOT "${MCATK_${_component}_LIBRARY}" STREQUAL "MCATK_${_component}_LIBRARY-NOTFOUND")
                    list(APPEND _mcatk_library ${MCATK_${_component}_LIBRARY})
                else()
                    if(${_component}_FIND_REQUIRED)
                        message(FATAL_ERROR "NOT FOUND ${_component} : ${MCATK_${_component}_LIBRARY}")
                    else()
                        message(WARNING "NOT FOUND ${_component} : ${MCATK_${_component}_LIBRARY}")                
                    endif()
                endif()
    
            endforeach()
    
            set(MCATK_LIBRARY ${_mcatk_library} CACHE STRING "MCATK LIBRARIES (cached)" FORCE)
            set(MCATK_FOUND_COMPONENTS ${_components} CACHE STRING "MCATK Library components found (cached)" FORCE)
            unset(_components)
            unset(MCATK_CONFIG_LIBS)
        endif()
    else()
        if(NOT MCATK_FIND_QUIETLY)
            message(STATUS "MCATK Component Libraries (cached): ")
            foreach(_component ${MCATK_FOUND_COMPONENTS})
                message(STATUS "\t ${_component}")
            endforeach()
        endif()
    endif()

    
    # MCATK INCLUDE DIRS
    if(NOT DEFINED MCATK_INCLUDE_DIR OR "${MCATK_INCLUDE_DIR}" STREQUAL "")

        # Temporary MCATK_INCLUDE_DIR
        set(_mcatk_include_dir )

        execute_mcatk_config("--cflags" MCATK_CONFIG_CFLAGS)
        convert_to_list(MCATK_CONFIG_CFLAGS)

        # Find Include directories
        set(_includedirs )
        foreach(_flag ${MCATK_CONFIG_CFLAGS})
            string(REGEX MATCH "-I" _incdir_flag ${_flag})
            if(NOT "${_incdir_flag}" STREQUAL "")
                string(REPLACE "-I" "" _inc_prefix ${_flag})
                #message(STATUS "_inc_prefix is ${_inc_prefix}")
                list(APPEND _includedirs ${_inc_prefix})
            endif()
        endforeach()

        if(DEFINED _includedirs)
            # we want to make the includes be of the form "#include <mcatk/SomeFile.txt>"
            # so remove mcatk from the end of include path
            # However, the current method does not account for things such as <mcatk/Algorithms/SomeFile.hh>
            # where ${_inc} is ${MCATK_ROOT_DIR}/include/mcatk/Algorithms
            # that might/could occur in the future
            foreach(_inc ${_includedirs})
                get_filename_component(_foldername ${_inc} NAME)
                if("${_foldername}" STREQUAL "mcatk")
                    get_filename_component(_path ${_inc} PATH)
                    list(REMOVE_ITEM _includedirs ${_inc})
                    list(APPEND _includedirs ${_path})
                endif()
            endforeach()
            list(APPEND _mcatk_include_dir ${_includedirs})
        endif()

        unset(_includedirs)
        unset(MCATK_CONFIG_CFLAGS)
        set(MCATK_INCLUDE_DIR ${_mcatk_include_dir} CACHE INTERNAL "MCATK INCLUDE DIRS (cached)")
    endif()

    # MCATK EXTERNAL INCLUDE DIRS
    if(NOT DEFINED MCATK_EXTERNAL_INCLUDE_DIR OR "${MCATK_EXTERNAL_INCLUDE_DIR}" STREQUAL "")

        # Temporary MCATK_INCLUDE_DIR
        set(_mcatk_external_include_dir )

        execute_mcatk_config("--external-cflags" MCATK_EXTERNAL_CONFIG_CFLAGS)
        convert_to_list(MCATK_EXTERNAL_CONFIG_CFLAGS)

        # Find Include directories
        set(_includedirs )
        foreach(_flag ${MCATK_EXTERNAL_CONFIG_CFLAGS})
            string(REGEX MATCH "-I" _incdir_flag ${_flag})
            if(NOT "${_incdir_flag}" STREQUAL "")
                string(REPLACE "-I" "" _inc_prefix ${_flag})
                #message(STATUS "_inc_prefix is ${_inc_prefix}")
                list(APPEND _includedirs ${_inc_prefix})
            endif()
        endforeach()

        if(DEFINED _includedirs)
            # we want to make the includes be of the form "#include <mcatk/SomeFile.txt>"
            # so remove mcatk from the end of include path
            # However, the current method does not account for things such as <mcatk/Algorithms/SomeFile.hh>
            # where ${_inc} is ${MCATK_ROOT_DIR}/include/mcatk/Algorithms
            # that might/could occur in the future
            foreach(_inc ${_includedirs})
                get_filename_component(_foldername ${_inc} NAME)
                if("${_foldername}" STREQUAL "mcatk")
                    get_filename_component(_path ${_inc} PATH)
                    list(REMOVE_ITEM _includedirs ${_inc})
                    list(APPEND _includedirs ${_path})
                endif()
            endforeach()
            list(APPEND _mcatk_external_include_dir ${_includedirs})
        endif()

        unset(_includedirs)
        unset(MCATK_EXTERNAL_CONFIG_CFLAGS)
        set(MCATK_EXTERNAL_INCLUDE_DIR ${_mcatk_external_include_dir} CACHE INTERNAL "MCATK EXTERNAL INCLUDE DIRS (cached)")
    endif()

    # MCATK EXTERNAL LIBRARIES
    if(NOT DEFINED MCATK_EXTERNAL_LIBRARY OR "${MCATK_EXTERNAL_LIBRARY}" STREQUAL "")

        # Temporary MCATK_EXTERNAL_LIBRARY
        set(_mcatk_external_lib )

        execute_mcatk_config("--external-libs" MCATK_EXTERNAL_CONFIG_LIBS)
        convert_to_list(MCATK_EXTERNAL_CONFIG_LIBS)

        # Add libraries
        set(_externallibs )
        foreach(_lib ${MCATK_EXTERNAL_CONFIG_LIBS})
            string(REGEX MATCH "-l" _lib_flag ${_lib})
            if(NOT "${_lib_flag}" STREQUAL "")
                string(REPLACE "-l" "" _lib_prefix ${_lib})
                list(APPEND _externallibs ${_lib_prefix})
            endif()
        endforeach()

        if(DEFINED _externallibs)
            foreach(_lib ${_externallibs})
                if(EXISTS "${_lib}")
                    list(APPEND _mcatk_external_lib ${_lib})            
                endif()
            endforeach()
        endif()

        unset(_externallibs)
        unset(MCATK_EXTERNAL_CONFIG_LIBS)
        set(MCATK_EXTERNAL_LIBRARY ${_mcatk_external_lib} CACHE INTERNAL "MCATK EXTERNAL LIBRARIES (cached)")
    endif()
        

else()
    message(FATAL_ERROR "MCATK was not found because mcatk-config [= \"${MCATK_CONFIG_FILE}\"] does not exist. ${DEFAULT_RECOMMEND_MSG}")
endif()

#------------------------------------------------------------------------------#
# Clean up
#------------------------------------------------------------------------------#
unset(SPECIFIC_MCATK_ROOT )
unset(SPECIFIC_MCATK_INCLUDEDIR )
unset(SPECIFIC_MCATK_LIBRARYDIR )
unset(SPECIFIC_MCATK_SYSTEM )
unset(SPECIFIC_MCATK_COMPILER )
unset(SPECIFIC_MCATK_CONFIG_FILE )
unset(DEFAULT_RECOMMEND_MSG )

#------------------------------------------------------------------------------#
#   Set MCATK_LIBRARIES and MCATK_INCLUDE_DIRS (not cached)
#------------------------------------------------------------------------------#
set(MCATK_LIBRARIES ${MCATK_LIBRARY})
set(MCATK_INCLUDE_DIRS ${MCATK_INCLUDE_DIR})
set(MCATK_LIBRARY_DIRS )
set(MCATK_EXTERNAL_LIBRARIES ${MCATK_EXTERNAL_LIBRARY})
set(MCATK_EXTERNAL_INCLUDE_DIRS ${MCATK_EXTERNAL_INCLUDE_DIR})

# Set MCATK_${_component}_FOUND
# Dont use HANDLE_COMPONENTS to do this because of ALL setting
foreach(_component ${MCATK_FOUND_COMPONENTS})
    FIND_PACKAGE_HANDLE_STANDARD_ARGS(MCATK_${_component} REQUIRED_VARS MCATK_${_component}_LIBRARY
                                      FAIL_MESSAGE "MCATK LIBARY ${_compoent} NOT FOUND - ${MCATK_ROOT_DIR} with platform ${MCATK_SYSTEM} and compiler ${MCATK_COMPILER}. mcatk-config was ${MCATK_CONFIG_FILE}"
    )
    # Get library full path
    set(_library ${MCATK_${_component}_LIBRARY})
    get_filename_component(_library_dir ${_library} REALPATH)
    get_filename_component(_library_dir ${_library_dir} PATH)
    list(APPEND MCATK_LIBRARY_DIRS ${_library_dir})
    # Clean up temporaries
    unset(_library)
    unset(_library_dir)
endforeach()

foreach(_lib ${MCATK_LIBRARIES})
    # Get library full path
    set(_library ${_lib})
    get_filename_component(_library_dir ${_library} REALPATH)
    get_filename_component(_library_dir ${_library_dir} PATH)
    list(APPEND MCATK_LIBRARY_DIRS ${_library_dir})
    # Clean up temporaries
    unset(_library)
    unset(_library_dir)
endforeach()

# Remove duplicates
if(MCATK_LIBRARY_DIRS)
    list(REMOVE_DUPLICATES MCATK_LIBRARY_DIRS)
endif()

# Set MCATK_FOUND
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MCATK REQUIRED_VARS MCATK_ROOT_DIR MCATK_INCLUDE_DIRS MCATK_CONFIG_FILE MCATK_SYSTEM MCATK_COMPILER
                                  VERSION_VAR MCATK_VERSION
                                  #HANDLE_COMPONENTS    # This returns an error sometimes because ALL is an option
                                  FAIL_MESSAGE "MCATK NOT FOUND - ${MCATK_ROOT_DIR} with platform ${MCATK_SYSTEM} and compiler ${MCATK_COMPILER}. mcatk-config was ${MCATK_CONFIG_FILE}"
)

if(NOT MCATK_FIND_QUIETLY)
    message(STATUS "")
    message(STATUS "MCATK FOUND         : ${MCATK_VERSION}")
    message(STATUS "MCATK ROOT DIR      : ${MCATK_ROOT_DIR}")
    message(STATUS "MCATK SYSTEM        : ${MCATK_SYSTEM}")
    message(STATUS "MCATK COMPILER      : ${MCATK_COMPILER}")
    message(STATUS "MCATK LIBRARY DIRS  : ${MCATK_LIBRARY_DIRS}")
    message(STATUS "MCATK CONFIG FILE   : ${MCATK_CONFIG_FILE}")
    #message(STATUS "MCATK INCLUDE DIRS  : ${MCATK_INCLUDE_DIRS}")
    #message(STATUS "MCATK LIBRARIES     : ${MCATK_LIBRARIES}")
    message(STATUS "")
endif()













