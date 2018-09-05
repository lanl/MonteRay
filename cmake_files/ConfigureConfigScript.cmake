# - Script for configuring and installing the mcatk-config script
#
#   mcatk-config provides a sh based interface to provide info on the
#       MCATK installation
#
#   mcatk-config is generated from a template file
#
#   mcatk-config is relocatable for the install tree
#
#
#-----------------------------------------------------------------------
#
# function get_system_include_dirs
#          return list of directories our C++ compiler searches
#          by default.
#
#          The idea comes from CMake's inbuilt technique to do this
#          for the Eclipse and CodeBlocks generators, but we implement
#          our own function because the CMake functionality is internal
#          so we can't rely on it.
function(get_system_include_dirs _dirs)
  # Only for GCC, Clang and Intel
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES GNU OR "${CMAKE_CXX_COMPILER_ID}" MATCHES Clang OR "${CMAKE_CXX_COMPILER_ID}" MATCHES Intel)
    # Proceed
    file(WRITE "${CMAKE_BINARY_DIR}/CMakeFiles/g4dummy" "\n")

    # Save locale, them to "C" english locale so we can parse in English
    set(_orig_lc_all      $ENV{LC_ALL})
    set(_orig_lc_messages $ENV{LC_MESSAGES})
    set(_orig_lang        $ENV{LANG})

    set(ENV{LC_ALL}      C)
    set(ENV{LC_MESSAGES} C)
    set(ENV{LANG}        C)

    execute_process(COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} -v -E -x c++ -dD g4dummy
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/CMakeFiles
      ERROR_VARIABLE _cxxOutput
      OUTPUT_VARIABLE _cxxStdout
      )

    file(REMOVE "${CMAKE_BINARY_DIR}/CMakeFiles/g4dummy")

    # Parse and extract search dirs
    set(_resultIncludeDirs )
    if( "${_cxxOutput}" MATCHES "> search starts here[^\n]+\n *(.+ *\n) *End of (search) list" )
      string(REGEX MATCHALL "[^\n]+\n" _includeLines "${CMAKE_MATCH_1}")
      foreach(nextLine ${_includeLines})
        string(REGEX REPLACE "\\(framework directory\\)" "" nextLineNoFramework "${nextLine}")
        string(STRIP "${nextLineNoFramework}" _includePath)
        list(APPEND _resultIncludeDirs "${_includePath}")
      endforeach()
    endif()

    # Restore original locale
    set(ENV{LC_ALL}      ${_orig_lc_all})
    set(ENV{LC_MESSAGES} ${_orig_lc_messages})
    set(ENV{LANG}        ${_orig_lang})

    set(${_dirs} ${_resultIncludeDirs} PARENT_SCOPE)
  else()
    set(${_dirs} "" PARENT_SCOPE)
  endif()
endfunction()



#==============================================================================#
# We do not need to do this if the system is Windows
#
if(UNIX)
    Get_System_Include_Dirs(_cxx_compiler_dirs)

    set(packages Boost MPI ndatk Loki TBB)
    
    foreach(_package ${packages})
        if(${_package}_FOUND)
            set(MCATK_BUILTWITH_${_package} "yes")
        else()
            set(MCATK_BUILTWITH_${_package} "no")
        endif()
    endforeach()
    
    #-------------------------------------------------------------------------#
    # External Packages
    set(package_include_dirs)
    set(package_library_dirs)
    set(package_libraries)
    foreach(_package ${packages})
        if(${_package}_FOUND)
            list(APPEND package_include_dirs ${${_package}_INCLUDE_DIRS})
            list(APPEND package_library_dirs ${${_pacakge}_LIBRARY_DIRS})
            
            # TBB needs a little special treatment
            if("${_package}" STREQUAL "TBB")
                #list(REMOVE_ITEM package_libraries optimized)
                #list(REMOVE_ITEM package_libraries debug)
                set(_tag "RELEASE")
                if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
                    set(_tag "DEBUG")
                endif()
                list(APPEND package_libraries ${TBB_LIBRARY_${_tag}})
                if(TBB_MALLOC_FOUND)
                    list(APPEND package_libraries ${TBB_MALLOC_LIBRARY_${_tag}})
                endif()
            else()
                list(APPEND package_libraries ${${_package}_LIBRARIES})        
            endif()
            
        else()
            message(STATUS "Package ${_package} NOT FOUND for mcatk-config")
        endif()
    endforeach() 
    
    set(MCATK_PACKAGE_INCLUDE_DIRS )
    foreach(_dir ${package_include_dirs})
        set(MCATK_PACKAGE_INCLUDE_DIRS "${MCATK_PACKAGE_INCLUDE_DIRS} ${_dir}")
    endforeach()

    set(MCATK_PACKAGE_LIBRARY_DIRS )
    foreach(_dir ${package_library_dirs})
        set(MCATK_PACKAGE_LIBRARY_DIRS "${MCATK_PACKAGE_LIBRARY_DIRS} ${_dir}")
    endforeach()

    set(MCATK_PACKAGE_LIBRARIES )
    foreach(_lib ${package_libraries})
        set(MCATK_PACKAGE_LIBRARIES "${MCATK_PACKAGE_LIBRARIES} ${_lib}")
    endforeach()
    
    #-------------------------------------------------------------------------#
    # MCATK Libraries    
    set(_library_list ${Toolkit_Libs})
    set(MCATK_LIBRARY_LIST)
    foreach(_lib ${Toolkit_libs})
        set(MCATK_LIBRARY_LIST "${MCATK_LIBRARY_LIST} ${_lib}")
    endforeach()
    set(MCATK_LIBRARY_LIST "\"${MCATK_LIBRARY_LIST}\"")

    #-------------------------------------------------------------------------#
    # Flags
    set(MCATK_CONFIG_FLAGS "${MCATK_FLAGS}")
    set(MCATK_CONFIG_DEFINITIONS )
    foreach(_def ${MCATK_DEFS})
        set(_str "")
        if(NOT "${MCATK_CONFIG_DEFINTIONS}" STREQUAL "")
            set(_str "${MCATK_CONFIG_DEFINTIONS} ")
        endif()
        set(MCATK_CONFIG_DEFINTIONS "${str}-D${_def}")
    endforeach()
    
    #--------------------------------------------------------------------------#
    # Build tree (NOT RECOMMENDED FOR USE)
    # - This is not really needed and is not perfect because MCATK
    #   does not redirect the libraries to one locations so setting
    #   the link path would be very difficult
    if(BUILDTREE_MCATK_CONFIG)
        set(MCATK_CONFIG_SELF_LOCATION "# BUILD TREE IS NON-RELOCATABLE")
        set(MCATK_CONFIG_INSTALL_PREFIX "${PROJECT_BINARY_DIR}")
        set(MCATK_CONFIG_INSTALL_EXECPREFIX \"\")
        set(MCATK_CONFIG_LIBDIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
        set(MCATK_CONFIG_INCLUDE_DIRS ${MCATK_INCLUDE_DIRS})
        set(MCATK_CONFIG_EXTERNAL_LIBDIR "${MCATK_PACKAGE_LIBRARY_DIRS}")
        set(MCATK_CONFIG_EXTERNAL_INCLUDE_DIRS "${MCATK_PACKAGE_INCLUDE_DIRS}")
        set(MCATK_CONFIG_EXTERNAL_LIBRARIES "${MCATK_PACKAGE_LIBRARIES}")
        
        get_property(_mcatk_buildtree_include_dirs GLOBAL PROPERTY MCATK_BUILDTREE_INCLUDE_DIRS)
    
        #message(STATUS "MCATK INCLUDE DIRS : ${MCATK_INCLUDE_DIRS}")
    
        if(${CMAKE_VERSION} VERSION_GREATER 2.7)
            configure_file( ${CMAKE_SOURCE_DIR}/cmake_files/templates/mcatk-config.in
                            ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/mcatk-config
                            @ONLY
            )
        
            file(   COPY
                    ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/mcatk-config
                    DESTINATION ${PROJECT_BINARY_DIR}
                    FILE_PERMISSIONS
                        OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE
                        WORLD_READ WORLD_EXECUTE
            )
        else()
            configure_file( ${CMAKE_SOURCE_DIR}/cmake_files/templates/mcatk-config.in
                            ${PROJECT_BINARY_DIR}/mcatk-config
                            @ONLY
            )
        endif()
    endif(BUILDTREE_MCATK_CONFIG)
    
    #--------------------------------------------------------------------------#
    # Install tree
    #
    # Calculate base of self-contained install
    file(RELATIVE_PATH _bin_to_prefix ${CMAKE_INSTALL_PREFIX}/${binary_install_prefix} ${CMAKE_INSTALL_PREFIX})
    # Strip any trailing path separators    
    string(REGEX REPLACE "[/\\]$" "" _bin_to_prefix "${_bin_to_prefix}")
    
    set(MCATK_CONFIG_INSTALL_PREFIX "$scriptloc/${_bin_to_prefix}")
    set(MCATK_CONFIG_INSTALL_EXECPREFIX \"\")
    set(MCATK_CONFIG_LIBDIR "\${prefix}/${library_install_prefix}")
    set(MCATK_CONFIG_INCLUDE_DIRS "\${prefix}/include")
    set(MCATK_CONFIG_EXTERNAL_LIBDIR "${MCATK_PACKAGE_LIBRARY_DIRS}")
    set(MCATK_CONFIG_EXTERNAL_INCLUDE_DIRS "${MCATK_PACKAGE_INCLUDE_DIRS}")
    set(MCATK_CONFIG_EXTERNAL_LIBRARIES "${MCATK_PACKAGE_LIBRARIES}")

    configure_file( ${CMAKE_SOURCE_DIR}/cmake_files/templates/mcatk-config.in
                    ${PROJECT_BINARY_DIR}/InstallTreeFiles/mcatk-config
                    @ONLY
    )
      
    install(FILES ${PROJECT_BINARY_DIR}/InstallTreeFiles/mcatk-config
            DESTINATION ${CMAKE_INSTALL_PREFIX}/${binary_install_prefix}
            PERMISSIONS
                OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE
    )

endif(UNIX)


