function( concatenateInstallSubdirs )

   if(NOT DEFINED Standalone ) 
   #if(NOT DEFINED Platform)
   #   message( FATAL_ERROR "Variable [ Platform ] not defined. Need to set it or call LocatePackagesDir first.")
   #endif()
   #if(NOT DEFINED compiler_install_prefix)
   #   message( FATAL_ERROR "Variable [ compiler_install_prefix ] not defined. Need to set it or call LocatePackagesDir first.")
   #endif()
   endif()

   if( CMAKE_BUILD_TYPE STREQUAL "Debug" )
     set(debugPrefix "/debug")
   else()
     set(debugPrefix "")
   endif()
   
   set(library_install_prefix lib/${debugPrefix} CACHE INTERNAL "Holds library install prefix path")
   set(binary_install_prefix  bin/${debugPrefix} CACHE INTERNAL "Holds library install prefix path")
   
   # Don't add specific sub-paths for MonteRay
   #set(library_install_prefix lib CACHE INTERNAL "Holds library install prefix path")
   #set(binary_install_prefix  bin CACHE INTERNAL "Holds library install prefix path")
   
   
   message( STATUS "${CMAKE_PROJECT_NAME} Installation Path Settings:" )
   message( STATUS "   Toolkit installation root is [ ${CMAKE_INSTALL_PREFIX} ]" )
   message( STATUS "   Library relative path is     [ ${library_install_prefix} ]" )
   message( STATUS "   Binary relative path is      [ ${binary_install_prefix} ]" )
   
endfunction(concatenateInstallSubdirs)
#================================================================================
# Function: appendGlobalList
# Append an element to a global list
function( appendGlobalList _listName _element )
    get_property( is_defined GLOBAL PROPERTY ${_listName} DEFINED )
    if(NOT is_defined)
      define_property(GLOBAL PROPERTY ${_listName}
                      BRIEF_DOCS "List of global elements of ${_listName}"
                      FULL_DOCS  "List of global elements of ${_listName}")
    endif()
    get_property( names GLOBAL PROPERTY ${_listName} )
    list( FIND names ${_element} location )
    if( location EQUAL -1 )
        set_property(GLOBAL APPEND PROPERTY ${_listName} "${_element}" )
    else()
        get_property( names GLOBAL PROPERTY ${_listName} )
        message( FATAL_ERROR "Element [ ${_element} ] was already present (${location}) in ${_listName} [ ${names} ]" )
    endif()
endfunction()
#--------------------------------------------------------------------------------

#================================================================================
# Function: setModuleName
# Create a name for an executable or library based on the current directory's name
function( setModuleName )
    get_filename_component( dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME )
    set( localName ${dirName} PARENT_SCOPE )
    appendGlobalList( KnownNames ${dirName} )
endfunction()
#--------------------------------------------------------------------------------

#================================================================================
# Function: addCMakeDirs
# Look in all the directories below the current one for CMakeLists.txt files.  If 
# any of the subdirectories contain such a file, register it with cmake (i.e. using
# add_subdirectory).  If the caller passes in arguments to this function, it is 
# assumed that these directories will be skipped even if they contain a CMakeLists.txt
function( addCMakeDirs )

    ########################################
    #  Subdirectories containing CMakeLists.txt files
    file( GLOB SubCMakes RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*/CMakeLists.txt" )

    foreach( subCMake ${SubCMakes} )
        # this strips off the "/CMakeLists.txt" part
        get_filename_component( dirName ${subCMake} PATH )
        # see if its in our to-ignore list
        list( FIND ARGV ${dirName} skipDirIndex )
        if( skipDirIndex EQUAL -1 )
            add_subdirectory( ${dirName} )
        else()
            message( "In directory [ ${localName} ] -- Skipping directory [ ${dirName} ]" )
        endif()

    endforeach()

endfunction()
#--------------------------------------------------------------------------------

#================================================================================
# Function: includeAllHeaders
# Look for all the directories below the current one for header files and add them to
# the list of included directories 
function( includeAllHeaders )
	#message( STATUS "%%%%%%%%%%DEBUG: Starting cmake_Files/GeneralFunctions :: includeAllHeaders ")
    file( GLOB_RECURSE AllHeaders "*.hh" "*.h")
    string( REPLACE ";" "|" SkipDirs "${ARGV}" )
    set( SkipDirs "(${SkipDirs})" )
    foreach( hdr ${AllHeaders} )
        string( REGEX MATCH ${SkipDirs} shouldSkip ${hdr} )
        if( shouldSkip )
#            message( "Skipping : ${hdr}" )
        else()
            get_filename_component( temp ${hdr} PATH )
            include_directories( ${temp} )
        endif()
    endforeach()
    
    file( GLOB_RECURSE AllGeneratedHeaders RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.hh.in" "*.h.in")
#    message( STATUS "%%%%%%%%%%DEBUG: cmake_Files/GeneralFunctions :: AllGeneratedHeaders=${AllGeneratedHeaders} ")
    string( REPLACE ";" "|" SkipDirs "${ARGV}" )
    set( SkipDirs "(${SkipDirs})" )
    foreach( hdr ${AllGeneratedHeaders} )
#    	message( STATUS "%%%%%%%%%%DEBUG: cmake_Files/GeneralFunctions :: includeAllHeaders :: generated header=${hdr} ")
        string( REGEX MATCH ${SkipDirs} shouldSkip ${hdr} )
        if( shouldSkip )
#            message( "Skipping : ${hdr}" )
        else()
            get_filename_component( temp ${hdr} PATH )
            set( fulltemppath "${CMAKE_CURRENT_BINARY_DIR}/${hdr}" )
            STRING(REPLACE ".in" "" modfulltemppath ${fulltemppath} )
            set( temppath "${CMAKE_CURRENT_BINARY_DIR}/${temp}" )
#            message( STATUS "including directory: ${temppath}" )
            include_directories( ${temppath} )
            install( FILES ${modfulltemppath} DESTINATION include )
        endif()
    endforeach()
    
endfunction()

function( globCudaFiles AllSkipDirs AllCudaFiles )
#    message( STATUS "%%%%%%%%%%DEBUG: Starting cmake_Files/GeneralFunctions :: globCudaFiles ")
    file( GLOB_RECURSE AllPossibleCudaFiles "*.cc" "*.c" "*.cu" "*.cpp" )
    string( REPLACE ";" "|" SkipDirs "${AllSkipDirs}" )
#    message( STATUS "%%%%%%%%%%DEBUG: AllSkipDirs=${AllSkipDirs} " )
#    message( STATUS "%%%%%%%%%%DEBUG: SkipDirs=${SkipDirs} " )
    set( SkipDirs "(${SkipDirs})" )
    foreach( cudaFile ${AllPossibleCudaFiles} )
        string( REGEX MATCH ${SkipDirs} shouldSkip ${cudaFile} )
        if( shouldSkip )
#            message( "Skipping : ${cudaFile}" )
        else()
#            message( STATUS "%%%%%%%%%%DEBUG: Starting cmake_Files/GeneralFunctions :: globCudaFiles - adding ${cudaFile} ")
            get_filename_component( temp ${cudaFile} PATH )
            list(APPEND CudaFileList ${cudaFile})
        endif()
    endforeach()
    
#    foreach( cudaFile ${CudaFileList} )
#       message( STATUS "%%%%%%%%%%DEBUG: Starting cmake_Files/GeneralFunctions :: globCudaFiles - adding ${cudaFile} ")
#    endforeach()
    set(${AllCudaFiles} ${CudaFileList} PARENT_SCOPE)
endfunction()

#--------------------------------------------------------------------------------

#================================================================================
# Function: UseMCATK
# Provide the tools for using a previously built toolkit library
# Locates and includes the MCATK main library and headers and then 
function( UseMCATK )

    set( installPaths ${MCATK_ReleaseDir} 
                      ${CMAKE_BINARY_DIR}/../${DefaultReleaseName} 
                      ${CMAKE_BINARY_DIR}/../../${DefaultReleaseName} )
    find_path( ToolkitInstallPath NAMES include/mcatk
               PATHS ${installPaths}
               DOC "Location of the installation directory for the toolkit"
             )
    if( NOT ToolkitInstallPath )
        message( WARNING "Toolkit not found in : [${CMAKE_BINARY_DIR}]" )
        unset( ToolkitInstallPath )
        set( searchPaths $ENV{BINARY_DIR} ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/.. ${CMAKE_BINARY_DIR}/../.. )
        find_path( ToolkitBuildPath NAMES src/libmcatk.so
                   PATHS ${searchPaths}
                   DOC "Location of the binary directory for the toolkit"
                 )
        if( NOT ToolkitBuildPath )
           message( FATAL_ERROR "Unable to locate toolkit binary files for installation. Searched in : "
                                 "[ ${searchPaths} ]" )
        endif()
        message( "Attempting to install toolkit resources from [ ${ToolkitBuildPath} ]..." )
        execute_process( COMMAND ${CMAKE_COMMAND} --build ${ToolkitBuildPath} -- -j 8 install )
        find_path( ToolkitInstallPath NAMES include/mcatk
                   PATHS ${installPaths}
                   DOC "Location of the installation directory for the toolkit"
                 )
        if( NOT ToolkitInstallPath )
            message( FATAL_ERROR "Attempt to install toolkit resources failed." 
                     "Failed to populate directory [ ${ToolkitInstallPath} ] )" )
        endif()
    endif()

    message( "Including toolkit information installed in [ ${ToolkitInstallPath} ]" )
    
    include( ${ToolkitInstallPath}/cmake_files/toolkit.cmake )
    
    # Set the installation of this package to be under the main package but in a subdirectory
    if( ARGV0 )
        set( CMAKE_INSTALL_PREFIX "${ToolkitInstallPath}/${ARGV0}" CACHE INTERNAL "Prefix prepended to install directories" FORCE )
        message( "Installing products in [ ${CMAKE_INSTALL_PREFIX} ]" )
    endif()
    
    ############################################
    #  List the directory names that only contain testing source
    set( TestDirNames unit_test punit_test pnightly nightly fi_test pfi_test PARENT_SCOPE )
    

    include_directories( ${ToolkitInstallPath}/include )
    include_directories( ${ToolkitInstallPath}/include/mcatk )
    link_directories   ( ${ToolkitInstallPath}/${library_install_prefix} )
#    if( CMAKE_BUILD_TYPE STREQUAL "Debug" )
#        link_directories   ( ${ToolkitInstallPath}/lib/debug     )
#    endif()
    
endfunction()
#--------------------------------------------------------------------------------

#================================================================================
# Function: checkBuildDirConsistency
# Check that the build directory does not have a misleading or confusing name
function( checkBuildDirConsistency )

    # Insure the user isn't building in a location that is suprising
    if( DEFINED ENV{BINARY_DIR} )
        if( NOT $ENV{BINARY_DIR} STREQUAL ${CMAKE_BINARY_DIR} )
            message( FATAL_ERROR "Environment variable BINARY_DIR [ $ENV{BINARY_DIR} ] is not equal to the "
                                 "current build directory [ ${CMAKE_BINARY_DIR} ] " )
        endif()
    endif()

    # Make sure the directory name isn't misleading
    string( REGEX MATCH "(gnu|intel|xl)" buildDirTool ${CMAKE_BINARY_DIR} )
    if( buildDirTool )
        string( TOLOWER ${CMAKE_CXX_COMPILER_ID} toolName )
        string( REGEX MATCH "${buildDirTool}" matchesCXX ${toolName} )
        if( NOT matchesCXX )
            message( FATAL_ERROR "Build directory name [ ${CMAKE_BINARY_DIR} ] refers to a tool [ ${buildDirTool} ] that doesn't"
                                 " match the configuration [ ${CMAKE_CXX_COMPILER_ID} ]" )
        endif()
    endif()
    
endfunction()
#--------------------------------------------------------------------------------

#================================================================================
# Function: checkBuildDirConsistency
# Check that the build directory does not have a misleading or confusing name
function( configureToolset toolsetSelection )

    set( ValidTools "gnu|intel|xl|clang" )
    string( REGEX MATCH "(${ValidTools})" toolset ${toolsetSelection} )
    if( NOT toolset )
        message( FATAL_ERROR "User specified toolset [ ${toolsetSelection} ].  Valid entries are [ ${ValidTools} ]" )
    endif()
    message( "Configuring for ${toolset}" )
    if( toolset STREQUAL "intel" )
        set( ENV{CC}  "icc" )
        set( ENV{CXX} "icpc" )
        set( ENV{FC}  "ifort" )
    elseif( toolset STREQUAL "gnu" )
        set( ENV{CC}  "gcc${GNU_VER}" )
        set( ENV{CXX} "g++${GNU_VER}" )
        set( ENV{FC}  "gfortran${GNU_VER}" )
    elseif( toolset STREQUAL "xl" )
        set( ENV{CC}  "xlc" )
        set( ENV{CXX} "xlC" )
        set( ENV{FC}  "xlf90" )
    elseif( toolset STREQUAL "clang" )
        set( ENV{CC}  "clang" )
        set( ENV{CXX} "clang++" )
        set( ENV{FC}  "gfortran" )
    endif()

endfunction()
#--------------------------------------------------------------------------------

#================================================================================
# Function: append_link_property
# Since setting a link property in this way overwrites any previous values, this 
# provides a way to append rather than overwrite.
function( append_link_property target_name value )
    get_target_property( orig_linkprops ${target_name} LINK_FLAGS )
    if( orig_linkprops )
        set( value "${orig_linkprops} ${value}" )
    endif()
    set_target_properties( ${target_name} PROPERTIES LINK_FLAGS "${value}" )
    get_target_property( new_linkprops ${target_name} LINK_FLAGS )
#    message( "Link props: Target [ ${target_name} ] - [ ${new_linkprops} ]" )
endfunction()
#--------------------------------------------------------------------------------

#================================================================================
# Function: determine_linker_vendor
# Not all linkers are binutils
function( determine_linker_vendor )
    execute_process( COMMAND ${CMAKE_LINKER} -v
                     OUTPUT_VARIABLE LINKER_DETAILS
                     ERROR_VARIABLE LINKER_DETAILS
                     OUTPUT_STRIP_TRAILING_WHITESPACE)
    if( LINKER_DETAILS MATCHES "GNU" )
        set( USING_GNU_LINKER true CACHE BOOL "This is the GNU linker." )
    elseif( LINKER_DETAILS MATCHES "LLVM" )
        set( USING_LLVM_LINKER true CACHE BOOL "This is the LLVM linker (i.e. Clang)." )
    else()
        message( FATAL_ERROR "System linker is unrecognized [${LINKER_VERSION}] so all current flags are assumed invalid." )
    endif()
endfunction()
#--------------------------------------------------------------------------------

