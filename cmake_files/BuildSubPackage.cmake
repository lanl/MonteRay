
# Function: CreateSubPackageRules
# Details: Build the API packages that depend on the toolkit library.  These may generate
#          other libraries or binaries
function( CreateSubPackageRules )
    set( toolkitReady ${CMAKE_INSTALL_PREFIX}/.ToolkitInstalled )
    add_custom_command( OUTPUT ${toolkitReady} 
                        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} -- -j 8 install
                        COMMAND ${CMAKE_COMMAND} -E touch ${toolkitReady}
                        DEPENDS mcatk
                        COMMENT "Installing toolkit"
                       )
    # pass on the release name if available
    if( ReleaseName )
        set( PackageReleaseFlag "-DReleaseName=${ReleaseName}" )
        if( isProdRelease )
           set( PackageProdReleaseFlag "-DisProdRelease=${isProdRelease}" )
        endif()
    endif()
    if( BatchMode )
        set( BatchFlag "-DBatchMode=ON" )
    endif()
    if( Standalone )
        set( StandaloneFlag "-DStandalone=ON" )
    endif()
    
    set( BuildFlag -DBuildType=${CMAKE_BUILD_TYPE} )
    string( TOLOWER ${CMAKE_CXX_COMPILER_ID} toolset )
    set( ToolsetFlag "-DToolset=${toolset}" )
    if( CompilerExtension )
        list( APPEND ToolsetFlag "-DGNU_VER=${CompilerExtension}" )
    endif()

    set( Qt4Output ${CMAKE_INSTALL_PREFIX}/API/Qt4/${bin_install_prefix}/Lnk3dntTool )
    set( FlatOutput ${CMAKE_INSTALL_PREFIX}/API/Flat/${library_install_prefix}/libmcatk_API.so )
    foreach( pkg Qt4 Flat )
        set( SrcFlag    "-DSrcDir=${CMAKE_SOURCE_DIR}/API/${pkg}" )
        set( BldDirFlag "-DBuildDir=${CMAKE_BINARY_DIR}/Build${pkg}" )
        add_custom_command( OUTPUT ${${pkg}Output}
                            COMMAND ${CMAKE_COMMAND} ${ToolsetFlag} ${PackageReleaseFlag} ${PackageProdReleaseFlag} ${BatchFlag} ${StandaloneFlag} ${BuildFlag} ${SrcFlag} ${BldDirFlag} -P ${cmake_dir}/BuildSubPackage.cmake 
                            DEPENDS ${toolkitReady}
                            COMMENT "Build API package [ ${pkg} ]" 
                          )
    endforeach()
    add_custom_target( BuildAll DEPENDS ${Qt4Output} ${FlatOutput} )
    add_custom_target( BuildQt DEPENDS ${Qt4Output} )
    add_custom_target( BuildFlat DEPENDS ${FlatOutput} )
endfunction()

function( BuildSubPackages )
    message( "Toolset = ${Toolset} BuildDir = ${BuildDir} SrcDir = ${SrcDir} BuildType=${BuildType}" )
    if( NOT SrcDir )
        message( FATAL_ERROR "The package src directory MUST be specified. SrcDir = *${SrcDir}*" )
    endif()
    include( ${SrcDir}/../../cmake_files/GeneralFunctions.cmake )
    
    if( NOT BuildDir )
         set( BuildDir $ENV{BINARY_DIR} )
         if( NOT BuildDir )
            message( FATAL_ERROR "Need to set environment variable *BINARY_DIR* before building." )
         endif()
    endif()
     
    configureToolset( ${Toolset} )
     
    get_filename_component( PkgName ${SrcDir} NAME )
        message( "Building ${PkgName} in ${BuildDir}" )
    endif()

    # Build the directory if it doesn't already exist 
    if( NOT EXISTS ${BuildDir} )
        execute_process( COMMAND ${CMAKE_COMMAND} -E make_directory ${BuildDir} )
    endif()

    if( ReleaseName )
        set( ReleaseInfo "-DReleaseName=${ReleaseName}" )
        if( isProdRelease )
          set( ProdReleaseInfo "-DisProdRelease=${isProdRelease}" )
        endif()
    endif()
    if( BatchMode )
        set( BatchFlag "-DBatchMode=ON" )
    endif()
    if( Standalone )
        set( StandaloneFlag "-DStandalone=ON" )
    endif()

    # Generate the build system in the build directory
    execute_process( COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=${BuildType} ${ReleaseInfo} ${ProdReleaseInfo} ${BatchFlag} ${StandaloneFlag} ${SrcDir} 
                     WORKING_DIRECTORY ${BuildDir} )

    # Build the package
    # Use cmake to invoke the appropriate build tool
    execute_process( COMMAND ${CMAKE_COMMAND} --build ${BuildDir} -- -j8 install )

endfunction()

###############################################################################
# Build secondary packages that use the toolkit
if( NOT BuildDir ) 
    CreateSubPackageRules()
else()
    BuildSubPackages()
endif()


