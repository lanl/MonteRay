################################################################################
#
#  Include this template cmake file after specifying any include and library
#  dependencies that the library may need
#
################################################################################
include( GeneralFunctions )
setModuleName()

set( libname  ${localName} )
set( testname ${libname} )

########################################
#  Gather sources

file( GLOB ${libname}_srcs "*.cpp" "*.cc" "*.cu" )
file( GLOB ${libname}_cuda_srcs "*.cu" )
file( GLOB ${libname}_headers "*.h" "*.hh" )

if( NOT enable_cuda )
    foreach( cudaFile ${${libname}_cuda_srcs} )
       set_source_files_properties( ${cudaFile} PROPERTIES LANGUAGE CXX )
       set_source_files_properties( ${cudaFile} PROPERTIES COMPILE_FLAGS ${COMPILER_CPP_FILE_FLAG} )
    endforeach()
endif()

foreach( configsrc ${config_srcs} )
    configure_file( ${configsrc}.in ${configsrc} )
    list( APPEND ${libname}_srcs ${CMAKE_CURRENT_BINARY_DIR}/${configsrc} )
endforeach()

# Set labels on library src files for later use in targeting code coverage
set_property( SOURCE ${${libname}_srcs}    APPEND PROPERTY LABELS ${SubProjectName}Src )
set_property( SOURCE ${${libname}_headers} APPEND PROPERTY LABELS ${SubProjectName}Src )

install( FILES ${${libname}_headers} DESTINATION include )

foreach( src ${ExcludeSource} )
    list( REMOVE_ITEM ${libname}_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${src} )
endforeach()

########################################
#  Handle includes:
#    The developer should propagate a list of toolkit headers, but the root include from the
#    the package directory will always come first (candycorn PGI issues)

# Pickup any 3rd party headers the library depend ons
foreach( pkg ${${libname}_packages} ) 
    include_directories( ${${pkg}_INCLUDE_DIRS} )
endforeach()

# Always include MPI for MonteRay
include_directories( ${MPI_INCLUDE_DIRS} )

########################################
#  Add these as a new library if there are any sources in the directory

list( LENGTH ${libname}_srcs NSrcs )
if( NSrcs GREATER 0 )
    # Set any flags since this will be placed in a dynamic library
#    set( CMAKE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CXX_FLAGS} ${CMAKE_CXX_FLAGS} -g -pg" )  #gprof
    set( CMAKE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CXX_FLAGS} ${CMAKE_CXX_FLAGS}" )
    add_library( ${libname} STATIC ${${libname}_srcs} ${${libname}_headers} )
    target_compile_features(${libname} PUBLIC cxx_std_11)

    if( enable_cuda )
        set_target_properties( ${libname} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endif()

    install( TARGETS ${libname} EXPORT ${ExportName} DESTINATION ${library_install_prefix} )
    # Add this library to the toolkit's main library (libmcatk.so)
    # If this is a rebuild, the name may already be registered, so check first
    list( FIND Toolkit_libs ${libname} index )
    if( index EQUAL -1  AND NOT Excluding_${UnitName} )
        set( Toolkit_libs ${Toolkit_libs} ${libname} CACHE INTERNAL "" )
    endif()
    
    set_property( TARGET ${libname} APPEND PROPERTY LABELS ${SubProjectName} )

else()
    unset( libname )
endif()

addCMakeDirs( ${ExcludeSrcDirectories} )
