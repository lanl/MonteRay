
#include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/src/DeviceProperties )
#include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/src/Utilities )
#
#ADD_SUBDIRECTORY(DeviceProperties)
#ADD_SUBDIRECTORY(Utilities)

#################################################################################
if( enable_cuda )
  # Create MonteRay CUDA library
  set( AllCudaFiles "")
  globCudaFiles( "${TestDirNames}" AllCudaFiles)
  set_source_files_properties( ${AllCudaFiles} PROPERTIES LANGUAGE CUDA )
  cuda_add_library(MonteRay SHARED ${AllCudaFiles} )
  CUDA_COMPILE( MONTERAY_O ${AllCudaFiles} )
  #message(STATUS "src/CMakeLists.txt :: MONTERAY_O=${MONTERAY_O}")
  install( TARGETS MonteRay EXPORT MonteRay DESTINATION ${library_install_prefix} )
else()
  set( AllCudaFiles "")
  globCudaFiles( "${TestDirNames}" AllCudaFiles)
endif()
#################################################################################

############################################
#  Toolkit Version Creation
#message( STATUS "%%%%%%%%%%DEBUG: Starting src/CmakeLists.txt")
includeAllHeaders( ${TestDirNames} )

set( ExportName "Toolkit" )

set_property( GLOBAL PROPERTY NightlyCount "0" )

addCMakeDirs()

############################################
#  Create toolkit library dependent on ToolkitVersion
#set( mcatkLibName mcatk )
#add_library( ${mcatkLibName} SHARED  )
#set_target_properties( ${mcatkLibName} PROPERTIES VERSION ${Toolkit_Major_Version}.${Toolkit_Minor_Version}
#                                        SOVERSION 1 )
#add_custom_command( TARGET ${mcatkLibName} POST_BUILD COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_INSTALL_PREFIX}/.ToolkitInstalled )

#if( USING_GNU_LINKER )
#    set( allSymbols "-Wl,-whole-archive" )
#    set( usedSymbols "-Wl,-no-whole-archive" )
#elseif( USING_LLVM_LINKER ) # Using apple's version of ld
#    append_link_property( ${mcatkLibName} -all_load )
#else()
#endif()

# Adding static libs to the main dynamic one
#if( USING_GNU_LINKER OR USING_LLVM_LINKER )
#    target_link_libraries( ${mcatkLibName} ${allSymbols} ${Toolkit_libs} ${usedSymbols} )
#
#    foreach( pkg UnitTest )
#        if( ${pkg}_LIBRARIES )
#           target_link_libraries( ${mcatkLibName} ${allSymbols} ${${pkg}_LIBRARIES} ${usedSymbols} )
#        endif()
#    endforeach()
#
#    if( USING_LLVM_LINKER )
#        target_link_libraries( ${mcatkLibName} ${MPI_CXX_LIBRARIES} )
#    endif()    
#
#    # This removes the previously mentioned libraries from being included when another project needs
#    # to link against libmcatk.so
#    target_link_libraries( ${mcatkLibName} LINK_PRIVATE )
#    
#    # This adds the compiler's library location to the rpath so that customers don't need to care
#    # which compiler was actually used to build the library
#    if( DEFINED mcatk_COMPILER_LIBRARY_DIR )
#        append_link_property( ${mcatkLibName} -Wl,-rpath,${mcatk_COMPILER_LIBRARY_DIR} )
#    endif()
#    if( CMAKE_CXX_COMPILER_ID STREQUAL "Intel" )
#        # Statically links in proprietary libraries needed by icpc
#        append_link_property( ${mcatkLibName} -static-intel )
#    endif()
#endif()
#
#install( TARGETS ${mcatkLibName} EXPORT ${ExportName} DESTINATION ${library_install_prefix} )
#
#install( EXPORT ${ExportName} FILE toolkit.cmake DESTINATION cmake_files )
#install( FILES ${CMAKE_SOURCE_DIR}/cmake_files/templates/FindMCATK.cmake DESTINATION cmake_files )
