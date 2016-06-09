################################################
#
# Common cmake stuff for the Intel's Thread Building Blocks
#
################################################
#if( NOT TBB_Init )
#  set( TBB_Init true )

    # Headers for tbb
#    set( TBB_Platform x86_64   )
#    set( TBB_Version 22_013oss )
#    set( TBB_Root    ${package_dir}/src/tbb${TBB_Version} )

#    if( CMAKE_BUILD_TYPE STREQUAL "Debug" )
#      set( TBB_Debug_Flag _debug )
#    endif()

#endif()

#set( TBB_LIBRARIES tbb${TBB_Debug_Flag} )
#set( TBB_LIBRARY_DIRS ${TBB_Root}/lib/${TBB_Platform}${TBB_Debug_Flag} )
#set( TBB_INCLUDE_DIRS ${TBB_Root}/include )


#===============================================================================
#
#        TBB - Intel Thread Building Blocks
#
#===============================================================================

option(USE_TBB "Include the Intel Thread Building Blocks package in the build" OFF)

if(USE_TBB)
    set(TBB_FOUND FALSE)
	set(TBB_Init TRUE)
	
    find_package(TBB COMPONENTS malloc)
    #find_package(TBB)
    if(TBB_FOUND)
        message(STATUS "TBB FOUND : ${TBB_ROOT_DIR}")
        message(STATUS "TBB LIBRARIES : ${TBB_LIBRARIES}")


		#add_definitions(-D_TBB_)

		# add TBB_LIBRARY_DIRS which FindTBB.cmake does not set		
		set(TBB_LIBRARY_DIRS )
		foreach(_lib ${TBB_LIBRARIES})
            get_filename_component(_lib_path ${_lib} PATH)
            list(APPEND TBB_LIBRARY_DIRS ${_lib_path})
            unset(_lib_path)
        endforeach()
        list(REMOVE_DUPLICATES TBB_LIBRARY_DIRS)
		#message(STATUS "TBB LIBRARY DIRS : ${TBB_LIBRARY_DIRS}")
		
    else()
        message(WARNING "\n\tNO TBB_ROOT FOUND -- ${TBB_ROOT_DIR} -- Please set TBB_ROOT_DIR\n")
    endif()

endif()


