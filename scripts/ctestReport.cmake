#######################################################################
# Main Section
#================================================================================
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR} )
include( PlatformInfo )

#######################################################################
# Collect system info
PlatformInfo()

if( NOT toolkitBuildDir )
    ParseBuildArgs( ${CTEST_SCRIPT_ARG} )
    configureCTest()
endif()

#######################################################################
# CTest Critical Section
#--------------------------------
get_filename_component( buildName ${toolkitBuildDir} NAME )
set( CTEST_BUILD_CONFIGURATION ${Build}     )
set( CTEST_SITE                ${CTestSite} )
if( DEFINED ENV{SYS_TYPE} )
    set( CTEST_BUILD_NAME       "$ENV{SYS_TYPE}-${buildName}" )
else()
    set( CTEST_BUILD_NAME       "linux-${buildName}" )
endif()

set( CTEST_PROJECT_SUBPROJECTS ToolkitLib FlatAPI QtAPI )

find_program( PYTHON python 
              PATHS /usr/lanl/bin )
find_file( MergeScript 
           MergeTestResults.py 
           PATHS ${CMAKE_CURRENT_LIST_DIR}
         )

set( subprojects ToolkitLib Flat Qt4 )

foreach( subproject ${subprojects} )

    if( ${subproject} STREQUAL ToolkitLib )
        set( SubProjectName ${subproject} )
        unset( SubProjectDir )

    else()
        set( SubProjectName ${subproject}API )
        set( SubProjectDir  "API/${subproject}" )
        if( ${subproject} STREQUAL Qt4 )
            set( SubProjectName "QtAPI" )
        endif()
    endif()

    set_property( GLOBAL PROPERTY SubProject ${SubProjectName} )
    set_property( GLOBAL PROPERTY Label      ${SubProjectName} )
    set( CTEST_SOURCE_DIRECTORY "${toolkitSrcDir}/${SubProjectDir}" )
    set( CTEST_BINARY_DIRECTORY "${toolkitBuildDir}/${SubProjectDir}")

    # This initiates a lot of work underneath ctest itself.  It is essential to
    # what follows.  APPEND appears here to prevent a new date tag from being created.
    ctest_start( ${Model} APPEND )

    file  ( READ "${CTEST_BINARY_DIRECTORY}/Testing/TAG" tag_file )
    string( REGEX MATCH "[^\n]*" BuildTag ${tag_file} )
    set   ( TEST_OUTPUT_DIR "${CTEST_BINARY_DIRECTORY}/Testing/${BuildTag}" )
    set   ( TEST_TEMP_DIR "${TEST_OUTPUT_DIR}/temp" )

    #######################################################################
    # Merge nightly tests into a single xml
    #--------------------------------
    execute_process( COMMAND ${PYTHON} ${MergeScript} WORKING_DIRECTORY ${TEST_TEMP_DIR} )
    configure_file( ${TEST_TEMP_DIR}/Test.xml ${TEST_OUTPUT_DIR} COPYONLY )
  
    #######################################################################
    # Submit results
    #--------------------------------
    ctest_submit()
    
endforeach()

