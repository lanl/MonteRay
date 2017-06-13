
################################################################################
#
#  Include this template cmake file after specifying any include and library
#  dependencies that the test executable may need
#
################################################################################

if( FastBuild )
    return()
endif()

# Should be inherting value of localName from library_template.cmake which called us
set( ParentDir ${localName} )

########################################
#  Get the name of directory
get_filename_component( TestDirName ${CMAKE_CURRENT_SOURCE_DIR} NAME )

if( NOT DEFINED testname )
    message( FATAL_ERROR "common_test_template.cmake must be called AFTER a testname has been defined." )
endif()

if( TestDirName STREQUAL "unit_test" )
    set( TestType "Unit" )
elseif( TestDirName STREQUAL "punit_test" )
    set( TestType "ParUnit" )
    set( parallel true )
elseif( TestDirName STREQUAL "fi_test" )
    set( TestType "FIUnit" )
elseif( TestDirName STREQUAL "pfi_test" )
    set( TestType "ParFIUnit" )
    set( parallel true )
elseif( TestDirName STREQUAL "nightly" )
    set( TestType "Nightly" )
    set( nightly true )
else()
    set( TestType "ParNightly" )
    set( nightly true )
    set( parallel true )
endif()

########################################
#  Add IPComm for runs which be using mpirun

list( FIND ${ParentDir}_packages MPI foundMPI ) 
if( foundMPI GREATER -1 )
    set( usingMPI true )
endif()
if( parallel OR usingMPI )
    list( APPEND ${ParentDir}_includes IPComm )
    list( INSERT ToolkitLibs     0     IPComm )
    list( INSERT ${ParentDir}_packages 0 Boost_MPI )
    if( NOT usingMPI )
        list( APPEND ${ParentDir}_packages MPI )
    endif()
    if( POE )
        set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -binitfini:poe_remote_main" )
    endif()
endif()

 
########################################
#  Application Name

set( appName ${TestType}${testname} )

########################################
#  Gather sources

file( GLOB ${appName}_srcs "*.cpp" "*.cc" "*.cu" )

foreach( src ${ExcludeSource} )
    list( REMOVE_ITEM ${appName}_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${src} )
endforeach()

########################################
#  Handle includes:
# Always add in the UnitTest stuff
list( APPEND ${ParentDir}_packages UnitTest )

# Serial tests will still require the mpi header
# NOTE: This may be indicative of larger issue of requiring headers without requiring the library
if( NOT usingMPI )
    include_directories( ${MPI_INCLUDE_DIRS} )
endif()

# Pickup any 3rd party headers the library depend ons
foreach( pkg ${${ParentDir}_packages} ) 

    include_directories( ${${pkg}_INCLUDE_DIRS} )

    if( DEFINED ${pkg}_LIBRARY_DIRS )
        link_directories( ${${pkg}_LIBRARY_DIRS} )
    endif()

endforeach()

#    The developer should propagate a list of toolkit headers, but the root include from the
#    the package directory will always come first (candycorn PGI issues)
include_directories( ${package_dir}/include )

#  Add the current source dir to the include list
list( APPEND ${ParentDir}_includes ${RelDirPath}/${TestDirName} )

add_definitions(-D${CMAKE_SYSTEM_NAME}) 

########################################
#  Add the new test

add_executable( ${appName} ${${appName}_srcs} )

set_property(TARGET ${appName} APPEND PROPERTY COMPILE_DEFINITIONS ${CXX11_FEATURE_LIST} )

########################################
#  Collect the suite names defined for these tests
set( SuiteNames )
foreach( curSrc ${${appName}_srcs} )
    file( STRINGS ${curSrc} curSuiteNames REGEX "^SUITE" )
    
    foreach( suiteName ${curSuiteNames} )
        string( REGEX REPLACE "SUITE[^a-zA-Z0-9]+" "" suiteName ${suiteName} )
        string( REGEX MATCHALL "^[a-zA-Z_0-9]+" suiteName ${suiteName} )
        list( APPEND SuiteNames ${suiteName} )
    endforeach()
endforeach()
if( DEFINED SuiteNames ) 
    list( REMOVE_DUPLICATES SuiteNames )
endif()    

########################################
#  Toolkit libraries that need to linked in

# Try to add parent directories library - removed in MonteRay
#if( DEFINED libname )
#    target_link_libraries( ${appName} ${libname} )
#endif()

# Add any additional
foreach( curLib ${ToolkitLibs} ) 
    target_link_libraries( ${appName} ${curLib} )
endforeach()

########################################
#  3rd Party packages that need to linked in

foreach( pkg ${${ParentDir}_packages} ) 
    if( DEFINED ${pkg}_LIBRARIES )
        target_link_libraries( ${appName} ${${pkg}_LIBRARIES} )
    endif()
endforeach()
if( CMAKE_SYSTEM MATCHES "Darwin" )
else()
    target_link_libraries( ${appName} rt )
endif()
if( Platform STREQUAL "BlueGeneQ" )
    target_link_libraries( ${appName} pthread )
    set( MPIEXEC_PREFLAGS ${MPI_EXECPREFLAGS} --partition=pdebug -t 1:30:00 )
endif()

# Need to explicitly link in thread library if this is known
if( CMAKE_THREAD_LIBS_INIT )
    target_link_libraries( ${appName} ${CMAKE_THREAD_LIBS_INIT} )
endif()

######################################################################
#  TESTING

# Create a link in each test directory to the lnk3dnt files on that system
#add_custom_command( 
#    TARGET ${appName} POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E create_symlink ${lnk3dnt_location} lnk3dnt
#    DEPENDS ${lnk3dnt_location}
#)

if( DEFINED mcatk_SOURCE_DIR )
    # Create a link in each test directory to the sample nuclear data
    # ONLY use if building in the actual toolkit
    set( NuclearDataLoc ${mcatk_SOURCE_DIR}/src/CollisionPhysics/NuclearDataExamples )
    add_custom_command( 
        TARGET ${appName} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink ${NuclearDataLoc} NuclearDataExamples
        DEPENDS ${NuclearDataLoc}
    )
endif()

foreach( testFile ${TestFiles2Link} )
    if( IS_ABSOLUTE ${testFile} )
        set( original ${testFile} )
    else()
        set( original ${CMAKE_CURRENT_SOURCE_DIR}/${testFile} )
    endif()
    get_filename_component( fileName ${original} NAME )
    set( fake     ${CMAKE_CURRENT_BINARY_DIR}/${fileName} )
#     message( "Trying to link "${original} " to " ${fake} )
    add_custom_command( 
      TARGET ${appName} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E create_symlink ${original} ${fake}
      DEPENDS ${original}
   )
endforeach()

foreach( testFile ${TestFiles2Copy} )
    set( original ${CMAKE_CURRENT_SOURCE_DIR}/${testFile} )
    set( fake     ${CMAKE_CURRENT_BINARY_DIR}/${testFile} )
#     message( "Trying to copy "${original} " to " ${fake} )
    add_custom_command( 
      TARGET ${appName} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${original} ${fake}
      DEPENDS ${original}
   )
endforeach()

# This will link files from remote sites (i.e. not stored in our source tree)
foreach( f ${RemoteFiles} )
    get_filename_component( fileName ${f} NAME )
    set( fake     ${CMAKE_CURRENT_BINARY_DIR}/${fileName} )
    add_custom_command( 
      TARGET ${appName} POST_BUILD
#      COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --blue --bold "Creating symbolic link..." 
      COMMAND ${CMAKE_COMMAND} -E create_symlink ${f} ${fake}
      DEPENDS ${f}
   )
endforeach()


set( runResults "(success|fail|debug)" )

if( parallel )
    # Define the number of processors to run with for testing
    if( nightly )
        set( mpiNProcs 8 )
        get_property( prevCount GLOBAL PROPERTY NightlyCount )
        if( MaxJobsPerNUMA AND NOT MaxJobsPerNUMA LESS mpiNProcs )
            set( NightlyApp ${CMAKE_CURRENT_BINARY_DIR}/${appName} )
            configure_file( ${cmake_dir}/RunNightly.cmake runSuite.cmake @ONLY )
        endif()
    else()
        set( mpiNProcs 1 2 3 4 )
    endif()
    
    #########################################################################################################
    # Looping over # of MPI ranks
    foreach( nProcs ${mpiNProcs} )
    
      #######################################################################################################
      # Loop over the 'suites'
      foreach( suiteName ${SuiteNames} )
      
        # Create a unique test name
        set( CTestID ${appName}_${suiteName}_wP${nProcs} )
        
        # Register with CTest
        if( EXISTS ${CMAKE_CURRENT_BINARY_DIR}/runSuite.cmake )
            math( EXPR prevCount "${prevCount}+1" )
            
            add_test( NAME ${CTestID} COMMAND ${CMAKE_COMMAND} -DSUITE=${suiteName} -DTestID=${prevCount} -P runSuite.cmake )
        else()
            add_test( NAME ${CTestID} COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${nProcs} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${appName}> ${suiteName} )
        endif()
        

        # Label with sub-project name
        if( DEFINED SubProjectName )
            set_property( TEST ${CTestID} APPEND PROPERTY LABELS ${SubProjectName} )
        endif()
        
        # Label as serial or parallel
        set_property( TEST ${CTestID} APPEND PROPERTY LABELS Parallel )
        
        # Add the number of processors requested as its own label
        set_property( TEST ${CTestID} PROPERTY PROCESSORS ${nProcs} )
        
        # Label based on whether this is a nightly test
        if( nightly )
            set_property( TEST ${CTestID} APPEND PROPERTY LABELS Nightly Nt${nProcs} ParNightly )
        else()
            set_property( TEST ${CTestID} APPEND PROPERTY LABELS Quick Qk${nProcs} )
        endif()
      endforeach()

        # This will cause it to run once after building (for all but nightlies!)
        if( NOT nightly AND NOT BatchMode AND NOT POE )
            set( CmdTag "Testing ${UnitName}[ ${TestDirName}.MPI${nProcs} ]..." )
            set( CmdTagSuccess "${CmdTag} PASS" )
            add_custom_command( 
                TARGET ${appName} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --blue --bold ${CmdTag} 
                COMMAND ${CMAKE_COMMAND} -DAPP=${appName} -DMPIEXEC=${MPIEXEC} -DNPROCS=${nProcs} -DUNIT=${UnitName} -P ${cmake_dir}/AnalyzeParallelRun${POE}.cmake
                COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --blue --bold ${CmdTagSuccess} 
            )
        endif()
    endforeach()
    if( nightly ) 
        set_property( GLOBAL PROPERTY NightlyCount ${prevCount} )
#        message( "Nightly Count: ${prevCount}" )
    endif()
else()
    foreach( suiteName ${SuiteNames} )
      # Register with CTest
      set( CTestID ${appName}_${suiteName} )
      if( DEFINED JobController )
          add_test( NAME ${CTestID} COMMAND ${JobController} $<TARGET_FILE:${appName}> ${suiteName} )
      else()
          add_test( NAME ${CTestID} COMMAND ${appName} ${suiteName} )
      endif()

      set_property( TEST ${CTestID} APPEND PROPERTY LABELS Serial ${SubProjectName} )
      
      # Specify to the job dispatch system that these are serial
      set_property( TEST ${CTestID} PROPERTY PROCESSORS 1 )
        
      if( nightly )
          set_property( TEST ${CTestID} APPEND PROPERTY LABELS Nightly SerialNightly )
      else()
          set_property( TEST ${CTestID} APPEND PROPERTY LABELS Quick SerialQuick )
      endif()
    endforeach()
    
    # This will cause it to run once after building (for all but nightlies!)
    if( NOT nightly AND NOT BatchMode )
        set( CmdTag "Testing ${UnitName}[ ${TestDirName} ]..." )
        set( CmdTagSuccess "${CmdTag} PASS" )
        add_custom_command( 
            TARGET ${appName} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${appName} ${appName}.copy
            COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --blue --bold ${CmdTag} 
            COMMAND ${CMAKE_COMMAND} -DAPP=${appName} -DTAG=${CmdTag} -P ${cmake_dir}/AnalyzeRun.cmake
#            COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --no-label-summary
            COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --blue --bold ${CmdTagSuccess} 
        )
    endif()
endif()

# stuff from a tutorial

# LIBRARY
#   ADD_LIBRARY( lib_${PROJECT_NAME} ${library_sources} )
# create symbolic lib target for calling target lib_XXX
#   ADD_CUSTOM_TARGET( lib DEPENDS lib_${PROJECT_NAME} )
# change lib_target properties
#   SET_TARGET_PROPERTIES( lib_${PROJECT_NAME} PROPERTIES
# create *nix style library versions + symbolic links
#   VERSION ${${PROJECT_NAME}_VERSION}
#   SOVERSION ${${PROJECT_NAME}_SOVERSION}
# allow creating static and shared libs without conflicts
#   CLEAN_DIRECT_OUTPUT 1
# avoid conflicts between library and binary target names
#   OUTPUT_NAME ${PROJECT_NAME} )
# install library
#   INSTALL( TARGETS lib_${PROJECT_NAME} DESTINATION lib PERMISSIONS
#   OWNER_READ OWNER_WRITE OWNER_EXECUTE
#   GROUP_READ GROUP_EXECUTE
#   WORLD_READ WORLD_EXECUTE )
