#######################################################################
# Build toolkit and run benchmark problems
#================================================================================

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR} )
include( PlatformInfo )

#######################################################################
# Parse argument list

ParseBuildArgs( ${CTEST_SCRIPT_ARG} )

#######################################################################
# Collect system info

PlatformInfo()

#######################################################################
# Max Procs for scaling study
set( Nodes 1 )
if( BatchSystem )
    string( REGEX MATCH "maxprocs" MaxProcsAvail ${CTEST_SCRIPT_ARG} )
    if( MaxProcsAvail )
        set( NP_regex ".*maxprocs=([0-9]+).*" )
        string( REGEX REPLACE "${NP_regex}" "\\1" NMaxProcs ${CTEST_SCRIPT_ARG} )
        math( EXPR Nodes "${NMaxProcs}/${ProcPerNode}" )
    endif()
    if( BatchSystem STREQUAL MOAB )
        include( MOABJobs )
    endif()
    
    # User may override the default account
    string( REGEX MATCH "account" RunAccountSet ${CTEST_SCRIPT_ARG} )
    if( RunAccountSet )
        set( Acct_regex ".*account=([a-z]+).*" )
        string( REGEX REPLACE "${Acct_regex}" "\\1" RunAccount ${CTEST_SCRIPT_ARG} )
        set( JobAccount ${RunAccount} CACHE STRING "Job account determines group quota" FORCE )
    endif()
endif()

# Check if profiling is needed
string( REGEX MATCH "profile" doProfiling ${CTEST_SCRIPT_ARG} )
set( attemptProfiling OFF )
if( doProfiling )
    set( attemptProfiling ON )
endif()

# Check if a non-default scheduling has been requested. 
# OpenMPI: --bynode (round robin) or --byslot (default)
# MPICH: -rr (round robin)
string( REGEX MATCH "schedule" ScheduleOpt ${CTEST_SCRIPT_ARG} )
if( ScheduleOpt )
    set( sched_regex ".*schedule=([^ ]+).*" )
    string( REGEX REPLACE ${sched_regex} "\\1" Schedule ${CTEST_SCRIPT_ARG} )
endif()

#######################################################################
# General Configuration
configureCTest( "Bench" )

# Guarantee the binary directory is deleted before building
ctest_empty_binary_directory( ${toolkitBuildDir} )

#######################################################################
# Repository: Subversion
#--------------------------------
initializeSVN()

# Before doing a parallel build, determine how many processors we have
if( BatchSystem STREQUAL slurm )
    set(CTEST_BUILD_FLAGS -j8)
    set( nHostProcs 64 )
else()
    determineProcessorCount()
    if( nHostProcs GREATER 16 )
        set( nHostProcs 16 )
        message( WARNING "Limiting procs to [ ${nHostProcs} ]" )
    endif()
    set( CTEST_BUILD_FLAGS -j${nHostProcs} )
endif()

set( CTEST_TEST_TIMEOUT "7200" )

set( CTEST_PROJECT_SUBPROJECTS ToolkitLib Benchmarks )

##############################################
# Toolkit
#--------------------------------
    set( CTEST_PROJECT_NAME mcatk )

    set_property( GLOBAL PROPERTY SubProject ToolkitLib )
    set_property( GLOBAL PROPERTY Label      ToolkitLib )

    set( CTEST_SOURCE_DIRECTORY "${toolkitSrcDir}")
    set( CTEST_BINARY_DIRECTORY "${toolkitBuildDir}")

    #######################################################################
    # START
    #--------------------------------
    ctest_start( ${Model} )

    ##############################################
    # UPDATE
    #--------------------------------
    ctest_update( RETURN_VALUE update_result )
    if( update_result EQUAL -1 )
        message( FATAL_ERROR "Repository update error: Server not responding" )
    endif()
    
    #######################################################################
    # Configure
    #--------------------------------
    set( config_options 
             -DFastBuild:BOOL=ON
             -DCMAKE_BUILD_TYPE:STRING=${CTEST_BUILD_CONFIGURATION}
        )
    ctest_configure( OPTIONS "${config_options}"
                    )
    #######################################################################
    # Build
    # ---------------------------------------------------------------
    ctest_build( TARGET "install" )

    #######################################################################
    # Submit results
    #--------------------------------
    ctest_submit()
    
##############################################
# Run benchmark problems
#--------------------------------
    set( CTEST_PROJECT_NAME Benchmarks )
    
    set_property( GLOBAL PROPERTY SubProject Benchmarks )
    set_property( GLOBAL PROPERTY Label      Benchmarks )

    set( CTEST_SOURCE_DIRECTORY "${toolkitSrcDir}/Benchmarks")
    set( CTEST_BINARY_DIRECTORY "${toolkitBuildDir}/Benchmarks")
    
    # Guarantee the binary directory is deleted before building
    ctest_empty_binary_directory( ${CTEST_BINARY_DIRECTORY} )
    file( MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY} )

    #######################################################################
    # START
    #--------------------------------
    ctest_start( ${Model} )

    #######################################################################
    # Configure
    set( config_options 
             -DNMaxProcs=${NMaxProcs} -DProfile=${attemptProfiling} -DSchedule=${Schedule}
             -DCMAKE_BUILD_TYPE:STRING=${CTEST_BUILD_CONFIGURATION}
        )
    ctest_configure( OPTIONS "${config_options}"
                    )
    #######################################################################
    # Build
    # ---------------------------------------------------------------
    ctest_build( )

if( NOT BatchSystem )
    #######################################################################
    # Test
    # ---------------------------------------------------------------
    ctest_test( INCLUDE_LABEL "Strong" SCHEDULE_RANDOM on )

else()
    #######################################################################
    # Submit tests (this will wait for results)
    # ---------------------------------------------------------------
    set( VerbosityFlags "-VV" )

    set( nBenchProcs ${NMaxProcs} )
    while( Nodes GREATER 0 )
        foreach( ScalingType weak strong )
            math( EXPR tag "90000 + ${nBenchProcs}" )
            string( REGEX REPLACE "^9" ${ScalingType} tag ${tag} ) 
            # set the options into a single argument
            set( ScriptDefines "-DProblemTag=${tag}" )
            set( OPTS "${Model} ${Tool} ${Build}" )
            JobSubmit( Benchmarks_${tag} runBenchmarks.cmake )
        endforeach()
        
        # Run the next lower power of 2
        math( EXPR nBenchProcs "${nBenchProcs} / 2" )
        # set the number of nodes required for the allocation
        math( EXPR Nodes "${nBenchProcs} / ${ProcPerNode}" )
    endwhile()
    
    WaitAll()
    #######################################################################
    # Merge tests into a single xml
    #--------------------------------
    find_program( PYTHON python 
                  PATHS /usr/lanl/bin )
    find_file( MergeScript 
               MergeTestResults.py 
               PATHS ${CMAKE_CURRENT_LIST_DIR}
             )

    file  ( READ "${CTEST_BINARY_DIRECTORY}/Testing/TAG" tag_file )
    string( REGEX MATCH "[^\n]*" BuildTag ${tag_file} )
    set   ( TEST_OUTPUT_DIR "${CTEST_BINARY_DIRECTORY}/Testing/${BuildTag}" )
    set   ( TEST_TEMP_DIR "${TEST_OUTPUT_DIR}/temp" )

    execute_process( COMMAND ${PYTHON} ${MergeScript} WORKING_DIRECTORY ${TEST_TEMP_DIR} )
    configure_file( ${TEST_TEMP_DIR}/Test.xml ${TEST_OUTPUT_DIR} COPYONLY )
  
endif()

    #######################################################################
    # Submit results
    #--------------------------------
    # non-queueing machines
    ctest_submit()
