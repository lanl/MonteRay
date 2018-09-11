#============================================================================================
# Platform Specifications
#  1) Site (lanl,llnl or snl)
#  2) System name
#  3) Default scratch directory
#  4) Job control parameters (MOAB?)
#  5) Processor count
#
#  NOTE: This is a macro it is expanded in-line
#============================================================================================
function( locateMCATKDir )
    find_path( MCATKDir 
               NAMES AutoBuilds packages 
               PATHS /home/xshares/PROJECTS
                     /home/xshares
                     /usr/projects
                     /usr/gapps
                     /projects
               PATH_SUFFIXES mcatk )
    
    if( NOT MCATKDir )
        message( FATAL_ERROR "Unable to locate MCATK's project directory on this system." )
    endif()
    # Some systems have a special place for checkout and building the toolkit
    find_path( BuildScratch
               NAMES user 
               PATHS /usr/projects/mcatk_autobuild )
endfunction()

function( PlatformInfo )

    if( ${Standalone} ) 
      set( Platform ${CMAKE_SYSTEM_NAME} CACHE INTERNAL "Platform on which this was configured.")
      return()
    endif()
	
    locateMCATKDir()
    
    site_name( FullSiteName )
    find_path( AutoBuildRoot 
               NAMES Results scripts 
               PATHS ${MCATKDir}/AutoBuilds
               PATH_SUFFIXES AutoBuilds )
    if( NOT AutoBuildRoot )
        message( FATAL_ERROR "Unable to locate autobuild directory on this computer." )
    endif()

    find_file( PlatformMetaFile PlatformDB.txt 
           PATHS ${CMAKE_MODULE_PATH} 
                 ${CTEST_SCRIPT_DIRECTORY} 
                 ${AutoBuildRoot} 
           PATH_SUFFIXES scripts
    )
    message(STATUS "scripts/PlatformInfo.cmake PlatformMetaFile = ${PlatformMetaFile}" )

    if( NOT PlatformMetaFile )
        message( FATAL_ERROR "Unable to locate the platform description file." )
    endif()

    if( POLICY CMP0007 )
        cmake_policy( PUSH )
        cmake_policy( SET CMP0007 OLD )
    endif()

    file( STRINGS ${PlatformMetaFile} Platforms )
    list( GET Platforms 0 Headers )
    string( REGEX REPLACE "," " - " Headers ${Headers} )
    list( REMOVE_AT Platforms 0 )

    # HPC maintains a script that parses a cluster's front and back end id's to create a single cluster id
    find_program( SYSNAME sys_name PATHS /usr/projects/hpcsoft/utilities/bin )
    if( SYSNAME )
        execute_process( COMMAND ${SYSNAME} OUTPUT_VARIABLE SystemID OUTPUT_STRIP_TRAILING_WHITESPACE )
    endif()
    message(STATUS "scripts/PlatformInfo.cmake SystemID = ${SystemID}" )

    foreach( line ${Platforms} )
        # convert line to a list of entries
        string( REGEX REPLACE "[ ,]+" ";" line ${line} )
        list( LENGTH line NEntries )
        
        # Look up site by matching
        if( SystemID ) 
            list( GET line 10 ClusterName )
            message(STATUS "scripts/PlatformInfo.cmake ClusterName = ${ClusterName}" )
            string( REGEX MATCH "${SystemID}" SiteID ${ClusterName} )
        else()
            list( GET line 0 RegexID )
            string( REGEX MATCH "^${RegexID}" SiteID ${FullSiteName} )
        endif()

        if( SiteID )
            message(STATUS "scripts/PlatformInfo.cmake SiteID found in database file: SiteID = ${SiteID}" )
        
            # Cluster short name
            set( ClusterID ${SiteID} CACHE STRING "Abbreviated name of compute cluster" )
            message(STATUS "scripts/PlatformInfo.cmake: ClusterID = ${ClusterID}" )
            
            # Host Site
            list( GET line 1 HostSite )
            set( HostDomain ${HostSite} CACHE STRING "Compute domain of the cluster" )
            message(STATUS "scripts/PlatformInfo.cmake HostSite = ${HostSite}" )
            
            # NUMA Info
            list( GET line 2 NumberNUMA )
            message(STATUS "scripts/PlatformInfo.cmake NumberNUMA = ${NumberNUMA}" )
            
            list( GET line 3 JobsPerNUMA )
            message(STATUS "scripts/PlatformInfo.cmake JobsPerNUMA = ${JobsPerNUMA}" )
            
            if( NumberNUMA STREQUAL "NA" )
            else()
                set( NumNUMA        ${NumberNUMA} CACHE STRING "Number of NUMA domains on cluster" )
                set( MaxJobsPerNUMA ${JobsPerNUMA} CACHE STRING "Maxiumum number of concurrent jobs allowed on a NUMA domain." )
            endif()
            # Scratch Directory
            list( GET line 4 ScratchDir )
            message(STATUS "scripts/PlatformInfo.cmake ScratchDir = ${ScratchDir}" )
            set( AutoScratch ${ScratchDir}/$ENV{USER}/AUTOBUILD CACHE PATH "Scratch directory on the cluster"  )
                        
            # Batch system basics
            if( NEntries GREATER 5 )
                list( GET line 5 batchSystem )
                message(STATUS "scripts/PlatformInfo.cmake batchSystem = ${batchSystem}" )
                set( BatchSystem ${batchSystem} CACHE STRING "Job control system" )
                message(STATUS "scripts/PlatformInfo.cmake BatchSystem = ${BatchSystem}" )
                list( GET line 6 acct )
                if( acct STREQUAL NA )
                    unset( acct )
                else()
                    set( JobAccount ${acct} CACHE STRING "Job account determines group quota" )
                endif()
                list( GET line 7 queue )
                if( queue STREQUAL NA )
                    unset( queue )
                else()
                    set( JobQueue ${queue} CACHE STRING "Job queue determines which allocation group" )
                endif()
                list( GET line 8 ppn )
                list( GET line 9 duration )
                set( ProcPerNode ${ppn} CACHE STRING "Number of computing processors per node" )
                set( JobDuration ${duration} CACHE STRING "Job runtime limit" )
                list( GET line 10 clustername )
                set( ClusterName ${clustername} CACHE STRING "Name of the compute cluster" )
                
                message(STATUS "scripts/PlatformInfo.cmake ClusterName = ${ClusterName}" )
            else()
            endif()
            break()
        endif()
    endforeach()
    
    if( POLICY CMP0007 )
        cmake_policy( POP )
    endif()
    
    if( NOT SiteID )
        message( FATAL_ERROR "System [ ${FullSiteName} ] requires an entry in [ ${PlatformMetaFile} ]." )
    endif()

    if( HostDomain STREQUAL lanl )
        set( CTestSite ${ClusterName} CACHE STRING "Site reported to ctest" )
        # Platform ID case -- LANL HPC
        set( Platform ${ClusterName} CACHE INTERNAL "Platform on which this was configured.")
        find_path( hasNetScratch $ENV{USER} PATH /netscratch NO_DEFAULT_PATH )
        if( hasNetScratch )
            set( RootSrcDir /netscratch/$ENV{USER}/mcatk CACHE PATH "Remote directory for holding source." )
        elseif( BuildScratch )
            set( RootSrcDir ${BuildScratch}/user/$ENV{USER}/AUTOBUILD CACHE PATH "Remote directory for holding source." )
        else()
            set( RootSrcDir ${MCATKDir}/user/$ENV{USER}/AUTOBUILD CACHE PATH "Remote directory for holding source." )
        endif()
        message(STATUS "scripts/PlatformInfo.cmake RootSrcDir = ${RootSrcDir}" )
    elseif( HostDomain STREQUAL xdiv )
        set( CTestSite ${ClusterID} CACHE STRING "Site reported to ctest" )
        # Platform ID for xdiv systems
        if( DEFINED ENV{PlatformOS} )
            set( Platform $ENV{PlatformOS} CACHE INTERNAL "Platform on which this was configured." )
        else()
            message( FATAL_ERROR "User must set environment variable *PlatformOS* in mcatk module file." )
        endif()
        set( RootSrcDir ${AutoScratch} CACHE PATH "Remote directory for holding source." )
    else()
        set( CTestSite ${ClusterName}-${HostDomain} CACHE STRING "Site reported to ctest" )
        set( RootSrcDir ${AutoScratch} CACHE PATH "Remote directory for holding source." )
    endif()

    # If this isn't part of a system with a job controller, we're done
    if( NOT BatchSystem )
        return()
    endif()

    message(STATUS "scripts/PlatformInfo.cmake BatchSystem = ${BatchSystem}" )
    if( BatchSystem STREQUAL MOAB )
        find_program( SubmitCmd msub 
                      PATHS $ENV{MOABHOMEDIR}/bin
                      ENV PATH
                    )
        if( NOT SubmitCmd )
           # check if explicit path is available
           if( DEFINED ENV{MSUB_PATH} )
              set( SubmitCmd $ENV{MSUB_PATH} )
           endif()
        endif()
        if( HostDomain STREQUAL "llnl" )
            # Generic case for LLNL TLCC
            set( Platform $ENV{SYS_TYPE} CACHE INTERNAL "Platform on which this was configured.")
        endif()

    # Special case for LLNL BlueGeneQ clusters (seq, rzuseq, etc.)
    elseif( BatchSystem STREQUAL slurm AND HostDomain STREQUAL llnl )
        find_program( SubmitCmd srun )
        set( SRUN_SERIAL ${SubmitCmd} --partition=${JobQueue} -n 1 -t 1:30:00 CACHE DOC "Required invocation line on BlueGeneQ" )

        set( Platform BlueGeneQ CACHE INTERNAL "Platform on which this was configured" )

    elseif( BatchSystem STREQUAL slurm )
        # LANL is shifting to a slurm-ONLY type batch system
        find_program( SubmitCmd sbatch )
        if( ClusterID STREQUAL trinitite )
            set( SBATCH_EXTRA_OPTIONS "SBATCH --gres=craynetwork:0" CACHE DOC "Flag allowing concurrent execution of tests." )
        endif()
        
    elseif( BatchSystem STREQUAL SBATCH )
        # Special case for SNL batching
        find_program( SubmitCmd sbatch )
        
    elseif( BatchSystem STREQUAL LSF AND HostDoman STREQUAL llnl )
        # For Sierra/Shark/RZManta
        find_program( SubmitCmd bsub )
        set( Platform $ENV{SYS_TYPE} CACHE INTERNAL "Platform on which this was configured.")        

    else()
        message( FATAL_ERROR "Batching system unrecognized [ ${BatchSystem} ] for this system" )
    endif()

    # Whatever the batching system, the command MUST be known
    if( NOT SubmitCmd )
        message( FATAL_ERROR "Unable to locate submission command [ ${SubmitCmd} ] for this system" )
    endif()

endfunction()

#================================================================================
# Function: determineProcessorCount
# Determines the number of cores/processors available on a system
function( determineProcessorCount )
    if( ProcPerNode )
        set( nHostProcs ${ProcPerNode} PARENT_SCOPE )
        return()
    endif()
    
    # Mac:
    if(APPLE)
        find_program( SYSCTL sysctl )
        if( SYSCTL )
            execute_process(COMMAND ${SYSCTL} -n hw.ncpu OUTPUT_VARIABLE PROCESSOR_COUNT )
        else()
            # Use a safe(?) value
            set( PROCESSOR_COUNT 8 )
        endif()
        set( nHostProcs ${PROCESSOR_COUNT} PARENT_SCOPE )
        return()
    endif()
    
    # Windows:
    if(WIN32)
        message( FATAL_ERROR "Don't know how to count available processor under windows" )
        set(PROCESSOR_COUNT "$ENV{NUMBER_OF_PROCESSORS}")
        set( nHostProcs ${PROCESSOR_COUNT} PARENT_SCOPE )
        return()
    endif()
    
    # Linux:
#    find_program( APRUN aprun DOC "Cray/MPICH utility for dispatching parallel jobs to backends." )
    find_program( APRUN srun DOC "Cray/MPICH utility for dispatching parallel jobs to backends." )
    find_program( LSCPU lscpu DOC "Linux utility for analyzing runtime hardware." )
    if( LSCPU )
        if( APRUN )
            execute_process( COMMAND ${APRUN} --quiet ${LSCPU} -e=core OUTPUT_VARIABLE info OUTPUT_STRIP_TRAILING_WHITESPACE )
        else()
            # e=core - physical cores,   e=cpu - cores+hyperthreads
            execute_process( COMMAND ${LSCPU} -e=core OUTPUT_VARIABLE info OUTPUT_STRIP_TRAILING_WHITESPACE )
        endif()
        # convert newlines to semicolons to convert this into a list of strings
        string( REGEX REPLACE "\\\n" ";" info ${info} )
        # Remove the header line
        list( REMOVE_AT info 0 )
        # This list will have duplicates if hyper-threading is enabled because it reports a line for each *cpu*.
        list( REMOVE_DUPLICATES info )
        list( LENGTH info PROCESSOR_COUNT )
    else()
        set(cpuinfo_file "/proc/cpuinfo")
        if(EXISTS "${cpuinfo_file}")
          file(STRINGS "${cpuinfo_file}" procs REGEX "^processor.: [0-9]+$")
          list(LENGTH procs PROCESSOR_COUNT)
        endif()
    endif()
    
    set( nHostProcs ${PROCESSOR_COUNT} PARENT_SCOPE )
endfunction()

#================================================================================
# Function: getDateStamp
# Determines current time
function( getDateStamp result )
    if(WIN32)
        execute_process(COMMAND "date" "/T" OUTPUT_VARIABLE RESULT)
        string(REGEX REPLACE "(..)/(..)/..(..).*" "\\3\\2\\1" RESULT ${RESULT})
    elseif(UNIX)
        execute_process(COMMAND date +%a%b%d_%Hh%Mm%Ss OUTPUT_VARIABLE RESULT)
        string( REPLACE "\n" "" RESULT "${RESULT}" )
    else()
        message(SEND_ERROR "date not implemented")
        set( RESULT 000000)
    endif()
    set( ${result} ${RESULT} PARENT_SCOPE )
endfunction ()

#================================================================================
# Function: configureToolSet
# Setup the desired toolset using environment variables
function( configureToolSet Tool )

    if( $ENV{PlatformOS} MATCHES "Darwin" )
      message("When building on machines with PlatformOS=Darwin, build system will not overide CC, CXX, FC")
      return()
    endif()

    if( Tool STREQUAL "intel" )
        set( ENV{CC}  "icc" )
        set( ENV{CXX} "icpc" )
        set( ENV{FC}  "ifort" )
    elseif( Tool STREQUAL "gnu" )
        set( ENV{CC}  "gcc" )
        set( ENV{CXX} "g++" )
        set( ENV{FC}  "gfortran" )
#    set( gnuCoverageCmd "gcov" )
    elseif( Tool STREQUAL "xl" )
        set( ENV{CC}  "xlc" )
        set( ENV{CXX} "xlC" )
        set( ENV{FC}  "xlf90" )
    elseif( Tool STREQUAL "clang" )
        set( ENV{CC}  "clang" )
        set( ENV{CXX} "clang++" )
        set( ENV{FC}  "gfortran" )
    endif()

    #--------------------------------
    # Check if user has specified different compilers
    set( CompilerEnvName CXX CC FC )
    foreach( compilerType ${CompilerEnvName} )
        set( matchString "${compilerType}=[^; ]+" )
        string( REGEX MATCH ${matchString} tempName "${ARGN}" )
        if( tempName ) 
            string( REGEX REPLACE "${compilerType}=" "" tempName ${tempName} )
            set( ENV{${compilerType}} ${tempName} )
        endif()
    endforeach()
    
    #--------------------------------
    # Parse out version information
    set( VersionCmds --version -dumpversion -qversion )
    foreach( ver_cmd ${VersionCmds} )
        if( NOT compilerVersion )
            execute_process( COMMAND $ENV{CXX} ${ver_cmd} 
                             RESULT_VARIABLE result 
                             OUTPUT_VARIABLE FullCompilerVersion
                             ERROR_VARIABLE error_version_msg )
            if( result EQUAL 0 )
                string( REGEX MATCH "[0-9]+\\.[0-9]+" compilerVersion ${FullCompilerVersion} )
            endif()
        endif()            
    endforeach()
    set( CompilerVersion ${compilerVersion} CACHE STRING "Version number of the c++ compiler." )
endfunction()

#================================================================================
# Function: validateOption
# Sorts through the arguments to see if any match the supplied values. if a match
# is found, it is stored in the value named Option
function( validateOption Option )
    string( REPLACE ";" "|" ValidOptions "${ARGN}" )

    if( ${Option} )
        # If its already defined, check that it matches the known options
        string( REGEX MATCH "(${ValidOptions})" isValid ${${Option}} )
        if( NOT isValid )
            message( FATAL_ERROR "Invalid option provided. [ ${Option} = ${${Option}} ]. Valid [ ${Option} ] options are [ ${ValidOptions} ]" )
        endif()
    else()
        # If it isn't defined, check the arguments for possible values
        string( REGEX MATCH "(${ValidOptions})" isValid ${OptionList} )
        if( isValid )
            set( ${Option} ${isValid} PARENT_SCOPE )
        else()
            if( ARGV1 STREQUAL OPTIONAL )
            else()
                message( FATAL_ERROR "Invalid option provided [ ${Option} ].  Valid options are [ ${ValidOptions} ]" )
            endif()
        endif()
    endif()
#    message( "Option[ ${Option} ] = [ ${isValid} ] -- CHECK" )
endfunction()

#================================================================================
# Function: ParseBuildArgs
# Determines current time
function( ParseBuildArgs )
    if( ARGN )
        string( REPLACE " " ";" OptionList ${ARGN} )
    endif()
    
    set( model ${Model} )
    validateOption( model ContinuousNightly Nightly Continuous Experimental  )
    set( Model ${model} PARENT_SCOPE )

    set( tool ${Tool} )
    validateOption( tool gnu intel xl clang GNU INTEL XL CLANG Gnu Intel Clang )
    string( TOLOWER ${tool} tool )
    set( Tool ${tool} PARENT_SCOPE )
    configureToolSet( ${tool} ${ARGN} )

    set( build ${Build} )    
    validateOption( build Release Debug RelWithDebInfo )
    set( Build ${build} PARENT_SCOPE )
    
    set( ConfigMsg "Model[ ${model} ], Tool[ ${tool}-${CompilerVersion} ], Build[ ${build} ]" )
    
    if( NOT BranchName )
        set( BranchRegex "[Bb]ranch=[^;]+" )
        string( REGEX MATCH ${BranchRegex} branchName "${ARGN}" ) 
        if( branchName ) 
            string( REGEX REPLACE "[Bb]ranch=" "" branchName ${branchName} )
            set( BranchName ${branchName} PARENT_SCOPE )
            set( ConfigMsg "${ConfigMsg} : **Branch[ ${branchName} ]**" )
        endif()
    else()
        set( ConfigMsg "${ConfigMsg} : **Branch[ ${BranchName} ]**" )        
    endif()

    # toggle using git (over svn)
    #string( REGEX MATCH "git" repoStyle "${ARGN}" ) 
    #if( repoStyle ) 
        set( useGIT ON PARENT_SCOPE )
    #    set( ConfigMsg "${ConfigMsg} : **Repo[ ${repoStyle} ]**" )
    #endif()

    if( ARGN AND NOT VerbosityFlags )
        validateOption( Verbosity OPTIONAL Verbose VeryVerbose )
        if( Verbosity )
            if( Verbosity STREQUAL Verbose )
                set( VerbosityFlags "-V" PARENT_SCOPE )
            else()
              set( VerbosityFlags "-VV" PARENT_SCOPE )
            endif()
            set( ConfigMsg "${ConfigMsg} ${Verbosity}" )
        endif()
    endif()

    #######################################################################
    # Memory Checking: VALGRIND
    #--------------------------------
    if( ARGN AND NOT CTEST_MEMORYCHECK_COMMAND )
        validateOption( doMemCheck OPTIONAL memcheck )
        if( doMemCheck )
            set( WITH_MEMCHECK true PARENT_SCOPE )
            set( ConfigMsg "${ConfigMsg} - Memory checking enabled" )
        endif()
    endif()

    #######################################################################
    # Coverage Analysis: BullsEye
    #--------------------------------
    if( ARGN )
        validateOption( doCoverage OPTIONAL coverage )
        if( doCoverage )
            set( WITH_COVERAGE true PARENT_SCOPE )
            set( ConfigMsg "${ConfigMsg} - Code coverage enabled" )
        endif()
    endif()

    message( "ConfigInfo: ${ConfigMsg}" )    
endfunction()

#######################################################################
# General Configuration
#--------------------------------
function( configureCTest )
    if( ${ARGC} EQUAL 1 )
        set( BuildTag "${ARGV0}-" )
    endif()
    if( toolkitBuildDir )
        set( srcDir   ${toolkitSrcDir} )
        set( buildDir ${toolkitBuildDir} )
    else()
        if( BranchName )
            string( REGEX REPLACE "/" "_" pathToBranchName ${BranchName} )
            set( srcDir   "${RootSrcDir}/${Model}-${CTestSite}/${pathToBranchName}" )
            set( buildDir "${AutoScratch}/${Model}-${CTestSite}/obj/${pathToBranchName}-${BuildTag}${Tool}${CompilerVersion}-${Build}" )
        else()
            set( srcDir   "${RootSrcDir}/${Model}-${CTestSite}/trunk" )
            set( buildDir "${AutoScratch}/${Model}-${CTestSite}/obj/${BuildTag}${Tool}${CompilerVersion}-${Build}" )
        endif()
        set( toolkitSrcDir   ${srcDir} PARENT_SCOPE )
        set( toolkitBuildDir ${buildDir} PARENT_SCOPE )
    endif()
    get_filename_component( buildName ${buildDir} NAME )
    
    #Make sure eclipse related environment variable doesn't interfere
    unset( ENV{BINARY_DIR} )

    # Allow a submission from an individual's local space
    if( Model STREQUAL Experimental )
        find_path( localBuild CMakeLists.txt PATHS ENV PWD )
        if( localBuild )
            set( srcDir ${localBuild} )
        endif()
    endif()

    #######################################################################
    # CTest Critical Section
    #--------------------------------
    set( CTEST_BUILD_CONFIGURATION ${Build}     PARENT_SCOPE )
    set( CTEST_SITE                ${CTestSite} PARENT_SCOPE )
    set( CTEST_SOURCE_DIRECTORY    ${srcDir}    PARENT_SCOPE )
    set( CTEST_BINARY_DIRECTORY    ${buildDir}  PARENT_SCOPE )
    if( DEFINED ENV{SYS_TYPE} )
        set( CTEST_BUILD_NAME       "$ENV{SYS_TYPE}-${buildName}" PARENT_SCOPE )
    else()
        set( CTEST_BUILD_NAME       "linux-${buildName}" PARENT_SCOPE )
    endif()
    set( CTEST_CMAKE_GENERATOR  "Unix Makefiles" PARENT_SCOPE )
    #--------------------------------
    # Turn on verbose test output only if one fails(?)
    set( CTEST_OUTPUT_ON_FAILURE TRUE  PARENT_SCOPE)

endfunction()

#######################################################################
# Post results to Dashboard
#--------------------------------
function( CollateAndPostResults )

    if( NOT toolkitSrcDir )
        message( FATAL_ERROR "This routine should only be called AFTER configureCTest." )
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
        if( APPLE )
            set( CMAKE_USE_OPENSSL OFF )
        endif()
        ctest_submit()
        
    endforeach()
endfunction()

#######################################################################
# Initialize Memory analysis
# NOTE: only valgrind for the present
#--------------------------------
function( initializeMemoryChecking )
    find_program( CTEST_MEMORYCHECK_COMMAND NAMES valgrind)
    set( CTEST_MEMORYCHECK_COMMAND_OPTIONS   "${CTEST_MEMORYCHECK_COMMAND_OPTIONS} --trace-children=yes --track-origins=yes" PARENT_SCOPE )
    set( CTEST_MEMORYCHECK_SUPPRESSIONS_FILE "${CTEST_SOURCE_DIRECTORY}/cmake_files/valgrind_suppressions.txt" PARENT_SCOPE )
endfunction()

#######################################################################
# Code Coverage Analysis
# NOTE: only bullseye for the present
#--------------------------------
# turn on coverage (Bullseye if its available) for building. If the Bullseye binary directory is the
# first in the path, then it's version of the compiler calls are already in place.  This just specifies
 # whether they are called directly or after Bullseye modifies the calling args.
function( stopBullsEye )
    set(RES 1)
    execute_process(COMMAND ${CoverageToggle} -0 RESULT_VARIABLE RES)
    if(RES)
        message(FATAL_ERROR "Failed to disable Bullseye coverage system.  Could not run cov01 -0")
    endif()
endfunction()
#  Start bullseye wrapper executable
function( startBullsEye )
    set(RES 1)
    execute_process(COMMAND ${CoverageToggle} -1 RESULT_VARIABLE RES)
    if(RES)
        message(FATAL_ERROR "Failed to enable Bullseye coverage system.  Could not run cov01 -1")
    endif()
endfunction()
# initialize the bullseye coverage interface
function( setupCodeCoverage )

    # Find cov01, the bullseye tool to turn on/off coverage
    find_path( BullseyeBin NAMES cov01 
               PATHS /home/xshares/PROJECTS/mcatk/packages/bin )
    if( NOT BullseyeBin ) 
        message( FATAL_ERROR "The path to the BullsEye code coverage tool was not located." )
    endif()
    
    # prepend the bullseye binary directory to the path since it intercepts the compiler calls
    set( ENV{PATH}    "${BullseyeBin}:$ENV{PATH}" )
    set( ENV{COVFILE} "${CTEST_BINARY_DIRECTORY}/Bullseye.cov" )
    
    find_program( CoverageToggle cov01 )
    if(NOT CoverageToggle )
        message(FATAL_ERROR "Could not find [ cov01 ].  This is required for Bullseye coverage analysis.")
    endif()

    # Make sure the coverage is turned OFF before proceeding
    stopBullsEye()
    
    # Go ahead and define this even though it won't be invoked(?)
    find_program(CTEST_COVERAGE_COMMAND NAMES gcov)
    
endfunction()

