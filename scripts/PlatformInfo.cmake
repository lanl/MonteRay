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
endfunction()

function( PlatformInfo )

    if( ${Standalone} ) 
      set( Platform ${CMAKE_SYSTEM_NAME} CACHE INTERNAL "Platform on which this was configured.")
      return()
    endif()
	
    locateMCATKDir()
    
    site_name( FullSiteName )
    find_path( AutoBuildRoot 
               NAMES scripts Results 
               PATHS ${MCATKDir}/AutoBuilds
               PATH_SUFFIXES AutoBuilds )
    if( NOT AutoBuildRoot )
        message( FATAL_ERROR "Unable to locate autobuild directory on this computer." )
    endif()

    find_file( PlatformMetaFile PlatformDB.txt 
               PATHS ${AutoBuildRoot} 
               PATH_SUFFIXES scripts
    )

    if( NOT PlatformMetaFile )
        message( FATAL_ERROR "Unable to locate the platform description file." )
    endif()

    file( STRINGS ${PlatformMetaFile} Platforms )
    list( GET Platforms 0 Headers )
    string( REGEX REPLACE "," " - " Headers ${Headers} )
    list( REMOVE_AT Platforms 0 )

    foreach( line ${Platforms} )
        # convert line to a list of entries
        string( REGEX REPLACE "[ ,]+" ";" line ${line} )
        list( LENGTH line NEntries )
        list( GET line 0 RegexID )
        string( REGEX MATCH "^${RegexID}" SiteID ${FullSiteName} )
#        message( "RegexID ${RegexID} ${line}" )
        if( SiteID )
            # Cluster short name
            set( ClusterID ${SiteID} CACHE STRING "Abbreviated name of compute cluster" )
            # Host Site
            list( GET line 1 HostSite )
            set( HostDomain ${HostSite} CACHE STRING "Compute domain of the cluster" )
            # NUMA Info
            list( GET line 2 NumberNUMA )
            list( GET line 3 JobsPerNUMA )
            if( NumberNUMA STREQUAL "NA" )
            else()
                set( NumNUMA        ${NumberNUMA} CACHE STRING "Number of NUMA domains on cluster" )
                set( MaxJobsPerNUMA ${JobsPerNUMA} CACHE STRING "Maxiumum number of concurrent jobs allowed on a NUMA domain." )
            endif()
            # Scratch Directory
            list( GET line 4 ScratchDir )
            set( AutoScratch ${ScratchDir}/$ENV{USER}/AUTOBUILD CACHE PATH "Scratch directory on the cluster"  )
            # Batch system basics
            if( NEntries GREATER 5 )
                list( GET line 5 batchSystem )
                set( BatchSystem ${batchSystem} CACHE STRING "Job control system" )
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
            else()
            endif()
            break()
        endif()
    endforeach()
    
    if( NOT SiteID )
        message( FATAL_ERROR "System [ ${FullSiteName} ] requires an entry in [ ${PlatformMetaFile} ]." )
    endif()

    if( HostDomain STREQUAL lanl )
        set( CTestSite ${ClusterName} CACHE STRING "Site reported to ctest" )
        # Platform ID case -- LANL HPC
        set( Platform ${ClusterName} CACHE INTERNAL "Platform on which this was configured.")
    elseif( HostDomain STREQUAL xdiv )
        set( CTestSite ${ClusterID} CACHE STRING "Site reported to ctest" )
        # Platform ID for xdiv systems
        if( DEFINED ENV{PlatformOS} )
            set( Platform $ENV{PlatformOS} CACHE INTERNAL "Platform on which this was configured." )
        else()
            message( FATAL_ERROR "User must set environment variable *PlatformOS* in mcatk module file." )
        endif()
    else()
        set( CTestSite ${ClusterName}-${HostDomain} CACHE STRING "Site reported to ctest" )
    endif()

    # If this isn't part of a system with a job controller, we're done
    if( NOT BatchSystem )
        return()
    endif()

    if( BatchSystem STREQUAL MOAB )
        find_program( SubmitCmd msub 
                      PATHS $ENV{MOABHOMEDIR}/bin
                      ENV PATH
                    )
        if( HostDomain STREQUAL "llnl" )
            # Generic case for LLNL TLCC
            set( Platform $ENV{SYS_TYPE} CACHE INTERNAL "Platform on which this was configured.")
        endif()

    # Special case for LLNL BlueGeneQ clusters (seq, rzuseq, etc.)
    elseif( BatchSystem STREQUAL slurm )
        find_program( SubmitCmd srun )
        set( SRUN_SERIAL ${SubmitCmd} --partition=${JobQueue} -n 1 -t 1:30:00 CACHE DOC "Required invocation line on BlueGeneQ" )

        set( Platform BlueGeneQ CACHE INTERNAL "Platform on which this was configured" )

    # Special case for SNL batching
    elseif( BatchSystem STREQUAL SBATCH )
        find_program( SubmitCmd sbatch )
        
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

  # Linux:
  find_program( APRUN aprun DOC "Cray/MPICH utility for dispatching parallel jobs to backends." )
  find_program( LSCPU lscpu DOC "Linux utility for analyzing runtime hardware." )
  if( LSCPU )
      if( APRUN )
          execute_process( COMMAND ${APRUN} --quiet ${LSCPU} -e=core OUTPUT_VARIABLE info OUTPUT_STRIP_TRAILING_WHITESPACE )
      else()
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

  # Mac:
  if(APPLE)
    find_program(cmd_sys_pro "system_profiler")
    if(cmd_sys_pro)
      execute_process(COMMAND ${cmd_sys_pro} OUTPUT_VARIABLE info)
      string(REGEX REPLACE "^.*Total Number Of Cores: ([0-9]+).*$" "\\1"
        PROCESSOR_COUNT "${info}")
    endif()
  endif()

  # Windows:
  if(WIN32)
    set(PROCESSOR_COUNT "$ENV{NUMBER_OF_PROCESSORS}")
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
            set( srcDir   "${AutoScratch}/${Model}-${CTestSite}/${BranchName}" )
            set( buildDir "${AutoScratch}/${Model}-${CTestSite}/obj/${BranchName}-${BuildTag}${Tool}${CompilerVersion}-${Build}" )
        else()
            set( srcDir   "${AutoScratch}/${Model}-${CTestSite}/trunk" )
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
# MCATK Subversion Access
#--------------------------------
function( initializeSVN )
    # insure we can find the 'svn' executable
    find_package( Subversion REQUIRED )

    locateMCATKDir()
    
    # Find software repository (mirror or master)
    find_path( MCATK_Repository 
               NAMES db 
               PATHS ${MCATKDir}
               PATH_SUFFIXES svn/repo 
                             master 
                             mirror
              )
    if( NOT MCATK_Repository )
        message( FATAL_ERROR "Unable to locate repository." )
    endif()

    if( BranchName )
        set( ToolkitBranch "mcatk/branch/${BranchName}" )
    else()
        set( ToolkitBranch "mcatk/trunk" )
    endif()

    # Note: you can't use variables set into parent scope. Weird but true.
    set( svnPath "file://${MCATK_Repository}/${ToolkitBranch}" )
    set( CTEST_SVN_COMMAND ${Subversion_SVN_EXECUTABLE} PARENT_SCOPE )
#   message( FATAL_ERROR " ctest src [ ${CTEST_SOURCE_DIRECTORY} ]" )
    set( CTEST_CHECKOUT_COMMAND "${Subversion_SVN_EXECUTABLE} co ${svnPath} ${CTEST_SOURCE_DIRECTORY}" PARENT_SCOPE )
    set( CTEST_UPDATE_COMMAND "${Subversion_SVN_EXECUTABLE}" PARENT_SCOPE )

    # Remove source directory if it appears corrupt
    if( EXISTS ${CTEST_SOURCE_DIRECTORY} )
        if( EXISTS ${CTEST_SOURCE_DIRECTORY}/CMakeLists.txt )
            message( STATUS "SVN: Directory already exists.  Checking..." )
            set( CTEST_CHECKOUT_COMMAND "${Subversion_SVN_EXECUTABLE} info ${CTEST_SOURCE_DIRECTORY}" PARENT_SCOPE )
        else()
            execute_process( COMMAND ${CMAKE_COMMAND} -E remove_directory ${CTEST_SOURCE_DIRECTORY} )
        endif()
    endif()
endfunction()

#######################################################################
# Post results to Dashboard
#--------------------------------
function( CollateAndPostResults )

    if( NOT toolkitSrcDir )
        if( BranchName )
            set( toolkitSrcDir   "${AutoScratch}/${Model}-${CTestSite}/${BranchName}" )
            set( toolkitBuildDir "${AutoScratch}/${Model}-${CTestSite}/obj/${BranchName}-${Tool}${CompilerVersion}-${Build}" )
        else()
            set( toolkitSrcDir   "${AutoScratch}/${Model}-${CTestSite}/trunk" )
            set( toolkitBuildDir "${AutoScratch}/${Model}-${CTestSite}/obj/${Tool}${CompilerVersion}-${Build}" )
        endif()
    endif()

    # set the options into a single argument
    set( ScriptDefines -DModel=${Model} -DTool=${Tool} -DBuild=${Build} -DtoolkitSrcDir=${toolkitSrcDir} -DtoolkitBuildDir=${toolkitBuildDir} )

    find_file( AutoReportScript 
               ctestReport.cmake 
               PATHS ${CMAKE_CURRENT_LIST_DIR} )

    set( CmdLine ${CMAKE_CTEST_COMMAND} ${ScriptDefines} -S ${AutoReportScript} )

    execute_process( COMMAND ${CmdLine}
                     RESULT DashboardResult 
                     OUTPUT_VARIABLE DashboardOut
                     ERROR_VARIABLE DashboardErr )

    if( DashboardResult EQUAL 0 )
        message( "Result submission successful. [ ${Model}-${Tool}-${Build} ]" )
    else()
        message( "While executing : [ ${CmdLine} ]" )
        message( FATAL_ERROR "Dashboard submission encountered difficulties.\n ERR: ${DashboardErr}" )
    endif()

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

