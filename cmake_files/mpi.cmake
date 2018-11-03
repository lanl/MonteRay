################################################
#
# Common cmake stuff for the MPI library stuff
#
################################################

find_package(MPI REQUIRED)

execute_process(COMMAND ${MPIEXEC} --version
                OUTPUT_VARIABLE MPI_VERSION_STRING 
                ERROR_VARIABLE  MPI_VERSION_STRING
                OUTPUT_STRIP_TRAILING_WHITESPACE)
string( REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" MPI_VERSION ${MPI_VERSION_STRING} ) 
if( NOT MPI_VERSION )
    string( REGEX MATCH "[0-9]+\\.[0-9]+" MPI_VERSION ${MPI_VERSION_STRING} ) 
endif()
string( REGEX REPLACE "\\." ";" MPI_VERSION ${MPI_VERSION} ) 
list( GET MPI_VERSION 0 MPI_MAJOR_VERSION ) 
list( GET MPI_VERSION 1 MPI_MINOR_VERSION ) 
list( GET MPI_VERSION 2 MPI_REVISION_VERSION ) 
message(STATUS "cmake/mpi.cmake -- MPI version is ${MPI_MAJOR_VERSION}.${MPI_MINOR_VERSION}.${MPI_REVISION_VERSION}" )
            
# For testing, we would like the mpi jobs to run concurrently and not all crowd on CPU #0
if( MPI_MAJOR_VERSION GREATER 9)
   # Sierra/Shark the MPI version under the hood of Spectrum is 10.x.x. 
   # Need a way to make this more LLNL/Sierra specific... PlatformOS is power8?
    set( MPIEXEC_PREFLAGS ${MPIEXEC_PREFLAGS} --mca btl vader,self --bind-to none )
    set( MPIEXEC_PREFLAGS ${MPIEXEC_PREFLAGS} --mca timer_require_monotonic 0 )
    message( "-- mpi.cmake - Setting mpiexec flags for Sierras MPI" )            
elseif( MPI_MAJOR_VERSION EQUAL 3 ) 
  set( MPIEXEC_PREFLAGS ${MPIEXEC_PREFLAGS} --mca btl vader,self --bind-to none )
  set( MPIEXEC_PREFLAGS ${MPIEXEC_PREFLAGS} --mca timer_require_monotonic 0 )
else()
  if( MPI_MINOR_VERSION EQUAL 4 )
      set( MPIEXEC_PREFLAGS ${MPIEXEC_PREFLAGS} --mca mpi_paffinity_alone 0 )
  elseif( MPI_MINOR_VERSION EQUAL 6 )
      set( MPIEXEC_PREFLAGS ${MPIEXEC_PREFLAGS} --mca btl sm,self --bind-to-none )
  else()
      set( MPIEXEC_PREFLAGS ${MPIEXEC_PREFLAGS} --mca btl sm,self --bind-to none )
      set( MPIEXEC_PREFLAGS ${MPIEXEC_PREFLAGS} --mca timer_require_monotonic 0 )
  endif()
endif()

# This turns OFF the mpi stuff under OpenMPI
add_definitions( -DOMPI_SKIP_MPICXX )