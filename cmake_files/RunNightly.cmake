################################################################################
# Analyze results of unit test runs

# Find the location of numactl on the processor where the job is executing.
find_program( NUMACTL numactl )

set( MaxNUMA @NumNUMA@ )

# This block here is to avoid race conditions. If absent, two jobs starting simultaneously could try
# to run on the same node. This would cause a resource conflict and slow the execution time 
# drastically. When the short job finished, it would delete the file (i.e. the resource lock) and the
# next job would incorrectly think the resource was available for more work. This is most likely to 
# occur on job submission.
file( GLOB Busy "@CMAKE_BINARY_DIR@/Node[0-9]_BUSY" )
if( NOT Busy )
    string( RANDOM LENGTH 2 ALPHABET "0123456789" RANDOM_SEED ${TestID} WtCount )
    execute_process( COMMAND sleep ".${WtCount}" )
#    message( "Waited [ .${WtCount} ] sec." )
endif()

# Before selecting a NUMA domain/node on which to run, check the file system to see which ones are
# NOT in use.
set( NODE 0 )
set( lockFile @CMAKE_BINARY_DIR@/Node${NODE}_BUSY )
while( EXISTS ${lockFile} )
    math( EXPR NODE "${NODE} + 1" )
    if( NODE EQUAL MaxNUMA )
        set( NODE 0 )
    endif()
    set( lockFile @CMAKE_BINARY_DIR@/Node${NODE}_BUSY )
endwhile()

#message( "Starting [ ${TestID}:${SUITE} ] on Node [ ${NODE} ]" )

# Once an available node has been identified, lock the resource by creating an empty file.
execute_process( COMMAND ${CMAKE_COMMAND} -E touch ${lockFile} )

# Use numactl to restrict the execution to the cpus and memory of a single numa domain.
set( NUMAArgs ${NUMACTL} --cpunodebind=${NODE} --membind=${NODE} )
execute_process( 
                 COMMAND @MPIEXEC@ @MPIEXEC_NUMPROC_FLAG@ @mpiNProcs@ ${NUMAArgs} @NightlyApp@ ${SUITE}
                 TIMEOUT 3600
                 RESULT_VARIABLE result
               )
               
# Upon completion, remove the resource lock (i.e. the file).
execute_process( COMMAND ${CMAKE_COMMAND} -E remove -f ${lockFile} )

# If there were any errors encountered, report them up the line.
if( NOT result EQUAL 0 )
    message( SEND_ERROR ${result} )
endif()
