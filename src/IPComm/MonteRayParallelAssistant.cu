#include <mpi.h>

#include "MonteRayParallelAssistant.hh"
#include "GPUUtilityFunctions.hh"

namespace MonteRay {

MonteRayParallelAssistant::MonteRayParallelAssistant() {
    int mpi_initialized = 0;
    MPI_Initialized( &mpi_initialized );

    if( !mpi_initialized ) {
//        std::cout << "MonteRayParallelAssistant::MonteRayParallelAssistant() -- MPI is not initialized.\n";
        return;
    }
//    std::cout << "MonteRayParallelAssistant::MonteRayParallelAssistant() -- MPI has been initialized.\n";


    MPI_Comm_dup(MPI_COMM_WORLD, &MONTERAY_COMM_WORLD);

    parallel = true;
    MPI_Comm_size(MONTERAY_COMM_WORLD, &world_size);
    MPI_Comm_rank(MONTERAY_COMM_WORLD, &world_rank);

    MPI_Comm_split_type( MONTERAY_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &MONTRERAY_COMM_SHMEM );
    MPI_Comm_size( MONTRERAY_COMM_SHMEM, &shared_memory_size );
    MPI_Comm_rank( MONTRERAY_COMM_SHMEM, &shared_memory_rank );


    // Only get the number of GPUs once per node
    int numberOfGPUs = 0;
    if( shared_memory_rank == 0 ){
        numberOfGPUs = getNumberOfGPUS();
        if( numberOfGPUs == 0 ) {
            // if not using GPUs, setup to use on cpu process per work group
            numberOfGPUs = shared_memory_size;
        }
    }
    // scatter numberOfGPUs to all processes on the node
    MPI_Bcast( &numberOfGPUs, 1, MPI_INT, 0, MONTRERAY_COMM_SHMEM);

    // split MONTRERAY_COMM_SHMEM into numberOfGPUs work groups
    if( numberOfGPUs <= 1 ) {
        deviceID = 0;
        MPI_Comm_dup(MONTRERAY_COMM_SHMEM, &WORK_GROUP_COMM);
    } else {
        deviceID = calcDeviceID(shared_memory_size, numberOfGPUs, shared_memory_rank );
        MPI_Comm_split(MONTRERAY_COMM_SHMEM, deviceID, shared_memory_rank, &WORK_GROUP_COMM);
    }
    MPI_Comm_size( WORK_GROUP_COMM, &WORK_GROUP_COMM_SIZE );
    MPI_Comm_rank( WORK_GROUP_COMM, &WORK_GROUP_COMM_RANK );
    setCudaDevice( deviceID );

    // Create inter-working group communicator
    if( WORK_GROUP_COMM_RANK == 0 ) {
        MPI_Comm_split( MONTERAY_COMM_WORLD, 0, world_rank, &INTER_WORK_GROUP_COMM);
        MPI_Comm_rank( INTER_WORK_GROUP_COMM, &INTER_WORK_GROUP_COMM_RANK );
        MPI_Comm_size( INTER_WORK_GROUP_COMM, &INTER_WORK_GROUP_COMM_SIZE );
    } else {
        MPI_Comm_split( MONTERAY_COMM_WORLD, MPI_UNDEFINED, world_rank, &INTER_WORK_GROUP_COMM);
        INTER_WORK_GROUP_COMM_RANK = -1;
        INTER_WORK_GROUP_COMM_SIZE = 0;
    }
}

void setMonteRayStackSize( size_t size) {
    const MonteRayParallelAssistant& PA( MonteRayParallelAssistant::getInstance() );
    if( PA.getWorkGroupRank() == 0 ) {
        setCudaStackSize( size );
    }
}

bool isWorkGroupMaster(void) {
    const MonteRayParallelAssistant& PA( MonteRayParallelAssistant::getInstance() );
    if( PA.getWorkGroupRank() == 0 ) {
        return true;
    }
    return false;
}


} // end namespace
