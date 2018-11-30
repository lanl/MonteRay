#ifndef MONTERAYPARALLELASSISTANT_HH_
#define MONTERAYPARALLELASSISTANT_HH_

#include <vector>
#include <mpi.h>

namespace MonteRay {

/// MonteRayParallelAssistant maintains information
/// about a parallel run.
class MonteRayParallelAssistant {

private:
    // private member data

    bool parallel = false;
    bool useSingleProcWorkGroup = false;

    int world_size = 0;
    int world_rank = 0;
    int shared_memory_size = 0;
    int shared_memory_rank = 0;
    int deviceID = 0;

    // MonteRay's duplicate of MPI_COMM_WORLD
    MPI_Comm MONTERAY_COMM_WORLD;

    // Shared memory communicator
    MPI_Comm MONTRERAY_COMM_SHMEM;

    // A work group - Can be one work group per node
    // or one work group per GPU with multiple work groups
    // per node.
    MPI_Comm WORK_GROUP_COMM;
    int WORK_GROUP_COMM_SIZE = 0;
    int WORK_GROUP_COMM_RANK = 0;

    // Inter-communicator between work groups, all rank 0
    // processes of all the work groups
    MPI_Comm INTER_WORK_GROUP_COMM;
    int INTER_WORK_GROUP_COMM_SIZE = 0;
    int INTER_WORK_GROUP_COMM_RANK = 0;

private:
    MonteRayParallelAssistant();
    MonteRayParallelAssistant( MonteRayParallelAssistant const & ) = delete;
    void operator=( MonteRayParallelAssistant const & )    = delete;


    ~MonteRayParallelAssistant() {
        if( parallel ) {
//            MPI_Comm_free( &INTER_WORK_GROUP_COMM );
//            MPI_Comm_free( &WORK_GROUP_COMM );
//            MPI_Comm_free( &MONTRERAY_COMM_SHMEM );
//            MPI_Comm_free( &MONTERAY_COMM_WORLD );
        }
    }

public:
    static MonteRayParallelAssistant& getInstance() {
        static MonteRayParallelAssistant instance;
        return instance;
    }

    bool isParallel() const { return parallel; }

    int getWorldSize() const { return world_size; }
    int getWorldRank() const { return world_rank; }
    int getSharedMemorySize() const { return shared_memory_size; }
    int getSharedMemoryRank() const { return shared_memory_rank; }

    const MPI_Comm& getWorkGroupCommunicator() const { return WORK_GROUP_COMM; }
    int getWorkGroupSize() const { return WORK_GROUP_COMM_SIZE; }
    int getWorkGroupRank() const { return WORK_GROUP_COMM_RANK; }

    const MPI_Comm& getInterWorkGroupCommunicator() const { return INTER_WORK_GROUP_COMM; }
    int getInterWorkGroupSize() const { return INTER_WORK_GROUP_COMM_SIZE; }
    int getInterWorkGroupRank() const { return INTER_WORK_GROUP_COMM_RANK; }

    int calcDeviceID( int group_size, int numGPUs, int rank ) const {
        return ( rank * numGPUs ) / group_size  ;
    }

    int getDeviceID(void) const { return deviceID; }

    bool usingSingleProcWorkGroup() const { return useSingleProcWorkGroup; }
};

void setMonteRayStackSize( size_t size);

bool isWorkGroupMaster(void);

} // end namespace



#endif /* MONTERAYPARALLELASSISTANT_HH_ */
