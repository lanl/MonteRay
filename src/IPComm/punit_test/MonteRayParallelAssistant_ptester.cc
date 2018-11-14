#include <UnitTest++.h>

#include <iostream>
#include <unistd.h>

#include "MonteRayParallelAssistant.hh"
#include "GPUUtilityFunctions.hh"

namespace MonteRayParallelAssistant_ptester_namespace{

SUITE( MonteRayParallelAssistant_ptester ){
    using namespace MonteRay;

    class setup{
     public:
         setup(){
             MPI_Comm_split_type( MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shared_memory_communicator );
             MPI_Comm_size( shared_memory_communicator, &shared_memory_size );
             MPI_Comm_rank( shared_memory_communicator, &shared_memory_rank );
             MPI_Comm_size(MPI_COMM_WORLD, &world_size);
             MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
         }
         ~setup(){}

         MPI_Comm shared_memory_communicator;
         int shared_memory_size;
         int shared_memory_rank;
         int world_size;
         int world_rank;
     };

    TEST_FIXTURE(setup, ctor ) {
        const MonteRayParallelAssistant& PA( MonteRayParallelAssistant::getInstance() );
        CHECK_EQUAL( true, PA.isParallel() );

        const bool debug = false;

        if( debug ) {
            char hostname[1024];
            gethostname(hostname, 1024);
            std::cout << "MonteRayParallelAssistnat_ptester :: test ctor -- hostname = " << hostname <<
                     ", world_rank=" << PA.getWorldRank() <<
                     ", world_size=" << PA.getWorldSize() <<
                     ", shared_memory_rank=" << PA.getSharedMemoryRank() <<
                     ", shared_memory_size=" << PA.getSharedMemorySize() <<
                     ", work_group_rank=" << PA.getWorkGroupRank() <<
                     ", work_group_size=" << PA.getWorkGroupSize() <<
                     ", inter_work_group_rank=" << PA.getInterWorkGroupRank() <<
                     ", inter_work_group_size=" << PA.getInterWorkGroupSize() <<
                     "\n";
        }

        if( world_size == 1 and world_rank == 0 ) {
            CHECK_EQUAL( 0, PA.getWorldRank() );
            CHECK_EQUAL( 1, PA.getWorldSize() );
            CHECK_EQUAL( 0, PA.getSharedMemoryRank() );
            CHECK_EQUAL( 1, PA.getSharedMemorySize() );
            CHECK_EQUAL( 0, PA.getWorkGroupRank() );
            CHECK_EQUAL( 1, PA.getWorkGroupSize() );
            CHECK_EQUAL( 0, PA.getInterWorkGroupRank() );
            CHECK_EQUAL( 1, PA.getInterWorkGroupSize() );
        }
    }

    TEST_FIXTURE(setup, tally ) {
        const MonteRayParallelAssistant& PA( MonteRayParallelAssistant::getInstance() );

        double value = 0.0;
        if( PA.getWorkGroupRank() == 0 ) {
            value = 10.0;
        } else {
            value = 1.0;
        }

        double workGroupTotal = 0.0;
        MPI_Reduce( &value, &workGroupTotal, 1, MPI_DOUBLE, MPI_SUM, 0, PA.getWorkGroupCommunicator() );

        double total = 0.0;

        if( PA.getInterWorkGroupCommunicator() != MPI_COMM_NULL ) {
            MPI_Reduce( &workGroupTotal, &total, 1, MPI_DOUBLE, MPI_SUM, 0, PA.getInterWorkGroupCommunicator() );
        }

        if( PA.getWorldRank() == 0 ) {
            double expected = 10.0*PA.getInterWorkGroupSize() + 1.0*(PA.getWorldSize()- PA.getInterWorkGroupSize());
            CHECK_CLOSE( expected, total, 1e-14 );
        } else if ( PA.getWorkGroupRank() == 0 ){
            double expected = ( 10.0 + 1.0*(PA.getWorkGroupSize()-1) );
            CHECK_CLOSE( expected, workGroupTotal, 1e-14 );
        }
    }


}

} // end namespace
