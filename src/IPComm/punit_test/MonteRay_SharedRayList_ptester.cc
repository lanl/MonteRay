#include <UnitTest++.h>

#include <iostream>
#include <mpi.h>

#include "MonteRay_SharedRayList.hh"
#include "Ray.hh"

namespace SharedRayList_ptester{

SUITE( mpi_shared_rayList_tester ){

    using namespace MonteRay;
    typedef MonteRay::Ray_t<1> ray_t;
    typedef SharedRayList<ray_t>  rayList_t;

    class ParticleSetup{
    public:
        ParticleSetup(){
            particle.pos[0] = 1.0f;
            particle.pos[1] = 0.0f;
            particle.pos[2] = 0.0f;
            particle.dir[0] = 1.0f;
            particle.dir[1] = 0.0f;
            particle.dir[2] = 0.0f;
            particle.energy[0] = 14.0f;
            particle.weight[0] = 1.0f;
            particle.index = 1U;
        }
        ~ParticleSetup(){}
        ray_t particle;

        friend bool operator< (const ParticleSetup &lhs, const ParticleSetup &rhs);
        friend bool operator<= (const ParticleSetup &lhs, const ParticleSetup &rhs);
    };

    bool operator< (const ParticleSetup &lhs, const ParticleSetup &rhs) {
        return lhs.particle.index < rhs.particle.index;
    }

    bool operator<= (const ParticleSetup &lhs, const ParticleSetup &rhs) {
        return lhs.particle.index <= rhs.particle.index;
    }

    class TestMasterList{
    public:
        TestMasterList(){
        }
        ~TestMasterList(){
        }

        static bool sortFunction(const ray_t& point1, const ray_t& point2 ) {
            if( point1.index < point2.index ) {
                return true;
            }
            return false;
        }

        typedef float gpuFloatType_t;
        void add(const void* collision, unsigned N) {
            const ray_t* ptrCollision = (const ray_t*) collision;
            for( auto i = 0; i< N; ++i) {
                masterList.push_back( *(ptrCollision+i) );
            }
        }

        ray_t get(unsigned i) {
            return masterList.at(i);
        }

        unsigned size() const { return masterList.size(); }
        std::vector<ray_t> masterList;

        void flush(bool) {
            //             masterList.clear();
        }

        void clearTally(void) {
            masterList.clear();
        }

        void debugPrint(void) {
            std::cout << "******************************************************************************************\n";
            for( auto i = 0; i< size() ; ++i) {
                std::cout << "Debug: TestMasterList::debugPrint -- i= " << i;
                ray_t particle = get(i);
                std::cout << " x = " << particle.pos[0];
                std::cout << " y = " << particle.pos[1];
                std::cout << " z = " << particle.pos[2];
                std::cout << " u = " << particle.dir[0];
                std::cout << " v = " << particle.dir[1];
                std::cout << " w = " << particle.dir[2];
                std::cout << " E = " << particle.energy;
                std::cout << " W = " << particle.weight;
                std::cout << " index = " << particle.index;
                std::cout << "\n";
            }
            std::cout << "******************************************************************************************\n";

            std::sort (masterList.begin(), masterList.end(), sortFunction );
            std::cout << "******************************************************************************************\n";
            for( auto i = 0; i< size() ; ++i) {
                std::cout << "Debug: TestMasterList::debugPrint -- i= " << i;
                ray_t particle = get(i);
                std::cout << " x = " << particle.pos[0];
                std::cout << " y = " << particle.pos[1];
                std::cout << " z = " << particle.pos[2];
                std::cout << " u = " << particle.dir[0];
                std::cout << " v = " << particle.dir[1];
                std::cout << " w = " << particle.dir[2];
                std::cout << " E = " << particle.energy;
                std::cout << " W = " << particle.weight;
                std::cout << " index = " << particle.index;
                std::cout << "\n";
            }
            std::cout << "******************************************************************************************\n";
        }



    };

    class setup{
    public:
        setup() : PA( MonteRayParallelAssistant::getInstance() ) {
            shared_memory_size = PA.getWorkGroupSize();
            shared_memory_rank = PA.getWorkGroupRank();
            MPI_Comm_dup( PA.getWorkGroupCommunicator(), &shared_memory_communicator);
        }
        ~setup(){}

        MPI_Comm shared_memory_communicator;

        const MonteRayParallelAssistant& PA;
        int shared_memory_size;
        int shared_memory_rank;

        TestMasterList master;
        ParticleSetup p;
    };

    TEST( ctor1_defaults ) {
         //        printf("Debug: ctor1_defaults test\n");
         const MonteRayParallelAssistant& PA( MonteRayParallelAssistant::getInstance() );

         TestMasterList master;
         rayList_t list(master, 10, 10);
         CHECK_EQUAL( true, list.isUsingMPI() );
         CHECK_EQUAL( 10, list.getNBuckets() );

         if( PA.getWorkGroupRank() == 0 ) {
             CHECK_EQUAL( 0, list.getCurrentBucket(0) );
         }
     }

    TEST( ctor2_defaults ) {
        //        printf("Debug: ctor_defaults test\n");
        const MonteRayParallelAssistant& PA( MonteRayParallelAssistant::getInstance() );

        TestMasterList master;
        rayList_t list(master, 10, PA.getWorkGroupRank(), PA.getWorkGroupSize(), PA.isParallel(), 10);
        CHECK_EQUAL( true, list.isUsingMPI() );
        CHECK_EQUAL( 10, list.getNBuckets() );

        if( PA.getWorkGroupRank() == 0 ) {
            CHECK_EQUAL( 0, list.getCurrentBucket(0) );
        }
    }

    TEST_FIXTURE(setup, getBucketHeader_with_a_particle ) {
        if(shared_memory_size != 2 ) { return; }

        rayList_t list(master, 10, shared_memory_rank, shared_memory_size, true,10);

        bucket_header_t* header;
        header = list.getBucketHeader( 1, 0 );
        if( shared_memory_rank == 0 ) {
            CHECK_EQUAL(0, header->size );
            CHECK_EQUAL(false, header->done );
        }
        MPI_Barrier( shared_memory_communicator );

        if(shared_memory_rank==1 ) {
            list.addCollision(1,p.particle);
        }
        MPI_Barrier( shared_memory_communicator );
        if( shared_memory_rank == 0 ) {
            CHECK_EQUAL(1, header->size );
            CHECK_EQUAL(true, header->done );
        }
    }

    TEST_FIXTURE(setup, size ) {
        if(shared_memory_size != 2 ) { return; }

        rayList_t list(master, 200, shared_memory_rank, shared_memory_size, true,10);
        if( shared_memory_rank == 0 ) {
            CHECK_EQUAL( 100, list.getParticlesPerRank() );
            CHECK_EQUAL( 10, list.getParticlesPerBucket() );
        }
    }
    TEST_FIXTURE(setup, isBucketFull_empty ) {
        if(shared_memory_size != 2 ) { return; }

        rayList_t list(master, 200, shared_memory_rank, shared_memory_size, true,10);

        if( shared_memory_rank == 0 ) {
            CHECK_EQUAL( false, list.isBucketFull(1,0) );
        }
    }

    TEST_FIXTURE(setup, addCollisionToRank1_not_full ) {
        if(shared_memory_size != 2 ) { return; }

        rayList_t list(master, 100, shared_memory_rank, shared_memory_size, true,10);

        bucket_header_t* header;
        header = list.getBucketHeader( 1, 0 );
        if( shared_memory_rank == 0 ) {
            CHECK_EQUAL(0, header->size );
            CHECK_EQUAL(false, header->done );
            CHECK_EQUAL( 5, list.getParticlesPerBucket() );
        }
        MPI_Barrier( shared_memory_communicator );

        if(shared_memory_rank==1 ) {
            list.addCollision(1,p.particle);
        }
        MPI_Barrier( shared_memory_communicator );

        if( shared_memory_rank == 0 ) {
            CHECK_EQUAL( 1U, list.bucketSize(1,0) );
            CHECK_EQUAL( false, list.isBucketFull(1,0) );
            CHECK_EQUAL( false, list.isBucketDone(1,0) );
            CHECK_EQUAL( 0, list.getCurrentBucket(1) );
        }
    }

    TEST_FIXTURE(setup, addCollisionToRank1_full ) {
        if(shared_memory_size != 2 ) { return; }

        rayList_t list(master, 10, shared_memory_rank, shared_memory_size, true,10);

        bucket_header_t* header;
        header = list.getBucketHeader( 1, 0 );

        if( shared_memory_rank == 0 ) {
            CHECK_EQUAL(0, header->size );
            CHECK_EQUAL(false, header->done );
            CHECK_EQUAL( 1, list.getParticlesPerBucket() );
        }

        MPI_Barrier( shared_memory_communicator );

        if(shared_memory_rank==1 ) {
            list.addCollision(1,p.particle);
        }
        MPI_Barrier( shared_memory_communicator );

        if( shared_memory_rank == 0 ) {
            CHECK_EQUAL( 1U, list.bucketSize(1,0) );
            CHECK_EQUAL( true, list.isBucketFull(1,0) );
            CHECK_EQUAL( true, list.isBucketDone(1,0) );
            CHECK_EQUAL( 1, list.getCurrentBucket(1) );
        }
    }

    TEST_FIXTURE(setup, add_a_Particle_goes_to_master_from_Rank0 ) {
        if(shared_memory_size != 2 ) { return; }

        rayList_t list(master, 10, shared_memory_rank, shared_memory_size, true,10);

        bucket_header_t* header;
        header = list.getBucketHeader( 1, 0 );

        if( shared_memory_rank == 0 ) {
            CHECK_EQUAL(0, header->size );
            CHECK_EQUAL(false, header->done );
            CHECK_EQUAL( 1, list.getParticlesPerBucket() );
        }

        MPI_Barrier( shared_memory_communicator );

        if(shared_memory_rank==0 ) {
            p.particle.index = 99;
            list.addCollision(0,p.particle);
        }
        MPI_Barrier( shared_memory_communicator );

        if(shared_memory_rank==0 ) {
            CHECK_EQUAL( 0U, list.bucketSize(0,0) );
            CHECK_EQUAL( false, list.isBucketFull(0,0) );
            CHECK_EQUAL( false, list.isBucketDone(0,0) );
            CHECK_EQUAL( 0, list.getCurrentBucket(0) );

            CHECK_EQUAL( 1U, master.size() );
            CHECK_EQUAL( 99U, master.get(0).index );
        }
    }

    TEST_FIXTURE(setup, copyToMaster ) {
        if(shared_memory_size != 2 ) { return; }

        rayList_t list(master, 200, shared_memory_rank, shared_memory_size, true,10);

        bucket_header_t* header;
        header = list.getBucketHeader( 1, 0 );

        if( shared_memory_rank == 0 ) {
            CHECK_EQUAL(0, header->size );
            CHECK_EQUAL(false, header->done );
            CHECK_EQUAL( 10, list.getParticlesPerBucket() );
        }

        MPI_Barrier( shared_memory_communicator );

        if(shared_memory_rank==1 ) {
            for( unsigned i=0; i<20; ++i){
                p.particle.index = i;
                list.addCollision(1,p.particle);
            }
        }
        MPI_Barrier( shared_memory_communicator );

        if( shared_memory_rank == 0 ) {
            CHECK_EQUAL( 10U, list.bucketSize(1,0) );
            CHECK_EQUAL( 10U, list.bucketSize(1,1) );
            CHECK_EQUAL( 0U, list.getCollisionFromLocal(1,0,0).index );
            CHECK_EQUAL( 1U, list.getCollisionFromLocal(1,0,1).index );
            CHECK_EQUAL( 8U, list.getCollisionFromLocal(1,0,8).index );
            CHECK_EQUAL( 9U, list.getCollisionFromLocal(1,0,9).index );
            CHECK_EQUAL( 10U, list.getCollisionFromLocal(1,0,10).index );
            CHECK_EQUAL( 11U, list.getCollisionFromLocal(1,0,11).index );
            CHECK_EQUAL( 18U, list.getCollisionFromLocal(1,0,18).index );
            CHECK_EQUAL( 19U, list.getCollisionFromLocal(1,0,19).index );
            CHECK_EQUAL( true, list.isBucketDone(1,0) );
            CHECK_EQUAL( true, list.isBucketDone(1,1) );

            list.copyToMaster(1);
            CHECK_EQUAL( 0U, master.get(0).index );
            CHECK_EQUAL( 9U, master.get(9).index );
            CHECK_EQUAL( 10U, master.get(10).index );
            CHECK_EQUAL( 19U, master.get(19).index );
            CHECK_EQUAL( 0U, list.bucketSize(1,0) );
            CHECK_EQUAL( 0U, list.bucketSize(1,1) );
            CHECK_EQUAL( false, list.isBucketDone(1,0) );
            CHECK_EQUAL( false, list.isBucketDone(1,1) );
        }
    }

    TEST_FIXTURE(setup, autoCopy_nonlocalStorageRankZero ) {
        if(shared_memory_size != 2 ) { return; }

        rayList_t list(master, 200, shared_memory_rank, shared_memory_size, true,10);

        bucket_header_t* header;
        header = list.getBucketHeader( 1, 0 );

        if( shared_memory_rank == 0 ) {
            CHECK_EQUAL(0, header->size );
            CHECK_EQUAL(false, header->done );
            CHECK_EQUAL( 10, list.getParticlesPerBucket() );
        }

        MPI_Barrier( shared_memory_communicator );

        if(shared_memory_rank==1 ) {
            for( unsigned i=0; i<20; ++i){
                p.particle.index = i;
                list.addCollision(1,p.particle);
            }
        }
        MPI_Barrier( shared_memory_communicator );

        if(shared_memory_rank==0 ) {
            CHECK_EQUAL( 10U, list.bucketSize(1,0) );
            CHECK_EQUAL( 10U, list.bucketSize(1,1) );
            CHECK_EQUAL( 0U, list.getCollisionFromLocal(1,0,0).index );
            CHECK_EQUAL( 1U, list.getCollisionFromLocal(1,0,1).index );
            CHECK_EQUAL( 8U, list.getCollisionFromLocal(1,0,8).index );
            CHECK_EQUAL( 9U, list.getCollisionFromLocal(1,0,9).index );
            CHECK_EQUAL( 10U, list.getCollisionFromLocal(1,0,10).index );
            CHECK_EQUAL( 11U, list.getCollisionFromLocal(1,0,11).index );
            CHECK_EQUAL( 18U, list.getCollisionFromLocal(1,0,18).index );
            CHECK_EQUAL( 19U, list.getCollisionFromLocal(1,0,19).index );
            CHECK_EQUAL( true, list.isBucketDone(1,0) );
            CHECK_EQUAL( true, list.isBucketDone(1,1) );

            for( unsigned i=20; i<29; ++i){
                p.particle.index = i;
                // add to rank 0
                list.addCollision(0,p.particle);
            }
            CHECK_EQUAL( 9U, list.getMasterSize() );
            for( unsigned i=29; i<40; ++i){
                p.particle.index = i;
                // add to rank 0
                list.addCollision(0,p.particle);
            }

            CHECK_EQUAL( 0U, list.getMasterSize() );
            CHECK_EQUAL( 40U, master.size() );

            CHECK_EQUAL( 20U, master.get(0).index );
            CHECK_EQUAL( 29U, master.get(9).index );
            CHECK_EQUAL( 0U, master.get(10).index );
            CHECK_EQUAL( 9U, master.get(19).index );
            CHECK_EQUAL( 10U, master.get(20).index );
            CHECK_EQUAL( 19U, master.get(29).index );
            CHECK_EQUAL( 30U, master.get(30).index );
            CHECK_EQUAL( 39U, master.get(39).index );

            CHECK_EQUAL( 0U, list.bucketSize(1,0) );
            CHECK_EQUAL( 0U, list.bucketSize(1,1) );
            CHECK_EQUAL( false, list.isBucketDone(1,0) );
            CHECK_EQUAL( false, list.isBucketDone(1,1) );
        }
    }

    TEST_FIXTURE(setup, process_1_marked_done ) {
        if(shared_memory_size != 2 ) { return; }

        rayList_t list(master, 200, shared_memory_rank, shared_memory_size, true,10);

        bucket_header_t* header;
        header = list.getBucketHeader( 1, 0 );

        if(shared_memory_rank==0 ) {
            CHECK_EQUAL(0, header->size );
            CHECK_EQUAL(false, header->done );
            CHECK_EQUAL( 10, list.getParticlesPerBucket() );
        }
        MPI_Barrier( shared_memory_communicator );

        if(shared_memory_rank==1 ) {
            for( unsigned i=0; i<15; ++i){
                p.particle.index = i;
                list.addCollision(1,p.particle);
            }
        }
        MPI_Barrier( shared_memory_communicator );

        if(shared_memory_rank==0 ) {
            CHECK_EQUAL( 1, list.getCurrentBucket(1) );
            CHECK_EQUAL( 10U, list.bucketSize(1,0) );
            CHECK_EQUAL( 5U, list.bucketSize(1,1) );
            CHECK_EQUAL( true, list.isBucketDone(1,0) );
            CHECK_EQUAL( false, list.isBucketDone(1,1) );
        }

        MPI_Barrier( shared_memory_communicator );

        if(shared_memory_rank==1 ) {
            // indicate rank 1 is done
            list.flushRank(1, false);
        }
        MPI_Barrier( shared_memory_communicator );

        if(shared_memory_rank==0 ) {
            CHECK_EQUAL( true, list.isBucketDone(1,1) );
            CHECK_EQUAL( true, list.isRankDone(1));
            CHECK_EQUAL( true, list.allDone());
        }
    }

    void addParticles( rayList_t& list, unsigned rank, unsigned number) {
        ParticleSetup setup;
        for( unsigned i=0; i<number; ++i){
            setup.particle.index = rank*10000+i;
            // add to rank
            list.addCollision(rank,setup.particle);
        }
        list.flushRank(rank, false);
    };

    TEST_FIXTURE(setup, rank0_finishhes_waits_on_rank1 ) {
        if(shared_memory_size != 2 ) { return; }

        rayList_t list(master, 200, shared_memory_rank, shared_memory_size, true,10);

        bucket_header_t* header;
        header = list.getBucketHeader( 1, 0 );

        if(shared_memory_rank==0 ) {
            CHECK_EQUAL(0, header->size );
            CHECK_EQUAL(false, header->done );
            CHECK_EQUAL( 10, list.getParticlesPerBucket() );
        }
        MPI_Barrier( shared_memory_communicator );

        if(shared_memory_rank==0 ) {
            for( unsigned i=0; i<9; ++i){
                p.particle.index = i;
                list.addCollision(0,p.particle);
            }
        }
        if(shared_memory_rank==1 ) {
            addParticles( list, shared_memory_rank, 25);
        }
        MPI_Barrier( shared_memory_communicator );

        if(shared_memory_rank==0 ) {
            CHECK_EQUAL( 9U, list.getMasterSize() );
            list.flushRank(0,true);
            CHECK_EQUAL( 0U, list.getMasterSize() );
            CHECK_EQUAL( 34U, master.size() );

            CHECK_EQUAL( 0U, master.get(0).index );
            CHECK_EQUAL( 8U, master.get(8).index );
            CHECK_EQUAL( 10000U, master.get(9).index );
            CHECK_EQUAL( 10001U, master.get(10).index );
            CHECK_EQUAL( 10024U, master.get(33).index );
        }
    }

    TEST_FIXTURE(setup, addParticles_From_2MPIProcesses ) {
        if(shared_memory_size != 2 ) { return; }

        rayList_t list(master, 200, shared_memory_rank, shared_memory_size, true,10);

        bucket_header_t* header;
        header = list.getBucketHeader( 1, 0 );

        if(shared_memory_rank==0 ) {
            CHECK_EQUAL(0, header->size );
            CHECK_EQUAL(false, header->done );
            CHECK_EQUAL( 10, list.getParticlesPerBucket() );
        }
        MPI_Barrier( shared_memory_communicator );

        if(shared_memory_rank==1 ) {
            addParticles( list, shared_memory_rank, 2500);
        }
        if(shared_memory_rank==0 ) {
            for( unsigned i=0; i<1009; ++i){
                p.particle.index = i;
                list.addCollision(0,p.particle);
            }
        }

        if(shared_memory_rank==0 ) {
            CHECK_EQUAL( 9U, list.getMasterSize() );
            list.flushRank(0,true);
        }
        MPI_Barrier( shared_memory_communicator );
        if(shared_memory_rank==0 ) {
            CHECK_EQUAL( 0U, list.getMasterSize() );
            CHECK_EQUAL( 3509U, master.size() );
            //            list.debugPrint();
        }

    }
    TEST_FIXTURE(setup, FourMPIProcesses_addLotsOfParticles ) {
        if(shared_memory_size != 4 ) { return; }

        rayList_t list(master, 400, shared_memory_rank, shared_memory_size, true,10);

        bucket_header_t* header;
        header = list.getBucketHeader( 1, 0 );
        if(shared_memory_rank==0 ) {
            CHECK_EQUAL(0, header->size );
            CHECK_EQUAL(false, header->done );
            CHECK_EQUAL( 10, list.getParticlesPerBucket() );
        }
        MPI_Barrier( shared_memory_communicator );

        if(shared_memory_rank > 0 ) {
            addParticles( list, shared_memory_rank, 2500);
        }
        if(shared_memory_rank==0 ) {
            for( unsigned i=0; i<1009; ++i){
                p.particle.index = i;
                list.addCollision(0,p.particle);
            }
        }

        if(shared_memory_rank==0 ) {
            CHECK_EQUAL( 9U, list.getMasterSize() );
            list.flushRank(0,true);
        }
        MPI_Barrier( shared_memory_communicator );
        if(shared_memory_rank==0 ) {
            CHECK_EQUAL( 0U, list.getMasterSize() );
            CHECK_EQUAL( 8509U, master.size() );
        }
    }
    TEST_FIXTURE(setup, clear ) {
        if(shared_memory_size != 2 ) { return; }

        rayList_t list(master, 200, shared_memory_rank, shared_memory_size, true,10);

        bucket_header_t* header;
        header = list.getBucketHeader( 1, 0 );

        if(shared_memory_rank==0 ) {
            CHECK_EQUAL(0, header->size );
            CHECK_EQUAL(false, header->done );
            CHECK_EQUAL( 10, list.getParticlesPerBucket() );
        }
        MPI_Barrier( shared_memory_communicator );

        if(shared_memory_rank==0 ) {
            for( unsigned i=0; i<9; ++i){
                p.particle.index = i;
                list.addCollision(0,p.particle);
            }
        }

        if(shared_memory_rank==0 ) {
            CHECK_EQUAL( 9U, list.getMasterSize() );
            list.clear(0);
        }
        MPI_Barrier( shared_memory_communicator );
        if(shared_memory_rank==0 ) {
            CHECK_EQUAL( 0U, list.getMasterSize() );
            CHECK_EQUAL( 0U, master.size() );
        }
    }


}

} // end namespace
