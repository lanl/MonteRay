#include <UnitTest++.h>

#include <vector>
#include <thread>
#include <iostream>

#include "MonteRay_SharedRayList.hh"
#include "MonteRayVector3D.hh"
#include "Ray.hh"

namespace SharedCollisionPointList_tester{

SUITE( shared_collisionPointList_tester ){
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
    };

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

    typedef rayList_t::store_func_t store_func_t;
    TEST( ctor_defaults ) {
        TestMasterList master;
        rayList_t list(master, 10,0,1,false);
        CHECK_EQUAL( false, list.isUsingMPI() );
        CHECK_EQUAL( 1000, list.getNBuckets() );
        CHECK_EQUAL( 1, list.getNRanks() );
        CHECK_EQUAL( 0, list.getCurrentBucket(0) );
    }

    TEST( getBucketHeader_initialized ) {
        TestMasterList master;
        rayList_t list(master, 10,0,2,false);
        bucket_header_t* header = list.getBucketHeader( 1, 0 );
        CHECK_EQUAL(0, header->size );
        CHECK_EQUAL(false, header->done );
    }

    TEST_FIXTURE(ParticleSetup, getBucketHeader_with_a_particle ) {
        TestMasterList master;
        rayList_t list(master, 10,0,2,false);
        bucket_header_t* header = list.getBucketHeader( 1, 0 );
        CHECK_EQUAL(0, header->size );
        CHECK_EQUAL(false, header->done );
        list.addCollision(1,particle);
        CHECK_EQUAL(1, header->size );
        CHECK_EQUAL(true, header->done );
    }

    TEST( size ) {
        TestMasterList master;
        rayList_t list(master, 100,0,1,false,10);
        CHECK_EQUAL( 100, list.size() );
        CHECK_EQUAL( 100, list.getParticlesPerRank() );
        CHECK_EQUAL( 10, list.getParticlesPerBucket() );
    }
    TEST( isBucketFull_empty ) {
        TestMasterList master;
        rayList_t list(master, 10,0,1,false);
        CHECK_EQUAL( false, list.isBucketFull(0,0) );
    }

    TEST( bucketSize_empty ) {
        TestMasterList master;
        rayList_t list(master, 10,0,1,false);
        CHECK_EQUAL( 0U, list.bucketSize(0,0) );
    }

    TEST_FIXTURE(ParticleSetup, addCollisionToRank1_not_full ){
        TestMasterList master;
        rayList_t list(master, 100,0,2,false,10);

        list.addCollision(1,particle);
        CHECK_EQUAL( 1U, list.bucketSize(1,0) );
        CHECK_EQUAL( false, list.isBucketFull(1,0) );
        CHECK_EQUAL( false, list.isBucketDone(1,0) );
        CHECK_EQUAL( 0, list.getCurrentBucket(1) );
    }

    TEST_FIXTURE(ParticleSetup, addCollisionToRank1_full ){
        TestMasterList master;
        rayList_t list(master, 10,0,2,false);
        CHECK_EQUAL( 0U, list.bucketSize(1,0) );
        CHECK_EQUAL( false, list.isBucketFull(1,0) );
        CHECK_EQUAL( false, list.isBucketDone(1,0) );
        CHECK_EQUAL( 0U, list.getCurrentBucket(1) );

        list.addCollision(1,particle);
        CHECK_EQUAL( 1U, list.bucketSize(1,0) );
        CHECK_EQUAL( true, list.isBucketFull(1,0) );
        CHECK_EQUAL( true, list.isBucketDone(1,0) );
        CHECK_EQUAL( 1, list.getCurrentBucket(1) );
    }


    TEST_FIXTURE(ParticleSetup, add_a_Particle_goes_to_master_from_Rank0 ){
        TestMasterList master;
        rayList_t list(master, 10,0,1,false);

        particle.index = 99;
        list.addCollision(0,particle);
        CHECK_EQUAL( 0, list.bucketSize(0,0) );
        CHECK_EQUAL( false, list.isBucketFull(0,0) );
        CHECK_EQUAL( false, list.isBucketDone(0,0) );
        CHECK_EQUAL( 0, list.getCurrentBucket(0) );

        CHECK_EQUAL( 1U, master.size() );
        CHECK_EQUAL( 99U, master.get(0).index );
    }

    TEST( add_via_parameters ){
        TestMasterList master;
        rayList_t list(master, 10,0,1,false);

        float pos[3];
        float dir[3];
        float energy;
        float weight;
        unsigned index;

        pos[0] = 0.0;
        pos[1] = 0.0;
        pos[2] = 1.0;
        dir[0] = 0.0;
        dir[1] = 0.0;
        dir[2] = 1.0;
        energy = 1.0;
        weight = 1.0;
        index = 98;

        list.addCollision2(0, pos, dir, energy, weight, index);
        CHECK_EQUAL( 0, list.bucketSize(0,0) );
        CHECK_EQUAL( false, list.isBucketFull(0,0) );
        CHECK_EQUAL( false, list.isBucketDone(0,0) );
        CHECK_EQUAL( 0, list.getCurrentBucket(0) );

        CHECK_EQUAL( 1U, master.size() );
        CHECK_EQUAL( 98U, master.get(0).index );
    }



    TEST_FIXTURE(ParticleSetup, fill_a_bucket ){
        TestMasterList master;
        rayList_t list(master, 200,0,2,false,10);
        CHECK_EQUAL( 0, list.getCurrentBucket(1) );
        for( unsigned i=0; i<15; ++i){
            list.addCollision(1,particle);
        }
        CHECK_EQUAL( 10U, list.bucketSize(1,0) );
        CHECK_EQUAL( true, list.isBucketFull(1,0) );
        CHECK_EQUAL( true, list.isBucketDone(1,0) );
        CHECK_EQUAL( 5U, list.bucketSize(1,1) );
        CHECK_EQUAL( false, list.isBucketFull(1,1) );
        CHECK_EQUAL( false, list.isBucketDone(1,1) );
        CHECK_EQUAL( 1, list.getCurrentBucket(1) );
    }

    TEST_FIXTURE(ParticleSetup, copyToMaster ){
        TestMasterList master;
        rayList_t list(master, 200,0,2,false,10);
        CHECK_EQUAL( 0, list.getCurrentBucket(1) );
        for( unsigned i=0; i<20; ++i){
            particle.index = i;
            list.addCollision(1,particle);
        }
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

    TEST_FIXTURE(ParticleSetup, autoCopy_nonlocalStorageRankZero ){
        TestMasterList master;

        // simulate 2 ranks
        rayList_t list(master, 200,0,2,false,10);
        CHECK_EQUAL( 0, list.getCurrentBucket(1) );
        for( unsigned i=0; i<20; ++i){
            particle.index = i;
            // add to rank 1
            list.addCollision(1,particle);
        }
        CHECK_EQUAL( 2, list.getCurrentBucket(1) );
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
            particle.index = i;
            // add to rank 0
            list.addCollision(0,particle);
        }
        CHECK_EQUAL( 9U, list.getMasterSize() );
        for( unsigned i=29; i<40; ++i){
            particle.index = i;
            // add to rank 0
            list.addCollision(0,particle);
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

    TEST_FIXTURE(ParticleSetup, process_1_marked_done ){
        TestMasterList master;

        // simulate 2 ranks
        rayList_t list(master, 200,0,2,false,10);
        CHECK_EQUAL( 0, list.getCurrentBucket(1) );
        for( unsigned i=0; i<15; ++i){
            particle.index = i;
            // add to rank 1
            list.addCollision(1,particle);
        }

        CHECK_EQUAL( 1, list.getCurrentBucket(1) );
        CHECK_EQUAL( 10U, list.bucketSize(1,0) );
        CHECK_EQUAL( 5U, list.bucketSize(1,1) );
        CHECK_EQUAL( true, list.isBucketDone(1,0) );
        CHECK_EQUAL( false, list.isBucketDone(1,1) );

        // indicate rank 1 is done
        list.flush(1);
        CHECK_EQUAL( true, list.isBucketDone(1,1) );
        CHECK_EQUAL( true, list.isRankDone(1));
        CHECK_EQUAL( true, list.allDone());
    }

    void addParticles( rayList_t& list, unsigned rank, unsigned number) {
        ParticleSetup setup;
        for( unsigned i=0; i<number; ++i){
            setup.particle.index = rank*100+i;
            // add to rank
            list.addCollision(rank,setup.particle);
        }
        list.flush(rank);
    };

    TEST_FIXTURE(ParticleSetup, rank0_finishhes_waits_on_rank1 ){

        TestMasterList master;

        // simulate 2 ranks
        rayList_t list(master, 200,0,2,false,10);
        std::thread thread1(addParticles, std::ref(list), 1U, 25U);

        for( unsigned i=0; i<9; ++i){
            particle.index = i;
            // add to rank 0
            list.addCollision(0,particle);
        }
        CHECK_EQUAL( 9U, list.getMasterSize() );
        thread1.join();

        list.flush(0, true);
        CHECK_EQUAL( 0U, list.getMasterSize() );
        CHECK_EQUAL( 34U, master.size() );

        CHECK_EQUAL( 0U, master.get(0).index );
        CHECK_EQUAL( 8U, master.get(8).index );
        CHECK_EQUAL( 100U, master.get(9).index );
        CHECK_EQUAL( 101U, master.get(10).index );
        CHECK_EQUAL( 124U, master.get(33).index );

    }

    TEST_FIXTURE(ParticleSetup, addLotsOfParticlesFromThread ){

        TestMasterList master;

        // simulate 2 ranks
        rayList_t list(master, 200,0,2,false,10);
        std::thread thread1(addParticles, std::ref(list), 1U, 2500U);

        for( unsigned i=0; i<1009; ++i){
            particle.index = i;
            // add to rank 0
            list.addCollision(0,particle);
        }
        CHECK_EQUAL( 9U, list.getMasterSize() );
        list.flush(0, true);

        thread1.join();
        CHECK_EQUAL( 0U, list.getMasterSize() );
        CHECK_EQUAL( 3509U, master.size() );
    }

    TEST_FIXTURE(ParticleSetup, FourThreads_addLotsOfParticles ){

        TestMasterList master;

        // simulate 2 ranks
        rayList_t list(master, 500,0,5,false,10);
        std::thread thread1(addParticles, std::ref(list), 1U, 2500U);
        std::thread thread2(addParticles, std::ref(list), 2U, 2500U);
        std::thread thread3(addParticles, std::ref(list), 3U, 2500U);
        std::thread thread4(addParticles, std::ref(list), 4U, 2500U);

        for( unsigned i=0; i<1009; ++i){
            particle.index = i;
            // add to rank 0
            list.addCollision(0,particle);
        }
        CHECK_EQUAL( 9U, list.getMasterSize() );
        list.flush(0,true);
        CHECK_EQUAL( 0U, list.getMasterSize() );
        CHECK_EQUAL( 11009U, master.size() );

        thread1.join();
        thread2.join();
        thread3.join();
        thread4.join();

    }

    TEST_FIXTURE(ParticleSetup, finalize_passed_to_controller ){

        TestMasterList master;

        // simulate 2 ranks
        rayList_t list(master, 200,0,2,false,10);
        std::thread thread1(addParticles, std::ref(list), 1U, 2500U);

        for( unsigned i=0; i<1009; ++i){
            particle.index = i;
            // add to rank 0
            list.addCollision(0,particle);
        }
        CHECK_EQUAL( 9U, list.getMasterSize() );
        list.flush(0,true);
        CHECK_EQUAL( 0U, list.getMasterSize() );
        CHECK_EQUAL( 3509U, master.size() );

        thread1.join();
    }

    TEST_FIXTURE(ParticleSetup, clear ){

        TestMasterList master;

        // simulate 2 ranks
        rayList_t list(master, 200,0,2,false);

        for( unsigned i=0; i<9; ++i){
            particle.index = i;
            // add to rank 0
            list.addCollision(0,particle);
        }

        CHECK_EQUAL( 9U, master.size() );
        list.clear(0);
        CHECK_EQUAL( 0U, list.getMasterSize() );
        CHECK_EQUAL( 0U, master.size() );
    }

    typedef MonteRay::Vector3D<double> PositionDouble_t;
    class dummyParticle {
    public:
        PositionDouble_t getPosition( void ) const {return pos;}
        PositionDouble_t getDirection( void ) const { return dir; }
        double getEnergy(void) const { return energy; }
        double getWeight(void) const { return weight; }
        double getSimulationTime(void) const { return time; }
        int getLocationIndex() const { return locationIndex; }

    private:
        PositionDouble_t pos = PositionDouble_t( 1.0, 2.0, 3.0 );
        PositionDouble_t dir = PositionDouble_t( 4.0, 5.0, 6.0 );
        double energy = 7.0;
        double weight = 8.0;
        double time = 10.0;
        int locationIndex = 9;

    };

    TEST( addDummyParticle ){
         TestMasterList master;
         rayList_t list(master, 100,0,2,false,10);
         dummyParticle particle;

         double prob = 20.0;
         list.add(particle,prob);

         CHECK_EQUAL( 1, master.size() );
         ray_t stored_ray = master.get(0);

         CHECK_CLOSE( 8.0*prob, stored_ray.getWeight(0), 1e-6);
     }

}

} // end namespace
