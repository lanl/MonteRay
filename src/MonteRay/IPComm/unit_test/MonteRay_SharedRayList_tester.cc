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
                std::cout << " E = " << particle.energy[0];
                std::cout << " W = " << particle.weight[0];
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
                std::cout << " E = " << particle.energy[0];
                std::cout << " W = " << particle.weight[0];
                std::cout << " index = " << particle.index;
                std::cout << "\n";
            }
            std::cout << "******************************************************************************************\n";
        }

    };

    typedef rayList_t::store_func_t store_func_t;
    TEST( ctor_defaults ) {
        TestMasterList master;
        rayList_t list(master, 10);
        CHECK_EQUAL( false, list.isUsingMPI() );
        CHECK_EQUAL( 1000, list.getNBuckets() );
        CHECK_EQUAL( 1, list.getNRanks() );
        CHECK_EQUAL( 0, list.getCurrentBucket(0) );
    }

    TEST( size ) {
        TestMasterList master;
        rayList_t list(master, 100);
        CHECK_EQUAL( 100, list.size() );
        CHECK_EQUAL( 100, list.getParticlesPerRank() );
        CHECK_EQUAL( 1, list.getParticlesPerBucket() );
    }
    TEST( isBucketFull_empty ) {
        TestMasterList master;
        rayList_t list(master, 10);
        CHECK_EQUAL( false, list.isBucketFull(0,0) );
    }

    TEST( bucketSize_empty ) {
        TestMasterList master;
        rayList_t list(master, 10);
        CHECK_EQUAL( 0U, list.bucketSize(0,0) );
    }

    TEST( add_via_parameters ){
        TestMasterList master;
        rayList_t list(master, 10);

        gpuFloatType_t pos[3];
        gpuFloatType_t dir[3];
        gpuFloatType_t energy;
        gpuFloatType_t weight;
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

    TEST_FIXTURE(ParticleSetup, clear ){

        TestMasterList master;

        // simulate 2 ranks
        rayList_t list(master, 200);

        for( unsigned i=0; i<9; ++i){
            particle.index = i;
            // add to rank 0
            list.addCollision(0,particle);
        }

        CHECK_EQUAL( 9U, master.size() );
        list.clearRank(0);
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
         rayList_t list(master, 100,10);
         dummyParticle particle;

         double prob = 20.0;
         list.add(particle,prob);

         CHECK_EQUAL( 1, master.size() );
         ray_t stored_ray = master.get(0);

         CHECK_CLOSE( 8.0*prob, stored_ray.getWeight(0), 1e-6);
     }

}

} // end namespace
