#include <UnitTest++.h>

#include <vector>
#include <thread>
#include <iostream>

#include "MonteRay_SharedRayList.hh"
#include "MonteRayVector3D.hh"
#include "Ray.hh"

namespace SharedNextEventRayList_tester{

SUITE( shared_next_event_ray_list_tester ){
    using namespace MonteRay;
    typedef MonteRay::Ray_t<3> ray3_t;
    typedef SharedRayList<ray3_t>  rayList_t;

    class TestMasterList{
    public:
        TestMasterList(){
        }
        ~TestMasterList(){
        }

        static bool sortFunction(const ray3_t& point1, const ray3_t& point2 ) {
            if( point1.index < point2.index ) {
                return true;
            }
            return false;
        }

        typedef float gpuFloatType_t;
        void add(const void* collision, unsigned N) {
            const ray3_t* ptrCollision = (const ray3_t*) collision;
            for( auto i = 0; i< N; ++i) {
                masterList.push_back( *(ptrCollision+i) );
            }
        }

        ray3_t get(unsigned i) {
            return masterList.at(i);
        }

        unsigned size() const { return masterList.size(); }
        std::vector<ray3_t> masterList;

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
                ray3_t particle = get(i);
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
                ray3_t particle = get(i);
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


    typedef MonteRay::Vector3D<double> PositionDouble_t;

    class dummyType {
    public:
        dummyType(){}
        bool isANeutron() const { return false; }
    };

    class dummyNextEventRay {
    public:
        dummyNextEventRay(){
            type = new dummyType();
        }
        ~dummyNextEventRay(){
            delete type;
        }
        PositionDouble_t getPosition( void ) const {return pos;}
        PositionDouble_t getDirection( void ) const { return dir; }
        double getEnergy() const { return energy; }
        double getWeight() const { return weight; }
        int getLocationIndex() const { return locationIndex; }
        //constexpr static unsigned getNPairs() { return 3;}

        const dummyType* getType() const { return type; }

    private:
        PositionDouble_t pos = PositionDouble_t( 1.0, 2.0, 3.0 );
        PositionDouble_t dir = PositionDouble_t( 4.0, 5.0, 6.0 );
        double energy = 7.0;
        double weight = 10.0;
        int locationIndex = 13;
        dummyType* type;
    };

    class dummyScatteringProbabilityResult{
    public:
        dummyScatteringProbabilityResult() : probability( 0.0 ), energy( 0.0 ) {}
        ~dummyScatteringProbabilityResult(){}
        double probability;
        double energy;
    };

    class dummuyPhotonScatteringProbabilities {
    public:
        dummuyPhotonScatteringProbabilities(){
            incoherent.energy = 0.0;
            incoherent.probability = 1.0;

            coherent.energy = 0.0;
            coherent.probability = 4.0;

            pairProduction.energy = 0.511;
            pairProduction.probability = 2.0;

        }
        ~dummuyPhotonScatteringProbabilities(){}

        dummyScatteringProbabilityResult incoherent;
        dummyScatteringProbabilityResult coherent;
        dummyScatteringProbabilityResult pairProduction;
    };

    TEST( addDummyNextEventRay ){
         TestMasterList master;
         rayList_t list(master, 100,0,1,false,10);
         dummyNextEventRay particle;
         dummuyPhotonScatteringProbabilities probs;
         unsigned detectorID = 10;

         double prob = 20.0;
         list.add(particle, probs, detectorID);
         list.flush(0,true);

         CHECK_EQUAL( 1, master.size() );
         ray3_t stored_ray = master.get(0);

         CHECK_CLOSE( 10.0, stored_ray.getWeight(0), 1e-6);
         CHECK_CLOSE( 40.0, stored_ray.getWeight(1), 1e-6);
         CHECK_CLOSE( 20.0, stored_ray.getWeight(2), 1e-6);
     }

}

} // end namespace
