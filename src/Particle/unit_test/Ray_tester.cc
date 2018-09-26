#include <UnitTest++.h>

#include <iostream>

#include "MonteRayDefinitions.hh"
#include "GPUUtilityFunctions.hh"
#include "MonteRayVector3D.hh"

#include "Ray.hh"

namespace Ray_simple_tester {

SUITE( Ray_simple_tests ) {

    using namespace MonteRay;

    TEST( ray_ctor ) {
        Ray_t<> ray;
        // std::cout << "Debug: running Ray_tester.cc -- ctor test\n";
        CHECK(true);
    }

    TEST( getN ) {
        Ray_t<> ray;
        CHECK_EQUAL( 1, ray.getN());
    }

    TEST( ParticleRay_t_getN ) {
        ParticleRay_t ray;
        CHECK_EQUAL( 1, ray.getN());
    }

    TEST( PointDetRay_t_getN ) {
        PointDetRay_t ray;
        CHECK_EQUAL( 3, ray.getN());
    }

    class dummyType {
    public:
        dummyType(){}
        bool isANeutron() const { return false; }
    };

    typedef MonteRay::Vector3D<double> PositionDouble_t;
    class dummyParticle {
    public:
        dummyParticle() {
            type = new dummyType();
        }

        ~dummyParticle() {
            delete type;
        }

        PositionDouble_t getPosition( void ) const {return pos;}
        PositionDouble_t getDirection( void ) const { return dir; }
        double getEnergy(void) const { return energy; }
        double getWeight(void) const { return weight; }
        int getLocationIndex() const { return locationIndex; }

        const dummyType* getType() const { return type; }

    private:
        PositionDouble_t pos = PositionDouble_t( 1.0, 2.0, 3.0 );
        PositionDouble_t dir = PositionDouble_t( 4.0, 5.0, 6.0 );
        double energy = 7.0;
        double weight = 8.0;
        int locationIndex = 9;
        dummyType* type;

    };

    TEST( Ctor1_takes_a_GenericParticle ) {
        dummyParticle particle;
        ParticleRay_t ray(particle);

        CHECK_CLOSE( 1.0, ray.getPosition()[0], 1e-6);
        CHECK_CLOSE( 2.0, ray.getPosition()[1], 1e-6);
        CHECK_CLOSE( 3.0, ray.getPosition()[2], 1e-6);
        CHECK_CLOSE( 4.0, ray.getDirection()[0], 1e-6);
        CHECK_CLOSE( 5.0, ray.getDirection()[1], 1e-6);
        CHECK_CLOSE( 6.0, ray.getDirection()[2], 1e-6);
        CHECK_CLOSE( 7.0, ray.getEnergy(0), 1e-6);
        CHECK_CLOSE( 8.0, ray.getWeight(0), 1e-6);
        CHECK_CLOSE( 9.0, ray.getIndex(), 1e-6);
    }

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
            incoherent.energy = 1.1;
            incoherent.probability = 1.5;

            coherent.energy = 2.2;
            coherent.probability = 4.0;

            pairProduction.energy = 0.511;
            pairProduction.probability = 2.0;

        }
        ~dummuyPhotonScatteringProbabilities(){}

        dummyScatteringProbabilityResult incoherent;
        dummyScatteringProbabilityResult coherent;
        dummyScatteringProbabilityResult pairProduction;
    };

    TEST( Ctor3_takes_a_GenericParticle_Results_and_detectorIndex ) {
        dummyParticle particle;
        dummuyPhotonScatteringProbabilities probs;
        unsigned detectorID = 10;

        PointDetRay_t ray(particle, probs, detectorID);

        CHECK_CLOSE( 1.0, ray.getPosition()[0], 1e-6);
        CHECK_CLOSE( 2.0, ray.getPosition()[1], 1e-6);
        CHECK_CLOSE( 3.0, ray.getPosition()[2], 1e-6);
        CHECK_CLOSE( 4.0, ray.getDirection()[0], 1e-6);
        CHECK_CLOSE( 5.0, ray.getDirection()[1], 1e-6);
        CHECK_CLOSE( 6.0, ray.getDirection()[2], 1e-6);
        CHECK_CLOSE( 1.1, ray.getEnergy(0), 1e-6);
        CHECK_CLOSE( 2.2, ray.getEnergy(1), 1e-6);
        CHECK_CLOSE( 0.511, ray.getEnergy(2), 1e-6);
        CHECK_CLOSE( 1.5*8.0, ray.getWeight(0), 1e-6);
        CHECK_CLOSE( 4.0*8.0, ray.getWeight(1), 1e-6);
        CHECK_CLOSE( 2.0*8.0, ray.getWeight(2), 1e-6);
        CHECK_EQUAL( 9, ray.getIndex() );
        CHECK_EQUAL( 10, ray.getDetectorIndex() );
    }

}

} // end namespace

