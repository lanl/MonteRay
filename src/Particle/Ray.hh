#ifndef RAY_H_
#define RAY_H_

#include "MonteRayTypes.hh"
#include "MonteRayAssert.hh"
#include "MonteRay_binaryIO.hh"
#include "MonteRayConstants.hh"
#include "Array.hh"

#ifndef __CUDACC_
#include <cmath>
#endif

namespace MonteRay{

using CollisionPosition_t = Array<gpuFloatType_t, 3>&;
using CollisionDirection_t = Array<gpuFloatType_t, 3>&;
using DetectorIndex_t = unsigned;

template< unsigned N = 1 >
class Ray_t{
public:
    CUDA_CALLABLE_MEMBER Ray_t(){}

    template< typename PARTICLE_T,
              unsigned N_ = N,
              typename std::enable_if<(N_ == 1)>::type* = nullptr >
    Ray_t( const PARTICLE_T& particle, double probability = 1.0) {

        pos[0] = particle.getPosition()[0];
        pos[1] = particle.getPosition()[1];
        pos[2] = particle.getPosition()[2];

        dir[0] = particle.getDirection()[0];
        dir[1] = particle.getDirection()[1];
        dir[2] = particle.getDirection()[2];

        energy[0] = particle.getEnergy();
        weight[0] = particle.getWeight()*probability;

        time = particle.getSimulationTime();

        index = particle.getLocationIndex();
    }

    template< typename PARTICLE_T, typename SCATTERING_PROBABILITES,
              unsigned N_ = N,
              typename std::enable_if<(N_ == 3)>::type* = nullptr >
    Ray_t( const PARTICLE_T& particle,
           const SCATTERING_PROBABILITES& results,
           unsigned argDetectorIndex) {

        pos[0] = particle.getPosition()[0];
        pos[1] = particle.getPosition()[1];
        pos[2] = particle.getPosition()[2];

        dir[0] = particle.getDirection()[0];
        dir[1] = particle.getDirection()[1];
        dir[2] = particle.getDirection()[2];

        energy[0] = results.incoherent.energy;
        energy[1] = results.coherent.energy;
        energy[2] = results.pairProduction.energy;

        weight[0] = particle.getWeight()*results.incoherent.probability;
        weight[1] = particle.getWeight()*results.coherent.probability;
        weight[2] = particle.getWeight()*results.pairProduction.probability;

        time = particle.getSimulationTime();

        index = particle.getLocationIndex();
        detectorIndex = argDetectorIndex;

        if( particle.getType()->isANeutron() ) {
            particleType = neutron;
        } else {
            particleType = photon;
        }
    }

    Array<gpuFloatType_t, 3> pos = { 0.0 };
    Array<gpuFloatType_t, 3> dir = { 0.0 };
    Array<gpuFloatType_t, N> energy = { 0.0 };
    Array<gpuFloatType_t, N> weight = { 0.0 };
    gpuFloatType_t time = { 0.0 };
    unsigned index = 0; // starting position mesh index
    DetectorIndex_t detectorIndex = 0;  // for next-event estimator
    ParticleType_t particleType = 0; // particle type 0 = neutron, 1=photon

    CUDA_CALLABLE_MEMBER constexpr static unsigned getN(void ) {
        return N;
    }

    CUDA_CALLABLE_MEMBER CollisionPosition_t getPosition() {
        return pos;
    }

    CUDA_CALLABLE_MEMBER CollisionDirection_t getDirection() {
        return dir;
    }

    CUDA_CALLABLE_MEMBER gpuFloatType_t getEnergy(unsigned index = 0) const {
        MONTERAY_ASSERT( index < N);
        return energy[index];
    }

    CUDA_CALLABLE_MEMBER gpuFloatType_t getWeight(unsigned index = 0) const {
        MONTERAY_ASSERT( index < N);
        return weight[index];
    }

    CUDA_CALLABLE_MEMBER gpuFloatType_t getTime() {
        return time;
    }

    CUDA_CALLABLE_MEMBER unsigned getIndex() const {
        return index;
    }

    CUDA_CALLABLE_MEMBER DetectorIndex_t getDetectorIndex() const {
        return detectorIndex;
    }

    CUDA_CALLABLE_MEMBER ParticleType_t getParticleType() const {
        return particleType;
    }

    CUDA_CALLABLE_MEMBER gpuFloatType_t speed(unsigned i=0) const {
        if( particleType == photon ) {
            return speed_of_light;
        } else {
            // neutron
#ifdef __CUDACC__
            return (neutron_speed_from_energy_const() * sqrtf( energy[i] ));
#else
            return neutron_speed_from_energy_const() * std::sqrt( energy[i] );
#endif
        }
    }

    template<typename S>
    CUDAHOST_CALLABLE_MEMBER void read(S& inFile) {
        short unsigned version;
        binaryIO::read( inFile, version );

        short unsigned num;
        binaryIO::read( inFile, num );

        binaryIO::read( inFile, pos );
        binaryIO::read( inFile, dir );
        binaryIO::read( inFile, energy );
        binaryIO::read( inFile, weight );

        if( version > 0 ) {
            binaryIO::read( inFile, time );
        }

        binaryIO::read( inFile, index );
        binaryIO::read( inFile, detectorIndex );
        binaryIO::read( inFile, particleType );
    }

    template<typename S>
    CUDAHOST_CALLABLE_MEMBER void write(S& outFile) const {

        const short unsigned version = 1;
        binaryIO::write( outFile, version );

        const short unsigned num = N;
        binaryIO::write( outFile, num );

        binaryIO::write( outFile, pos );
        binaryIO::write( outFile, dir );
        binaryIO::write( outFile, energy );
        binaryIO::write( outFile, weight );
        binaryIO::write( outFile, time );
        binaryIO::write( outFile, index );
        binaryIO::write( outFile, detectorIndex );
        binaryIO::write( outFile, particleType );
    }

    static unsigned filesize(unsigned version) {
        unsigned total = 0;
        total += 2*sizeof(short unsigned);
        total += sizeof(pos);
        total += sizeof(dir);
        total += sizeof(energy);
        total += sizeof(weight);

        if( version > 0 ) {
            total += sizeof(time);
        }

        total += sizeof(index);
        total += sizeof(detectorIndex);
        total += sizeof(particleType);
        return total;
    }
};

typedef Ray_t<3> PointDetRay_t;
typedef Ray_t<1> ParticleRay_t;

} // end namespace;

#endif /* RAY_H_ */
