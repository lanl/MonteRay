#ifndef RAY_H_
#define RAY_H_

#include "MonteRayTypes.hh"
#include "MonteRayAssert.hh"
#include "MonteRay_binaryIO.hh"
#include "MonteRayConstants.hh"
#include "ThirdParty/Array.hh"
#include "ThirdParty/Math.hh"
#include "MonteRayVector3D.hh"

namespace MonteRay{

using Position_t = Vector3D<gpuRayFloat_t>;
using Direction_t = Position_t;
using DetectorIndex_t = unsigned;

template< unsigned N = 1 >
class Ray_t{
public:
    Position_t  pos = { 0.0 };
    Direction_t dir = { 0.0 };
    Array<gpuRayFloat_t, N> energy = { 0.0 };
    Array<gpuRayFloat_t, N> weight = { 1.0 };
    gpuRayFloat_t time = 0.0;
    unsigned index = 0; // starting position mesh index
    DetectorIndex_t detectorIndex = 0;  // for next-event estimator
    ParticleType_t particleType = 0; // particle type 0 = neutron, 1=photon

    constexpr Ray_t() = default;

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

        particleType = particle.getType()->isANeutron() ? neutron : photon;
    }

    constexpr unsigned static getN(void ) {
        return N;
    }

    constexpr const Position_t& getPosition() const { return pos; }

    constexpr const auto& position() const { return pos; }
    constexpr auto& position() { return pos; }

    constexpr const auto& direction() const { return dir; }
    constexpr auto& direction() { return dir; }

    constexpr const Direction_t& getDirection() const { return dir; }
    constexpr void setDirection(const Direction_t& posIn){ pos = posIn; }

    constexpr gpuRayFloat_t getEnergy(unsigned index = 0) const {
        MONTERAY_ASSERT( index < N);
        return energy[index];
    }
    constexpr void setEnergy(gpuRayFloat_t val, unsigned index = 0) {
        MONTERAY_ASSERT( index < N);
        energy[index] = val;
    }

    constexpr gpuRayFloat_t getWeight(unsigned index = 0) const {
        MONTERAY_ASSERT( index < N);
        return weight[index];
    }
    constexpr void setWeight(gpuRayFloat_t val, unsigned index = 0) {
        MONTERAY_ASSERT( index < N);
        weight[index] = val;
    }

    constexpr gpuRayFloat_t getTime() const {
        return time;
    }
    constexpr void setTime(gpuRayFloat_t val) {
        time = val;
    }

    constexpr unsigned getIndex() const {
        return index;
    }

    constexpr DetectorIndex_t getDetectorIndex() const {
        return detectorIndex;
    }

    constexpr ParticleType_t getParticleType() const {
        return particleType;
    }

    constexpr gpuRayFloat_t speed(unsigned i=0) const {
        return particleType == photon ?
          speed_of_light : 
          neutron_speed_from_energy_const() * Math::sqrt(energy[i]);
    }

    // NOTE: ALWAYS READS AND WRITES IN SINGLE PRECISION DUE TO MONTERAY UNIT TESTS
    template<typename S>
    CUDAHOST_CALLABLE_MEMBER void read(S& inFile) {
        short unsigned version;
        binaryIO::read( inFile, version );

        short unsigned num;
        binaryIO::read( inFile, num );

        Vector3D<gpuFloatType_t> read_pos;
        Vector3D<gpuFloatType_t> read_dir;
        Array<gpuFloatType_t, N> read_energy;
        Array<gpuFloatType_t, N> read_weight;

        binaryIO::read( inFile, read_pos );
        binaryIO::read( inFile, read_dir );
        binaryIO::read( inFile, read_energy );
        binaryIO::read( inFile, read_weight );

        for (int i = 0; i < 3; i++){
          pos[i] = static_cast<gpuRayFloat_t>(read_pos[i]);
          dir[i] = static_cast<gpuRayFloat_t>(read_dir[i]);
        }

        for( int i = 0; i < N; i++){
          energy[i] = read_energy[i];
          weight[i] = read_weight[i];
        }

        if( version > 0 ) {
            gpuFloatType_t read_time;
            binaryIO::read( inFile, read_time );
            time = static_cast<gpuRayFloat_t>(read_time);
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

        Vector3D<gpuFloatType_t> write_pos;
        Vector3D<gpuFloatType_t> write_dir;
        for (int i = 0; i < 3; i++){
          write_pos[i] = pos[i];
          write_dir[i] = dir[i];
        }

        Array<gpuFloatType_t, N> write_energy;
        Array<gpuFloatType_t, N> write_weight;
        for( int i = 0; i < N; i++){
          write_energy[i] = energy[i];
          write_weight[i] = weight[i];
        }

        gpuFloatType_t write_time = time;

        binaryIO::write( outFile, write_pos );
        binaryIO::write( outFile, write_dir );
        binaryIO::write( outFile, write_energy );
        binaryIO::write( outFile, write_weight );
        binaryIO::write( outFile, write_time );
        binaryIO::write( outFile, index );
        binaryIO::write( outFile, detectorIndex );
        binaryIO::write( outFile, particleType );
    }

    template <typename Precision>
    static unsigned filesize(unsigned version) {
      unsigned total = 0;
      total += 2*sizeof(short unsigned);
      Vector3D<Precision> read_pos;
      Vector3D<Precision> read_dir;
      Array<Precision, N> read_energy;
      Array<Precision, N> read_weight;
      total += sizeof(read_pos);
      total += sizeof(read_dir);
      total += sizeof(read_energy);
      total += sizeof(read_weight);

      if( version > 0 ) {
          total += sizeof(std::declval<Precision>());
      }

      total += sizeof(index);
      total += sizeof(detectorIndex);
      total += sizeof(particleType);
      return total;
    }

    static unsigned filesize(unsigned version) {
      return filesize<gpuFloatType_t>(version);
    }
};

using PointDetRay_t = Ray_t<3>; 
using ParticleRay_t = Ray_t<1>; 

} // end namespace;

#endif /* RAY_H_ */
