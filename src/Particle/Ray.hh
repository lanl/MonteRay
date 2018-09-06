#ifndef RAY_H_
#define RAY_H_

#include "GPUErrorCheck.hh"
#include "MonteRay_binaryIO.hh"
#include "MonteRayConstants.hh"

namespace MonteRay{

typedef gpuFloatType_t* CollisionPosition_t;
typedef gpuFloatType_t* CollisionDirection_t;
typedef unsigned DetectorIndex_t;

template< unsigned N = 1 >
class Ray_t{
public:
	typedef MonteRay::ParticleType_t ParticleType_t;
	CUDA_CALLABLE_MEMBER Ray_t(){}

	gpuFloatType_t pos[3] = { 0.0 };
	gpuFloatType_t dir[3] = { 0.0 };
    gpuFloatType_t energy[N] = { 0.0 };
    gpuFloatType_t weight[N] = { 0.0 };
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
		return energy[index];
	}

	CUDA_CALLABLE_MEMBER gpuFloatType_t getWeight(unsigned index = 0) const {
		return weight[index];
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
		binaryIO::read( inFile, index );
		binaryIO::read( inFile, detectorIndex );
		binaryIO::read( inFile, particleType );
	}

	template<typename S>
	CUDAHOST_CALLABLE_MEMBER void write(S& outFile) const {

		const short unsigned version = 0;
		binaryIO::write( outFile, version );

		const short unsigned num = N;
		binaryIO::write( outFile, num );

		binaryIO::write( outFile, pos );
		binaryIO::write( outFile, dir );
		binaryIO::write( outFile, energy );
		binaryIO::write( outFile, weight );
		binaryIO::write( outFile, index );
		binaryIO::write( outFile, detectorIndex );
		binaryIO::write( outFile, particleType );
	}

	static unsigned filesize(void) {
		unsigned total = 0;
		total += 2*sizeof(short unsigned);
		total += sizeof(pos);
		total += sizeof(dir);
		total += sizeof(energy);
		total += sizeof(weight);
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
