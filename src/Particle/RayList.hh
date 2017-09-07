#ifndef RAYLIST_HH_
#define RAYLIST_HH_

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <iostream>

#include "Ray.hh"
#include "MonteRayCopyMemory.hh"

namespace MonteRay{

typedef unsigned RayListSize_t;

template<unsigned N = 1>
class RayList_t : public CopyMemoryBase<RayList_t<N>>{
public:
	using Base = CopyMemoryBase<RayList_t<N>>;

	/// Primary RayList_t constructor.
	/// Takes the size of the list as an argument.
	typedef Ray_t<N> RAY_T;
	CUDAHOST_CALLABLE_MEMBER RayList_t(RayListSize_t num = 1);

	/// Copy constructor
	CUDAHOST_CALLABLE_MEMBER RayList_t(const RayList_t<N>& rhs);

	CUDAHOST_CALLABLE_MEMBER ~RayList_t();

	CUDAHOST_CALLABLE_MEMBER void init() {
		nAllocated = 0;
		nUsed = 0;
		points = NULL;
	}

	CUDAHOST_CALLABLE_MEMBER void copy(const RayList_t<N>* rhs);

	CUDA_CALLABLE_MEMBER RayListSize_t size(void) const {
		return nUsed;
	}

	CUDA_CALLABLE_MEMBER RayListSize_t capacity(void) const {
		return nAllocated;
	}

	CUDA_CALLABLE_MEMBER void clear(void) {
		nUsed = 0;
	}

	CUDA_CALLABLE_MEMBER CollisionPosition_t getPosition(RayListSize_t i) const {
		return points[i].pos;
	}

	CUDA_CALLABLE_MEMBER CollisionPosition_t getDirection(RayListSize_t i) const {
		return points[i].dir;
	}

	CUDA_CALLABLE_MEMBER gpuFloatType_t getEnergy(RayListSize_t i, unsigned index = 0) const {
		return points[i].energy[index];
	}

	CUDA_CALLABLE_MEMBER gpuFloatType_t getWeight(RayListSize_t i, unsigned index = 0) const {
		return points[i].weight[index];
	}

	CUDA_CALLABLE_MEMBER unsigned getIndex(RayListSize_t i) const {
		return points[i].index;
	}

	CUDA_CALLABLE_MEMBER DetectorIndex_t getDetectorIndex(RayListSize_t i) const {
		return points[i].detectorIndex;
	}

	CUDA_CALLABLE_MEMBER ParticleType_t getParticleType(RayListSize_t i) const {
		return points[i].particleType;
	}

	CUDA_CALLABLE_MEMBER RAY_T pop(void) {
#if !defined( RELEASE )
		if( nUsed == 0 ) {
			printf("RayList::pop -- no points.  %s %d\n", __FILE__, __LINE__);
			ABORT( "RayList.hh -- RayList::pop" );
		}
#endif

		nUsed -= 1;
		return points[nUsed];
	}

	CUDA_CALLABLE_MEMBER RAY_T getParticle(RayListSize_t i) {
#if !defined( RELEASE )
		if( i >= nUsed ) {
			printf("RayList::getParticle -- index exceeds size.  %s %d\n", __FILE__, __LINE__);
			ABORT( "RayList.hh -- RayList::getParticle" );
		}
#endif
		return points[i];
	}

	CUDA_CALLABLE_MEMBER void add(const RAY_T& point ) {
#if !defined( RELEASE )
	    if( size() >= capacity() ) {
	    	printf("RayList::add -- index > number of allocated points.  %s %d\n", __FILE__, __LINE__);
	    	ABORT( "RayList.hh -- RayList::add" );
	    }
#endif
		points[size()] = point;
		++nUsed;
	}

	CUDA_CALLABLE_MEMBER static unsigned getN(void ) {
		return N;
	}

	RayListSize_t nAllocated = 0 ;
	RayListSize_t nUsed = 0;
    RAY_T* points = NULL;

};

typedef RayList_t<1> CollisionPoints;
typedef RayList_t<1> ParticleRayList;
typedef RayList_t<3> PointDetRayList;

} // end namespace



#endif /* RAYLIST_HH_ */
