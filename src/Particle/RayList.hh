#ifndef RAYLIST_HH_
#define RAYLIST_HH_

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <iostream>

#include "Ray.hh"

namespace MonteRay{

typedef unsigned RayListSize_t;

template<unsigned N = 1, bool isCudaIntermediate = false >
class RayList_t {
public:
	typedef Ray_t<N> RAY_T;
	CUDAHOST_CALLABLE_MEMBER RayList_t(RayListSize_t num = 1 ){
	    if( num == 0 ) { num = 1; }
	    //std::cout << "RayList_t - ctor(n), n=" << num << " \n";

	    if( ! isCudaIntermediate ) {
	    	points = (RAY_T*) malloc( num*sizeof( RAY_T ) );
	    } else {
#ifdef __CUDACC__
	    	cudaMalloc(&points, num*sizeof( RAY_T ) );
#else
	    	throw std::runtime_error("RayList_t::RayList_t(num) -- can NOT allocate CUDA intermediate without CUDA.");
#endif
	    }
	    nAllocated = num;
	}

	CUDAHOST_CALLABLE_MEMBER RayList_t(const RayList_t<N,false>& rhs) :
		RayList_t( rhs.size() )
	{
		//std::cout << "RayList_t - ctor(rhs) \n";

		nUsed = rhs.nUsed;
		if( ! isCudaIntermediate ) {
			std::memcpy( points, rhs.points, nUsed*sizeof( RAY_T ) );
		} else {
#ifdef __CUDACC__
			CUDA_CHECK_RETURN( cudaMemcpy(points, rhs.points, sizeof( RAY_T ) * capacity(), cudaMemcpyHostToDevice));
#else
			throw std::runtime_error("RayList_t::RayList_t(RayList_t&) -- can NOT initialize CUDA intermediate without CUDA.");
#endif
		}
	}

	CUDAHOST_CALLABLE_MEMBER RayList_t(const RayList_t<N,true>& rhs) :
		RayList_t( rhs.size() )
	{
		//std::cout << "RayList_t - ctor(rhs) \n";

		nUsed = rhs.nUsed;
		if( ! isCudaIntermediate ) {
#ifdef __CUDACC__
			CUDA_CHECK_RETURN( cudaMemcpy(points, rhs.points, sizeof( RAY_T ) * capacity(), cudaMemcpyDeviceToHost));
#else
			throw std::runtime_error("RayList_t::RayList_t(RayList_t&) -- can NOT copy from CUDA intermediate without CUDA.");
#endif
		} else {
			std::memcpy( points, rhs.points, nUsed*sizeof( RAY_T ) );
		}
	}

	CUDAHOST_CALLABLE_MEMBER void operator=(const RayList_t<N,false>& rhs) {
		//std::cout << "RayList_t - ctor(rhs) \n";

		nUsed = rhs.nUsed;
		if( ! isCudaIntermediate ) {
			std::memcpy( points, rhs.points, nUsed*sizeof( RAY_T ) );
		} else {
#ifdef __CUDACC__
			CUDA_CHECK_RETURN( cudaMemcpy(points, rhs.points, sizeof( RAY_T ) * capacity(), cudaMemcpyHostToDevice));
#else
			throw std::runtime_error("RayList_t::RayList_t(RayList_t&) -- can NOT initialize CUDA intermediate without CUDA.");
#endif
		}
	}

	CUDA_CALLABLE_MEMBER ~RayList_t(){
		if( ! isCudaIntermediate ) {
			free(points);
		} else {
#ifdef __CUDACC__
			cudaFree( points );
#else
			throw std::runtime_error("RayList_t::~RayList_t -- Destructor can NOT free CUDA intermediate without CUDA.");
#endif
		}
	}

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
	    	printf("CollisionPoints::add -- index > number of allocated points.  %s %d\n", __FILE__, __LINE__);
	    	ABORT( "CollisionPoints.h -- CollisionPoints::add" );
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

typedef RayList_t<1,false> CollisionPoints;
typedef RayList_t<1,false> ParticleRayList;
typedef RayList_t<3,false> PointDetRayList;

} // end namespace



#endif /* RAYLIST_HH_ */
