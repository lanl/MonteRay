#include "RayList.hh"

namespace MonteRay {

template<unsigned N>
CUDAHOST_CALLABLE_MEMBER
RayList_t<N>::RayList_t(RayListSize_t num) {
	if( num == 0 ) { num = 1; }
	if( Base::debug ) {
		std::cout << "RayList_t::RayList_t(n), n=" << num << " \n";
	}
	init();
	points = (RAY_T*) MonteRayHostAlloc( num*sizeof( RAY_T ), Base::isManagedMemory );
	nAllocated = num;
}

template<unsigned N>
CUDAHOST_CALLABLE_MEMBER
RayList_t<N>::~RayList_t(){
	if( ! Base::isCudaIntermediate ) {
		MonteRayHostFree(points, Base::isManagedMemory );
	} else {
#ifdef __CUDACC__
		MonteRayDeviceFree( points );
#else
		throw std::runtime_error("RayList_t::~RayList_t -- Destructor can NOT free CUDA intermediate without CUDA.");
#endif
	}
}

/// Copy constructor
template<unsigned N>
CUDAHOST_CALLABLE_MEMBER
RayList_t<N>::RayList_t(const RayList_t<N>& rhs) :
	RayList_t<N>::RayList_t( rhs.nAllocated )
{
	nUsed = rhs.nUsed;
	std::memcpy( points, rhs.points, rhs.nUsed * sizeof( RAY_T) );
}

template<unsigned N>
CUDAHOST_CALLABLE_MEMBER void
RayList_t<N>::copy(const RayList_t<N>* rhs) {
	if( Base::debug ) {
		std::cout << "Debug: RayList_t::operator= (const RayList_t<N>& rhs) \n";
	}

	if( Base::isCudaIntermediate && rhs->isCudaIntermediate ) {
		throw std::runtime_error("RayList_t::operator= -- can NOT copy CUDA intermediate to CUDA intermediate.");
	}

	if( !Base::isCudaIntermediate && !rhs->isCudaIntermediate ) {
		throw std::runtime_error("RayList_t::operator= -- can NOT copy CUDA non-intermediate to CUDA non-intermediate.");
	}

	if( nAllocated > 0 && nAllocated != rhs->nAllocated) {
		throw std::runtime_error("RayList_t::operator= -- can NOT change the size of the RayList.");
	}

	if( Base::isCudaIntermediate ) {
		// target is the intermediate, origin is the host
		if( points == NULL ) {
			points = (RAY_T*) MonteRayDeviceAlloc( rhs->nAllocated*sizeof(RAY_T) );
		}
		MonteRayMemcpy(points, rhs->points, rhs->nAllocated*sizeof(RAY_T), cudaMemcpyHostToDevice);
	} else {
		// target is the host, origin is the intermediate
		MonteRayMemcpy(rhs->points, points, rhs->nAllocated*sizeof(RAY_T), cudaMemcpyDeviceToHost);
	}

	nAllocated = rhs->nAllocated;
	nUsed = rhs->nUsed;
}

} // end namespace

template class MonteRay::RayList_t<1>;
template class MonteRay::RayList_t<3>;
