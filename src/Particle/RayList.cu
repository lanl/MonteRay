#include "RayList.hh"
#include "MonteRayCopyMemory.t.hh"

namespace MonteRay {

template<unsigned N>
CUDAHOST_CALLABLE_MEMBER
RayList_t<N>::RayList_t(RayListSize_t num) {
    if( num == 0 ) { num = 1; }
    if( Base::debug ) {
        std::cout << "RayList_t::RayList_t(n), n=" << num << " \n";
    }
    reallocate( num );
}

template<unsigned N>
CUDAHOST_CALLABLE_MEMBER
void
RayList_t<N>::reallocate(size_t n) {
    if( points != NULL ) { MonteRayHostFree( points, Base::isManagedMemory ); }

    init();
    points = (RAY_T*) MONTERAYHOSTALLOC( n*sizeof( RAY_T ), Base::isManagedMemory, "host RayList_t::points" );
    nAllocated = n;
}

template<unsigned N>
CUDAHOST_CALLABLE_MEMBER
RayList_t<N>::~RayList_t(){
    if( ! Base::isCudaIntermediate ) {
        MonteRayHostFree(points, Base::isManagedMemory );
    } else {
        MonteRayDeviceFree( points );
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

#ifdef __CUDACC__
    if( Base::debug ) {
        std::cout << "Debug: RayList_t::copy (const RayList_t<N>& rhs) \n";
    }

    if( Base::isCudaIntermediate && rhs->isCudaIntermediate ) {
        throw std::runtime_error("RayList_t::copy -- can NOT copy CUDA intermediate to CUDA intermediate.");
    }

    if( !Base::isCudaIntermediate && !rhs->isCudaIntermediate ) {
        throw std::runtime_error("RayList_t::copy -- can NOT copy CUDA non-intermediate to CUDA non-intermediate.");
    }

    if( nAllocated > 0 && nAllocated != rhs->nAllocated) {
        throw std::runtime_error("RayList_t::copy -- can NOT change the size of the RayList.");
    }

    if( Base::isCudaIntermediate ) {
        // target is the intermediate, origin is the host
        if( points == NULL ) {
            points = (RAY_T*) MONTERAYDEVICEALLOC( rhs->nAllocated*sizeof(RAY_T), "device - RayList_t::points" );
        }
        MonteRayMemcpy(points, rhs->points, rhs->nAllocated*sizeof(RAY_T), cudaMemcpyHostToDevice);
    } else {
        // target is the host, origin is the intermediate
        MonteRayMemcpy(rhs->points, points, rhs->nAllocated*sizeof(RAY_T), cudaMemcpyDeviceToHost);
    }

    nAllocated = rhs->nAllocated;
    nUsed = rhs->nUsed;
#else
    throw std::runtime_error("RayList_t::copy -- Only valid when compiling with CUDA.");
#endif
}


template<unsigned N>
void
RayList_t<N>::writeToFile( const std::string& filename) const {
    std::ofstream out;
    out.open( filename.c_str(), std::ios::binary | std::ios::out);
    write( out );
    out.close();
}

template<unsigned N>
void
RayList_t<N>::readFromFile( const std::string& filename) {
    std::ifstream in;
    in.open( filename.c_str(), std::ios::binary | std::ios::in);
    if( ! in.good() ) {
        throw std::runtime_error( "MonteRayNextEventEstimator::readFromFile -- can't open file for reading" );
    }
    read( in );
    in.close();
}

template class RayList_t<1>;
template class RayList_t<3>;
template class CopyMemoryBase<RayList_t<1>>;
template class CopyMemoryBase<RayList_t<3>>;

} // end namespace


