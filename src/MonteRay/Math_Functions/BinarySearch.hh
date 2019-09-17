#ifndef BINARYSEARCH_HH_
#define BINARYSEARCH_HH_

#include "MonteRayTypes.hh"

#ifdef __CUDACC__
#include <thrust/execution_policy.h>
#include <thrust/distance.h>
#include <thrust/binary_search.h>
#include <thrust/system/detail/generic/scalar/binary_search.h>
#include <thrust/device_ptr.h>

#else
#include <algorithm>
#include <iterator>
#endif

namespace MonteRay {

template<typename T, typename VALUE_T>
CUDA_CALLABLE_MEMBER unsigned UpperBoundIndex(const T* const values, unsigned count, const VALUE_T value ) {
    // modified from http://en.cppreference.com/w/cpp/algorithm/upper_bound
    unsigned it, step;
    unsigned first = 0U;

    while (count > 0U) {
        it = first;
        step = count / 2U;
        it += step;
        if(!(value < values[it])) {
            first = ++it;
            count -= step + 1;
        } else {
            count = step;
        }
    }
    return first ;
}

template<typename T, typename VALUE_T>
CUDA_CALLABLE_MEMBER unsigned LowerBoundIndex(const T* const values, unsigned count, const VALUE_T value ) {
    // modified from http://en.cppreference.com/w/cpp/algorithm/upper_bound
    unsigned first = UpperBoundIndex(values,count,value);
    if( first > 0U ) { --first; }
    return first ;
}

/// \brief Wrapper that handles the dispatch to either std::upper_bound or thrust::upper_bound
///
/// \details Does not handle host to device dispatch.  Call directly to thrust::upper_bound for that.
template<typename ForwardIterator, typename LessThanComparable >
inline
CUDA_CALLABLE_MEMBER
ForwardIterator upper_bound( ForwardIterator first, ForwardIterator last, const LessThanComparable& value ) {
#ifdef __CUDACC__
    auto N = thrust::distance(first, last);
    //return thrust::system::detail::generic::scalar::upper_bound_n( first, N, value, thrust::less<LessThanComparable>());
    return thrust::system::detail::generic::scalar::upper_bound( first, last, value, thrust::less<LessThanComparable>());
    //return thrust::upper_bound(thrust::device, first, first+N, value);
#else
    return std::upper_bound(first, last, value);
#endif
}

/// \brief Wrapper that handles the dispatch to either std::distance or thrust::distance
template<typename InputIterator >
inline
CUDA_CALLABLE_MEMBER
#ifdef __CUDACC__
typename thrust::iterator_traits<InputIterator>::difference_type
#else
typename std::iterator_traits<InputIterator>::difference_type
#endif
distance( InputIterator first, InputIterator last ) {
#ifdef __CUDACC__
    return thrust::distance(first, last);
#else
    return std::distance(first, last );
#endif
}

/// \brief LowerBoundIndex - perform binary search over a range
///
/// \details Search an array "values" starting at location "start" and
///          ending at location "end".  To search the entire array "start"
///          should be 0, and "end" should be the number of elements in "values".
template<typename T, typename VALUE_T>
CUDA_CALLABLE_MEMBER size_t LowerBoundIndex(const T* const values, size_t start, size_t end, const VALUE_T value ) {
    int index = MonteRay::distance( values, MonteRay::upper_bound<const T*, VALUE_T>( static_cast<const T*>(values+start), static_cast<const T*>(values+end), value) );
    if( index > 0 ) --index;
    return index;
}

}

#endif /* BINARYSEARCH_HH_ */
