#ifndef BINARYSEARCH_HH_
#define BINARYSEARCH_HH_

#include "MonteRayTypes.hh"

namespace MonteRay {

template<typename T, typename VALUE_T>
CUDA_CALLABLE_MEMBER unsigned UpperBoundIndex(const T* const values, unsigned count, VALUE_T value ) {
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
CUDA_CALLABLE_MEMBER unsigned LowerBoundIndex(const T* const values, unsigned count, VALUE_T value ) {
    // modified from http://en.cppreference.com/w/cpp/algorithm/upper_bound
    unsigned first = UpperBoundIndex(values,count,value);
    if( first > 0U ) { --first; }
    return first ;
}





}

#endif /* BINARYSEARCH_HH_ */
