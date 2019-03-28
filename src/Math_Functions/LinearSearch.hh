#ifndef LINEARSEARCH_HH_
#define LINEARSEARCH_HH_

#include "MonteRayTypes.hh"

namespace MonteRay {

template<typename T, typename VALUE_T>
CUDA_CALLABLE_MEMBER int UpperBoundIndexLinear(const T* const values, const int lower, const int upper, const VALUE_T value ) {
    for(int i=lower; i < upper+1; ++i ){
        if( value < values[ i ] ) {
            return i;
        }
    }

    return upper+1;

}

template<typename T, typename VALUE_T>
CUDA_CALLABLE_MEMBER int LowerBoundIndexLinear(const T* const values, const int lower, const int upper, const VALUE_T value ) {
    int first = UpperBoundIndexLinear(values,lower,upper,value);
    if( first > lower ) { --first; }
    return first ;
}

}

#endif /* LINEARSEARCH_HH_ */
