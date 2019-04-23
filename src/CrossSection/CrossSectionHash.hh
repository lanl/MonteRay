#ifndef CROSSSECTIONHASH_HH_
#define CROSSSECTIONHASH_HH_

#include <bitset>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <cstring>

#include "MonteRayTypes.hh"
#include "MonteRayConstants.hh"
#include "MonteRayAssert.hh"
#include "ManagedAllocator.hh"
#include "FasterHash.hh"
#include "BinarySearch.hh"

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#endif

namespace MonteRay {

///\brief Provides a fast hash lookup method for an energy grid.

///\details Provides a fast hash lookup method for a positive energy grid over 1e-11 to 1e3.
///         Intended for energy grids only.
template<typename T = gpuFloatType_t, typename HASHTYPE = FasterHash, class Allocator = managed_allocator<size_t> >
class CrossSectionHash_t : public Managed {
public:
    using Index_t = size_t;
    using NuclearData_t = T;
    using IndexVec_t = std::vector< Index_t, Allocator >;

    template<typename ENERGYBINCONTAINER_T>
    CrossSectionHash_t(ENERGYBINCONTAINER_T& XValues ) :
        minValue( invHashFunction(0) ),
        maxValue( invHashFunction(numIndices-1) )
    {
        if( XValues.empty() ) {
#ifndef __CUDA_ARCH__
            throw std::runtime_error("CrossSectionHash_t::CrossSectionHash_t(XValues) -- XValues can not be empty");
#endif
        }

        minTable = XValues.front();
        maxTable = XValues.back();
        tableSize = XValues.size();

        checkValidEnergyBins(XValues);

        BinLo_vec.resize( numIndices );
        BinLo = BinLo_vec.data();
        BinLo_size = BinLo_vec.size();

        // using std::iota and std::transform to fill in the hash table
        // using iota instead of a traditional for loop will incur some additional
        // memory access overhead.   This is a tradeoff some that transform can easily
        // be used through thrust.

        const auto begin = XValues.data();
        const auto end = begin + XValues.size();

        // functor to convert integer bin numbers to energy then to XValue index
         auto getXValuesIndices = [=] (Index_t i ) {

             // convert bin i value to energy  (goes from 1e-11 to 1e3)
             T energy = HASHTYPE::template invHashFunction<T>(i, minIndex );
             MONTERAY_ASSERT_MSG(std::isnormal(energy), "Energy in hash bin structure is not a normal value." );

             // perform lowerbound search for energy within XValues
             Index_t index = MonteRay::distance(begin,MonteRay::upper_bound(begin,end,energy));
             if ( index > 0 ) --index;

             return index;
         };

        // fill BinLo_vec with the index of the bin with 0, 1, 2, ... NBins  (use sequence in thrust)
        std::iota(BinLo, BinLo + BinLo_size, 0);

        // lookup the lowerbound index for each bin  i
        std::transform( BinLo, BinLo + BinLo_size, BinLo, getXValuesIndices );

    }

    ~CrossSectionHash_t() = default;

    CUDA_CALLABLE_MEMBER Index_t size() const { return numIndices;}

    CUDA_CALLABLE_MEMBER
    Index_t getHashIndex(const NuclearData_t value ) const {
        Index_t index = hashFunction(value);
        if( index < minIndex ) {
            return 0;
        }
        index -= minIndex;
        if( index > numIndices-1 ) {
            return numIndices-1;
        }
        return index;
    }

    std::pair<Index_t,Index_t> getIndex( const NuclearData_t value ) const {
        Index_t lower;
        Index_t upper;
        getIndex(lower, upper, value);
        return std::make_pair( lower, upper);
    }

    CUDA_CALLABLE_MEMBER
    void getIndex( Index_t& lowerBin, Index_t& upperBin, const NuclearData_t value ) const {
        if( value <= minTable ) { lowerBin = 0; upperBin=0; return; }
        if( value > maxTable ) { lowerBin =  tableSize-1; upperBin= tableSize; return; }

        Index_t index = getHashIndex(value);

        // get lower cross-section bin
        lowerBin = getBinLo( index );

        // value is at the end of the hash grid
        if( index == size() - 1 ) { upperBin = tableSize; return; }

        // get upper cross-section bin
        upperBin = getBinLo( index + 1 ) + 1;
        MONTERAY_ASSERT_MSG( upperBin <= tableSize, "cross-section table upper bound index is greater than the cross-section table size" );

        return;
    }

    CUDA_CALLABLE_MEMBER
    Index_t getBinLo( const Index_t index ) const {
        MONTERAY_ASSERT_MSG( index < numIndices, "hash index is greater than the size of the hash table" );
        return BinLo[ index ];
    }

    CUDA_CALLABLE_MEMBER
    static Index_t hashFunction(const double value) {
        return HASHTYPE::hashFunction( value );
    }

    CUDA_CALLABLE_MEMBER
    static Index_t hashFunction(const float value) {
        return HASHTYPE::hashFunction( value );
    }

    size_t bytesize() const {
        size_t total = 0;
        total += sizeof( *this );
        total += sizeof( Index_t ) * BinLo_size;
        return total;
    }

    CUDA_CALLABLE_MEMBER Index_t getMinIndex() const { return minIndex; }
    CUDA_CALLABLE_MEMBER Index_t getMaxIndex() const { return maxIndex; }
    CUDA_CALLABLE_MEMBER Index_t getNumIndices() const { return numIndices; }
    CUDA_CALLABLE_MEMBER NuclearData_t getMinValue() const { return minValue; }
    CUDA_CALLABLE_MEMBER NuclearData_t getMaxValue() const { return maxValue; }
    CUDA_CALLABLE_MEMBER NuclearData_t getTableSize() const { return tableSize; }

//private:
    static constexpr NuclearData_t targetMinValue = 1.0e-11; // min. value of the table
    static constexpr NuclearData_t targetMaxValue = 1.0e+3;  // max. value of the table

    const Index_t minIndex = hashFunction( targetMinValue );
    const Index_t maxIndex = hashFunction( targetMaxValue );
    const Index_t numIndices = maxIndex-minIndex+1;

    const NuclearData_t minValue = targetMinValue;
    const NuclearData_t maxValue = targetMaxValue;

    NuclearData_t minTable;
    NuclearData_t maxTable;
    Index_t tableSize;

    IndexVec_t BinLo_vec;
    Index_t* BinLo = nullptr;
    int BinLo_size = 0;

public:

    CUDA_CALLABLE_MEMBER
    T invHashFunction( const Index_t index ) const {
        T value = HASHTYPE::template invHashFunction<T>(index, minIndex );

#ifndef __CUDA_ARCH__
        MONTERAY_ASSERT_MSG(std::isnormal(value), "Value at hash bin is not a normal value." );
#endif
        return value;
    }


//private:

    template<typename ENERGYBINCONTAINER_T>
    void checkValidEnergyBins( const ENERGYBINCONTAINER_T& energyBins ) const {
        T previous_value = 0.0;
        for( auto itr = energyBins.begin(); itr != energyBins.end(); ++itr ){
            MONTERAY_ASSERT_MSG( *itr >= previous_value, "Values are not ascending" );
            previous_value = *itr;
        }
    }

};

using CrossSectionHash = CrossSectionHash_t<>;

} /* namespace MonteRay */

#endif /* CROSSSECTIONHASH_HH_ */

