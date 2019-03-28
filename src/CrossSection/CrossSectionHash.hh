#ifndef CROSSSECTIONHASH_HH_
#define CROSSSECTIONHASH_HH_

#include <bitset>
#include <algorithm>
#include <iostream>

#include "MonteRayTypes.hh"
#include "MonteRayConstants.hh"
#include "MonteRayAssert.hh"
#include "ManagedAllocator.hh"

namespace MonteRay {

template<typename T = gpuFloatType_t>
class CrossSectionHash : public Managed {
public:
    typedef size_t Index_t;
    typedef T NuclearData_t;
    typedef managed_vector<NuclearData_t> NuclearDataVec_t;
    typedef managed_vector< Index_t > IndexVec_t;

    CrossSectionHash(NuclearDataVec_t& XValues ) :
        minValue( invHashFunction<T>(0) ),
        maxValue( invHashFunction<T>(numIndices-1) )
    {
        //std::cout << "Debug: CrossSectionHash::ctor - numIndices = " << numIndices << "\n";
        BinLo.reserve( numIndices );

        if( ! XValues.empty() ) {
            minTable = XValues.front();
            maxTable = XValues.back();
            tableSize = XValues.size();

            for( Index_t i=0; i<numIndices; ++i){
                T energy = invHashFunction<T>(i);
                if( ! std::isnormal(energy) ) {
                    throw std::runtime_error("CrossSectionHash::Constructor -- energy in bin structure is not a normal value.");
                }

                Index_t index = std::distance(XValues.begin(),std::upper_bound(XValues.begin(),XValues.end(),energy));
                if ( index > 0 ) --index;

                BinLo.push_back( index );
            }
        }
    };
    ~CrossSectionHash() = default;

    Index_t size() const { return numIndices;}

    Index_t getHashIndex(const NuclearData_t value ) const {
        Index_t index = hashFunction<T>(value);
        if( index < minIndex ) {
            return 0;
        }
        index -= minIndex;
        if( index > numIndices-1 ) {
            return numIndices-1;
        }
        return index;
    }

    std::pair<int,int> getIndex( const NuclearData_t value ) const {
        int lower;
        int upper;
        getIndex(lower, upper, value);
        return std::make_pair( lower, upper);
    }

    void getIndex( int& lowerBin, int& upperBin, const NuclearData_t value ) const {
        //std::cout << "Debug:  getIndex -- value=" << value << " minTable=" << minTable << "\n";
        if( value <= minTable ) { lowerBin = 0; upperBin=0; return; }
        if( value > maxTable ) { lowerBin =  tableSize-1; upperBin= tableSize-1; return; }

        Index_t index = getHashIndex(value);
        MONTERAY_ASSERT( index < BinLo.size() );

        lowerBin = getBinLo( index );
        if( lowerBin >= tableSize - 2 ) { upperBin= tableSize-1; return;  }

        upperBin = getBinLo( index + 1 ) + 1;

        return;
    }

    Index_t getBinLo( const Index_t index ) const {
        if( index >= numIndices ) {
            std::cout << "Debug: CrossSectionHash::getBinLo -- index >= numIndices,  index=" <<
                    index << " numIndices= " << numIndices << "\n";
        }
        MONTERAY_ASSERT( index < numIndices );
        return BinLo[ index ];
    }

private:
    static constexpr NuclearData_t targetMinValue = 1.0e-11; // min. value of the table
    static constexpr NuclearData_t targetMaxValue = 1.0e+3;  // max. value of the table

public:

    template < typename TV = T, typename std::enable_if< 4 < sizeof(TV) >::type* = nullptr >
    static int hashFunction(const TV value) {
        // For double
        // shifts the bits and returns binary equivalent integer
        //std::cout << "Debug -- Calling hashFunction(double)\n";

        int i = *((uint64_t*)(void*)&value) >> 45;
        return i;
    }

    template < typename TV = T, typename std::enable_if< sizeof(TV) < 5 >::type* = nullptr >
    static int hashFunction(const TV value) {
        // For float
        // shifts the bits and returns binary equivalent integer
        //std::cout << "Debug -- Calling hashFunction(float)\n";

        int i = *((uint32_t*)(void*)&value) >> 16;
        return i;
    }

    const Index_t minIndex = hashFunction<T>( targetMinValue );
    const Index_t maxIndex = hashFunction<T>( targetMaxValue );
    const Index_t numIndices = maxIndex-minIndex+1;

    template < typename TV = T, typename std::enable_if< 4 < sizeof(TV) >::type* = nullptr >
    TV invHashFunction( const Index_t index ) const {
        //std::cout << "Debug -- Calling invHashFunction(index)->double\n";
        Index_t value = (index + minIndex) << 45 ;
        return *((TV*)(void*)(&value)) ;
    }

    template < typename TV = T, typename std::enable_if< sizeof(TV) < 5 >::type* = nullptr >
    TV invHashFunction( const Index_t index ) const {
        //std::cout << "Debug -- Calling invHashFunction(index)->float\n";
        Index_t value = (index + minIndex) << 16 ;
        return *((TV*)(void*)(&value)) ;
    }

    const NuclearData_t minValue = targetMinValue;
    const NuclearData_t maxValue = targetMaxValue;

    NuclearData_t minTable;
    NuclearData_t maxTable;
    unsigned tableSize;

private:
    IndexVec_t BinLo;

};

} /* namespace MonteRay */

#endif /* CROSSSECTIONHASH_HH_ */

