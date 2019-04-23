#include <UnitTest++.h>

#include <vector>
#include <cmath>

#ifdef __CUDACC__
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#endif

#include "CrossSectionHash.hh"

namespace Log2Hash_tester_namespace {

using namespace MonteRay;

class testLog2Hash {

public:

    CUDA_CALLABLE_MEMBER
    static size_t hashFunction(const double value) {
        // For double
        // shifts the bits and returns binary equivalent integer
        //std::cout << "Debug -- Calling hashFunction(double)\n";

        MONTERAY_ASSERT_MSG( value >= 0.0, "Negative values are not allowed.");

        std::uint64_t i = (( std::log2(value)+40.0)*100.0);
        return i;
    }

    CUDA_CALLABLE_MEMBER
    static size_t hashFunction(const float value) {
        // For float
        // shifts the bits and returns binary equivalent integer
        //std::cout << "Debug -- Calling hashFunction(float)\n";

        MONTERAY_ASSERT_MSG( value >= 0.0f, "Negative values are not allowed.");

        std::uint64_t i = (( std::log2(value)+40.0f)*100.0f);
        return i;
    }

    template < typename TV, typename std::enable_if< sizeof(TV) == 8 >::type* = nullptr >
    CUDA_CALLABLE_MEMBER
    static
    TV invHashFunction( const size_t index, const size_t minIndex = 0 ) {
        //std::cout << "Debug -- Calling invHashFunction(index)->double\n";

        TV value = std::exp2( ((index + minIndex) / 100.0 ) - 40.0 );
        return value;
    }

    template < typename TV, typename std::enable_if< sizeof(TV) == 4 >::type* = nullptr >
    CUDA_CALLABLE_MEMBER
    static
    TV invHashFunction( const size_t index, const size_t minIndex = 0 ) {
        //std::cout << "Debug -- Calling invHashFunction(index)->float\n";

        TV value = std::exp2( ((index + minIndex) / 100.0f ) - 40.0f );
        return value;
    }

};

SUITE( Log2Hash_tester ) {
    using doubleXSHash = CrossSectionHash_t<double, testLog2Hash>;
    using floatXSHash = CrossSectionHash_t<float, testLog2Hash>;

    typedef doubleXSHash::NuclearData_t DoubleNuclearData_t;
    typedef std::vector<double> DoubleNuclearDataVec_t;
    typedef doubleXSHash::Index_t DoubleIndex_t;

    typedef floatXSHash::NuclearData_t FloatNuclearData_t;
    typedef std::vector<float> FloatNuclearDataVec_t;
    typedef floatXSHash::Index_t FloatIndex_t;

    TEST( ctor_w_double ) {
        DoubleNuclearDataVec_t testValues = { 1.0, 2.0};
        doubleXSHash hash(testValues);

        double min = hash.getMinValue();
        double max = hash.getMaxValue();

        CHECK_EQUAL( 345, size_t( (std::log2( 1e-11 ) + 40)*100 ) );
        CHECK_EQUAL( 678, size_t( (std::log2( 1e-10 ) + 40)*100 ) );
        CHECK_EQUAL( 4996, size_t( (std::log2( 1e+3 ) + 40)*100 ) );

        CHECK_EQUAL( 4996-345 + 1, hash.size() );
        CHECK_EQUAL( 4652, hash.size() );
        CHECK_EQUAL( 345, hash.getMinIndex() );
        CHECK_EQUAL( 4651, hash.getHashIndex( hash.getMaxValue()) );

        //printf( "Debug exp2( ( ( 0 + 345) / 100.0) - 40.0 ) = %20.15e \n", std::exp2( ( ( 0 + 345) / 100.0) - 40.0 ) );
        CHECK_EQUAL( 9.939251007413245e-12 , testLog2Hash::invHashFunction<double>( 0, 345 ) );
        CHECK_EQUAL( 9.939251007413245e-12 , hash.invHashFunction( 0 ) );
    }


    TEST( ctor_w_float ) {
        FloatNuclearDataVec_t testValues = { 1.0, 2.0};
        doubleXSHash hash(testValues);
        CHECK_EQUAL( 4652, hash.size() );
    }

    TEST( getHashIndex_double ) {
        DoubleNuclearDataVec_t testValues = { 1.0, 2.0};
        doubleXSHash hash(testValues);
        CHECK_EQUAL( 9.939251007413245e-12 , hash.getMinValue() );
        CHECK_EQUAL( 345, doubleXSHash::hashFunction( hash.getMinValue()) );
        CHECK_EQUAL( 345, hash.getMinIndex() );
        CHECK_EQUAL( 0, hash.getHashIndex( hash.getMinValue() ) );
        CHECK_EQUAL( 9.939251007413245e-12, hash.invHashFunction( 0 ) );

        //printf( "Debug exp2( ( ( 4651 + 345) / 100.0) - 40.0 ) = %20.15e \n", std::exp2( ( ( 4651 + 345) / 100.0) - 40.0 ) );
        CHECK_EQUAL( 9.959986661501810e+02, hash.getMaxValue() );
        CHECK_EQUAL( 4996, doubleXSHash::hashFunction( hash.getMaxValue()) );
        CHECK_EQUAL( 4996, hash.getMaxIndex() );
        CHECK_EQUAL( 4651, hash.getHashIndex( hash.getMaxValue()) );
        CHECK_EQUAL( 9.959986661501810e+02, hash.invHashFunction( hash.getNumIndices() -1  ));

//        printf( "Debug: hash.invHashFunction( 1 ) = %20.15e \n", hash.invHashFunction( 1 ) );
//        printf( "Debug: hash.invHashFunction( 2 ) = %20.15e \n", hash.invHashFunction( 2 ) );
//        printf( "Debug: hash.invHashFunction( 3 ) = %20.15e \n", hash.invHashFunction( 3 ) );
        CHECK_CLOSE( 1.000838396532159e-11, hash.invHashFunction( 1 ), 1e-26 );
        CHECK_CLOSE( 1.007799778097923e-11, hash.invHashFunction( 2 ), 1e-26);
        CHECK_CLOSE( 1.014809579901632e-11, hash.invHashFunction( 3 ), 1e-26 );

        CHECK_EQUAL( 37320, hash.bytesize() );
    }


    TEST( getHashIndex_float ) {
        FloatNuclearDataVec_t testValues = { 1.0, 2.0};
        floatXSHash hash(testValues);
        CHECK_EQUAL( 4652, hash.size() );

        //printf( "Debug exp2( ( ( 0 + 345) / 100.0) - 40.0 ) = %20.15e \n", std::exp2( ( ( 0 + 345) / 100.0f) - 40.0f ) );
        CHECK_CLOSE( 9.939256015445430e-12 , hash.getMinValue(), 1e-26 );
        CHECK_EQUAL( 345, floatXSHash::hashFunction( hash.getMinValue()) );
        CHECK_EQUAL( 345, hash.getMinIndex() );
        CHECK_EQUAL( 0, hash.getHashIndex( hash.getMinValue() ) );
        CHECK_CLOSE( 9.939256015445430e-12 , hash.invHashFunction( 0 ), 1e-26 );

        //printf( "Debug exp2( ( ( 4651 + 345) / 100.0) - 40.0 ) = %20.15e \n", std::exp2( ( ( 4651 + 345) / 100.0f) - 40.0f ) );
        CHECK_EQUAL( 9.959980468750000e+02, hash.getMaxValue() );
        CHECK_EQUAL( 4996, floatXSHash::hashFunction( 1e+3 ) );
        CHECK_EQUAL( 4996, floatXSHash::hashFunction( hash.getMaxValue()) );
        CHECK_EQUAL( 4996, hash.getMaxIndex() );
        CHECK_EQUAL( 4651, hash.getHashIndex( hash.getMaxValue()) );
        CHECK_EQUAL( 9.959980468750000e+02, hash.invHashFunction( hash.getNumIndices() -1  ));

        //printf( "Debug: hash.invHashFunction( 0 ) = %20.15e \n", hash.invHashFunction( 0 ) );
        //printf( "Debug: hash.invHashFunction( 1 ) = %20.15e \n", hash.invHashFunction( 1 ) );
        //printf( "Debug: hash.invHashFunction( 2 ) = %20.15e \n", hash.invHashFunction( 2 ) );
        //printf( "Debug: hash.invHashFunction( 3 ) = %20.15e \n", hash.invHashFunction( 3 ) );
        CHECK_CLOSE( 1.000837780706920e-11, hash.invHashFunction( 1 ), 1e-26 );
        CHECK_CLOSE( 1.007800613794796e-11, hash.invHashFunction( 2 ), 1e-26);
        CHECK_CLOSE( 1.014809243582437e-11, hash.invHashFunction( 3 ), 1e-26 );

        CHECK_EQUAL( 37304, hash.bytesize() );
    }

    TEST( test_all_bin_eneries_are_valid_double ) {
        using T = double;
        DoubleNuclearDataVec_t testValues = { 1.0, 2.0};
        doubleXSHash hash(testValues);

        for( unsigned i=0; i< hash.size(); ++i ){
            T value = hash.invHashFunction( i );
            CHECK_EQUAL( true, std::isnormal(value) );
        }
    }

    TEST( test_all_bin_eneries_are_valid_float ) {
        using T = float;
        FloatNuclearDataVec_t testValues = { 1.0, 2.0};
        floatXSHash hash(testValues);

        for( unsigned i=0; i< hash.size(); ++i ){
            T value = hash.invHashFunction( i );
            CHECK_EQUAL( true, std::isnormal(value) );
        }
    }

    TEST( testLowerBin_double ) {
        using T = double;
        //                                        0      1      2           3      4    5    6
        DoubleNuclearDataVec_t testValues = { 1e-13, 5e-13, 1e-11, 1.0009e-11, 1e-10, 1.0, 1e6 };
        doubleXSHash hash(testValues);

        CHECK_EQUAL( 37320, hash.bytesize() );

        CHECK_EQUAL( 1, hash.getBinLo( 0 ) ); // 9.939256015445430e-12
        CHECK_EQUAL( 2, hash.getBinLo( 1 ) ); // 1.000837780706920e-11
        CHECK_EQUAL( 3, hash.getBinLo( 2 ) ); // 1.007800613794796e-11
        CHECK_EQUAL( 3, hash.getBinLo( 3 ) ); // 1.014809243582437e-11

        CHECK_EQUAL( 0, hash.getIndex( 1e-14 ).first );
        CHECK_EQUAL( 0, hash.getIndex( 1e-14 ).second );

        T energy = 1e-13;
        CHECK_EQUAL( 0, hash.getHashIndex(energy) );
        CHECK_EQUAL( 9.939251007413245e-12, hash.getMinValue() );
        CHECK_EQUAL( 0, hash.getIndex( 1e-13 ).first );
        CHECK_EQUAL( 0, hash.getIndex( 1e-13 ).second );

        energy = 1e-12;
        CHECK_EQUAL( 0, hash.getHashIndex(energy) );
        CHECK_EQUAL( 1, hash.getBinLo( 0 ) ); // 9.939256015445430e-12
        CHECK_EQUAL( 2, hash.getBinLo( 1 ) ); // 1.000837780706920e-11
        CHECK_EQUAL( 1, hash.getIndex( 1e-12 ).first);
        CHECK_EQUAL( 3, hash.getIndex( 1e-12 ).second);

        CHECK_EQUAL( 1, hash.getIndex( 1e-11 ).first);
        CHECK_EQUAL( 3, hash.getIndex( 1e-11 ).second);

        energy = 1.00091e-11;
        CHECK_EQUAL( 1, hash.getHashIndex(energy) );
        CHECK_CLOSE( 1.000837780706920e-11, hash.invHashFunction(1), 1e-15 );
        CHECK_CLOSE( 1.007800613794796e-11, hash.invHashFunction(2), 1e-15 );
        CHECK_EQUAL( 2, hash.getIndex( energy ).first );
        CHECK_EQUAL( 4, hash.getIndex( energy ).second );

        // simulate cross-section lookup using hash indices
        unsigned index = std::distance( testValues.begin(),
                                         std::upper_bound( testValues.begin()+hash.getIndex( energy ).first,
                                                           testValues.begin()+hash.getIndex( energy ).second,
                                                           energy )
                                      );
        if( index > 0 ) --index;
        CHECK_EQUAL( 3, index);

        CHECK_EQUAL( 5, hash.getIndex( 1e2 ).first );
        CHECK_EQUAL( 6, hash.getIndex( 1e2 ).second );

        CHECK_EQUAL( 4651, hash.getHashIndex( 1e3 ) );
        CHECK_EQUAL( 5, hash.getIndex( 1e3 ).first );
        CHECK_EQUAL( 7, hash.getIndex( 1e3 ).second );

        CHECK_EQUAL( 4651, hash.getHashIndex( 1e4 ) );
        CHECK_EQUAL( 5, hash.getIndex( 1e4 ).first );
        CHECK_EQUAL( 7, hash.getIndex( 1e4 ).second );

        energy = 1e5;
        CHECK_EQUAL( 4651, hash.getHashIndex(energy) );
        CHECK_EQUAL( 5, hash.getBinLo( 4651 ) );
        CHECK_EQUAL( 5, hash.getIndex( energy ).first );
        CHECK_EQUAL( 7, hash.getIndex( energy ).second );

        CHECK_EQUAL( 4651, hash.getHashIndex( 1e6 ) );
        CHECK_EQUAL( 5, hash.getIndex( 1e6 ).first );
        CHECK_EQUAL( 7, hash.getIndex( 1e6 ).second );

        CHECK_EQUAL( 6, hash.getIndex( 1e7 ).first );
        CHECK_EQUAL( 7, hash.getIndex( 1e7 ).second );
    }
    TEST( testLowerBin_float ) {
        using T = float;
        //                                        0      1      2           3      4    5    6
        FloatNuclearDataVec_t testValues = { 1e-13, 5e-13, 1e-11, 1.0009e-11, 1e-10, 1.0, 1e6 };
        floatXSHash hash(testValues);

        CHECK_EQUAL( 37304, hash.bytesize() );

        CHECK_EQUAL( 1, hash.getBinLo( 0 ) ); // 9.939256015445430e-12
        CHECK_EQUAL( 2, hash.getBinLo( 1 ) ); // 1.000837780706920e-11
        CHECK_EQUAL( 3, hash.getBinLo( 2 ) ); // 1.007800613794796e-11
        CHECK_EQUAL( 3, hash.getBinLo( 3 ) ); // 1.014809243582437e-11

        CHECK_EQUAL( 0, hash.getIndex( 1e-14 ).first );
        CHECK_EQUAL( 0, hash.getIndex( 1e-14 ).second );

        T energy = 1e-13;
        CHECK_EQUAL( 0, hash.getHashIndex(energy) );
        CHECK_EQUAL( 9.939256015445430e-12, hash.getMinValue() );
        CHECK_EQUAL( 0, hash.getIndex( 1e-13 ).first );
        CHECK_EQUAL( 0, hash.getIndex( 1e-13 ).second );

        energy = 1e-12;
        CHECK_EQUAL( 0, hash.getHashIndex(energy) );
        CHECK_EQUAL( 1, hash.getBinLo( 0 ) ); // 9.939256015445430e-12
        CHECK_EQUAL( 2, hash.getBinLo( 1 ) ); // 1.000837780706920e-11
        CHECK_EQUAL( 1, hash.getIndex( 1e-12 ).first);
        CHECK_EQUAL( 3, hash.getIndex( 1e-12 ).second);

        CHECK_EQUAL( 1, hash.getIndex( 1e-11 ).first);
        CHECK_EQUAL( 3, hash.getIndex( 1e-11 ).second);

        energy = 1.00091e-11;
        CHECK_EQUAL( 1, hash.getHashIndex(energy) );
        CHECK_CLOSE( 1.000837780706920e-11, hash.invHashFunction(1), 1e-15 );
        CHECK_CLOSE( 1.007800613794796e-11, hash.invHashFunction(2), 1e-15 );
        CHECK_EQUAL( 2, hash.getIndex( energy ).first );
        CHECK_EQUAL( 4, hash.getIndex( energy ).second );

        // simulate cross-section lookup using hash indices
        unsigned index = std::distance( testValues.begin(),
                                         std::upper_bound( testValues.begin()+hash.getIndex( energy ).first,
                                                           testValues.begin()+hash.getIndex( energy ).second,
                                                           energy )
                                      );
        if( index > 0 ) --index;
        CHECK_EQUAL( 3, index);

        CHECK_EQUAL( 5, hash.getIndex( 1e2 ).first );
        CHECK_EQUAL( 6, hash.getIndex( 1e2 ).second );

        CHECK_EQUAL( 4651, hash.getHashIndex( 1e3 ) );
        CHECK_EQUAL( 5, hash.getIndex( 1e3 ).first );
        CHECK_EQUAL( 7, hash.getIndex( 1e3 ).second );

        CHECK_EQUAL( 4651, hash.getHashIndex( 1e4 ) );
        CHECK_EQUAL( 5, hash.getIndex( 1e4 ).first );
        CHECK_EQUAL( 7, hash.getIndex( 1e4 ).second );

        energy = 1e5;
        CHECK_EQUAL( 4651, hash.getHashIndex(energy) );
        CHECK_EQUAL( 5, hash.getBinLo( 4651 ) );
        CHECK_EQUAL( 5, hash.getIndex( energy ).first );
        CHECK_EQUAL( 7, hash.getIndex( energy ).second );

        CHECK_EQUAL( 4651, hash.getHashIndex( 1e6 ) );
        CHECK_EQUAL( 5, hash.getIndex( 1e6 ).first );
        CHECK_EQUAL( 7, hash.getIndex( 1e6 ).second );

        CHECK_EQUAL( 6, hash.getIndex( 1e7 ).first );
        CHECK_EQUAL( 7, hash.getIndex( 1e7 ).second );
    }



    TEST( test_bytesize_with_larger_array ) {
           using T = double;
           DoubleNuclearDataVec_t testValues = { 1e-13, 5e-13, 1e-11, 1.0005e-11, 1e-10, 1e-9, 1e-8, 1e-5, 1.0, 1e6 };
           doubleXSHash hash(testValues);

           // doesn't change based on the array size- fixed size
           CHECK_EQUAL( 37320, hash.bytesize() );
    }


    TEST( energies_stop_before_end_of_hashTable ) {
         using T = double;
         //                                        0      1      2           3      4   5    6   7
         DoubleNuclearDataVec_t testValues = { 1e-13, 5e-13, 1e-11, 1.0009e-11, 1e-10, .1, 1.0, 20 };
         doubleXSHash hash(testValues);

         double energy = .5;
         CHECK_EQUAL( 3555, hash.getHashIndex(energy) );
         CHECK_EQUAL( 5, hash.getBinLo( 3555 ) );
         CHECK_EQUAL( 5, hash.getIndex( energy ).first );
         CHECK_EQUAL( 6, hash.getIndex( energy ).second );

         energy = 1.5;
         CHECK_EQUAL( 3713, hash.getHashIndex(energy) );
         CHECK_EQUAL( 6, hash.getBinLo( 3713 ) );
         CHECK_EQUAL( 6, hash.getIndex( energy ).first );
         CHECK_EQUAL( 7, hash.getIndex( energy ).second );

         energy = 20.0;
         CHECK_EQUAL( 4087, hash.getHashIndex(energy) );
         CHECK_EQUAL( 6, hash.getBinLo( 4087 ) );
         CHECK_CLOSE( 19.9733, hash.invHashFunction(4087), 1e-4 );
         CHECK_CLOSE( 20.1122, hash.invHashFunction(4088), 1e-4 );
         CHECK_EQUAL( 6, hash.getIndex( energy ).first );
         CHECK_EQUAL( 8, hash.getIndex( energy ).second );

         energy = 100.0;
         CHECK_EQUAL( 4319, hash.getHashIndex(energy) );
         CHECK_EQUAL( 7, hash.getBinLo( 4319 ) );
         CHECK_EQUAL( 7, hash.getIndex( energy ).first );
         CHECK_EQUAL( 8, hash.getIndex( energy ).second );

         energy = 1000.0;
         CHECK_EQUAL( 4651, hash.getHashIndex(energy) );
         CHECK_EQUAL( 7, hash.getBinLo( 4651 ) );
         CHECK_EQUAL( 7, hash.getIndex( energy ).first );
         CHECK_EQUAL( 8, hash.getIndex( energy ).second );

     }

    TEST( energies_start_after_beginning_of_hashTable ) {
         using T = double;
         //                                        0     1    2   3
         DoubleNuclearDataVec_t testValues = {   1e-9, .1, 1.0, 20 };
         doubleXSHash hash(testValues);

         double energy = 1e-12;
         CHECK_EQUAL( 0, hash.getHashIndex(energy) );
         CHECK_EQUAL( 0, hash.getBinLo( 0 ) );
         CHECK_EQUAL( 0, hash.getIndex( energy ).first );
         CHECK_EQUAL( 0, hash.getIndex( energy ).second );

         energy = 1e-11;
         CHECK_EQUAL( 0, hash.getHashIndex(energy) );
         CHECK_EQUAL( 0, hash.getBinLo( 0 ) );
         CHECK_EQUAL( 0, hash.getIndex( energy ).first );
         CHECK_EQUAL( 0, hash.getIndex( energy ).second );

         energy = 1.1e-11;
         CHECK_EQUAL( 14, hash.getHashIndex(energy) );
         CHECK_EQUAL( 0, hash.getBinLo( 14 ) );
         CHECK_EQUAL( 0, hash.getIndex( energy ).first );
         CHECK_EQUAL( 0, hash.getIndex( energy ).second );

         energy = 1.2e-11;
         CHECK_EQUAL( 27, hash.getHashIndex(energy) );
         CHECK_EQUAL( 0, hash.getBinLo( 27 ) );
         CHECK_EQUAL( 0, hash.getIndex( energy ).first );
         CHECK_EQUAL( 0, hash.getIndex( energy ).second );

         energy = 1.0e-10;
         CHECK_EQUAL( 333, hash.getHashIndex(energy) );
         CHECK_EQUAL( 0, hash.getBinLo( 333 ) );
         CHECK_EQUAL( 0, hash.getIndex( energy ).first );
         CHECK_EQUAL( 0, hash.getIndex( energy ).second );

         energy = 1.1e-9;
         CHECK_EQUAL( 679, hash.getHashIndex(energy) );
         CHECK_EQUAL( 0, hash.getBinLo( 679 ) );
         CHECK_EQUAL( 0, hash.getIndex( energy ).first );
         CHECK_EQUAL( 1, hash.getIndex( energy ).second );

         energy = 0.01;
         CHECK_EQUAL( 2990, hash.getHashIndex(energy) );
         CHECK_EQUAL( 0, hash.getBinLo( 2990 ) );
         CHECK_EQUAL( 0, hash.getIndex( energy ).first );
         CHECK_EQUAL( 1, hash.getIndex( energy ).second );

         energy = 0.15;
         CHECK_EQUAL( 3381, hash.getHashIndex(energy) );
         CHECK_EQUAL( 1, hash.getBinLo( 3381 ) );
         CHECK_EQUAL( 1, hash.getIndex( energy ).first );
         CHECK_EQUAL( 2, hash.getIndex( energy ).second );

     }

}

} // end namespace
