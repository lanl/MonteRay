#include <UnitTest++.h>

#include <vector>
#include <cmath>

#ifdef __CUDACC__
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#endif

#include "CrossSectionHash.hh"

namespace LogHash_tester_namespace {

using namespace MonteRay;

class testLogHash {

public:

    CUDA_CALLABLE_MEMBER
    static size_t hashFunction(const double value) {
        // For double
        // shifts the bits and returns binary equivalent integer
        //std::cout << "Debug -- Calling hashFunction(double)\n";

        MONTERAY_ASSERT_MSG( value >= 0.0, "Negative values are not allowed.");

        std::uint64_t i = (( std::log(value)+30.0)*200.0);
        return i;
    }

    CUDA_CALLABLE_MEMBER
    static size_t hashFunction(const float value) {
        // For float
        // shifts the bits and returns binary equivalent integer
        //std::cout << "Debug -- Calling hashFunction(float)\n";

        MONTERAY_ASSERT_MSG( value >= 0.0f, "Negative values are not allowed.");

        std::uint64_t i = (( std::log(value)+30.0f)*200.0f);
        return i;
    }

    template < typename TV, typename std::enable_if< sizeof(TV) == 8 >::type* = nullptr >
    CUDA_CALLABLE_MEMBER
    static
    TV invHashFunction( const size_t index, const size_t minIndex = 0 ) {
        //std::cout << "Debug -- Calling invHashFunction(index)->double\n";

        TV value = std::exp( ((index + minIndex) / 200.0 ) - 30.0 );
        return value;
    }

    template < typename TV, typename std::enable_if< sizeof(TV) == 4 >::type* = nullptr >
    CUDA_CALLABLE_MEMBER
    static
    TV invHashFunction( const size_t index, const size_t minIndex = 0 ) {
        //std::cout << "Debug -- Calling invHashFunction(index)->float\n";

        TV value = std::exp( ((index + minIndex) / 200.0f ) - 30.0f );
        return value;
    }

};

SUITE( LogHash_tester ) {
    using doubleXSHash = CrossSectionHash_t<double, testLogHash>;
    using floatXSHash = CrossSectionHash_t<float, testLogHash>;

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

        //printf( "Debug log( 1e-11 ) = %20.15e \n", std::log( 1e-11 ) );
        CHECK_EQUAL( 934, size_t( (std::log( 1e-11 ) + 30)*200 ) );
        CHECK_EQUAL( 1394, size_t( (std::log( 1e-10 ) + 30)*200 ) );
        CHECK_EQUAL( 7381, size_t( (std::log( 1e+3 ) + 30)*200 ) );

        CHECK_EQUAL( 7381-934 + 1, hash.size() );
        CHECK_EQUAL( 6448, hash.size() );
        CHECK_EQUAL( 934, hash.getMinIndex() );
        CHECK_EQUAL( 6447, hash.getHashIndex( hash.getMaxValue()) );

        //printf( "Debug exp( ( ( 0 + 934) / 200.0) - 30.0 ) = %20.15e \n", std::exp( ( ( 0 + 934) / 200.0) - 30.0 ) );
        CHECK_EQUAL( 9.984372453092965e-12 , testLogHash::invHashFunction<double>( 0, 934 ) );
        CHECK_EQUAL( 9.984372453092965e-12 , hash.invHashFunction( 0 ) );
    }


    TEST( ctor_w_float ) {
        FloatNuclearDataVec_t testValues = { 1.0, 2.0};
        doubleXSHash hash(testValues);
        CHECK_EQUAL( 6448, hash.size() );
    }

    TEST( getHashIndex_double ) {
        DoubleNuclearDataVec_t testValues = { 1.0, 2.0};
        doubleXSHash hash(testValues);
        CHECK_EQUAL( 9.984372453092965e-12 , hash.getMinValue() );
        CHECK_EQUAL( 934, doubleXSHash::hashFunction( hash.getMinValue()) );
        CHECK_EQUAL( 934, hash.getMinIndex() );
        CHECK_EQUAL( 0, hash.getHashIndex( hash.getMinValue() ) );
        CHECK_EQUAL( 9.984372453092965e-12, hash.invHashFunction( 0 ) );

        //printf( "Debug exp( ( ( 6447 + 934) / 200.0) - 30.0 ) = %20.15e \n", std::exp( ( ( 6447 + 934) / 200.0) - 30.0 ) );
        CHECK_EQUAL( 9.972485133152535e+02, hash.getMaxValue() );
        CHECK_EQUAL( 7381, doubleXSHash::hashFunction( hash.getMaxValue()) );
        CHECK_EQUAL( 7381, hash.getMaxIndex() );
        CHECK_EQUAL( 6447, hash.getHashIndex( hash.getMaxValue()) );
        CHECK_EQUAL(9.972485133152535e+02, hash.invHashFunction( hash.getNumIndices() -1  ));

        //printf( "Debug: hash.invHashFunction( 0 ) = %20.15e \n", hash.invHashFunction( 0 ) );
        //printf( "Debug: hash.invHashFunction( 1 ) = %20.15e \n", hash.invHashFunction( 1 ) );
        //printf( "Debug: hash.invHashFunction( 2 ) = %20.15e \n", hash.invHashFunction( 2 ) );
        //printf( "Debug: hash.invHashFunction( 3 ) = %20.15e \n", hash.invHashFunction( 3 ) );
        CHECK_CLOSE( 1.003441932828211e-11, hash.invHashFunction( 1 ), 1e-26 );
        CHECK_CLOSE( 1.008471706447709e-11 , hash.invHashFunction( 2 ), 1e-26);
        CHECK_CLOSE( 1.013526691912393e-11, hash.invHashFunction( 3 ), 1e-26 );

        CHECK_EQUAL( 51672, hash.bytesize() );
    }


    TEST( getHashIndex_float ) {
        FloatNuclearDataVec_t testValues = { 1.0, 2.0};
        floatXSHash hash(testValues);
        CHECK_EQUAL( 9.984373570970373e-12, hash.getMinValue() );
        CHECK_EQUAL( 934, doubleXSHash::hashFunction( hash.getMinValue()) );
        CHECK_EQUAL( 934, hash.getMinIndex() );
        CHECK_EQUAL( 0, hash.getHashIndex( hash.getMinValue() ) );
        CHECK_EQUAL( 9.984373570970373e-12 , hash.invHashFunction( 0 ) );

        //printf( "Debug exp( ( ( 6447 + 934) / 200.0) - 30.0 ) = %20.15e \n", std::exp( ( ( 6447 + 934) / 200.0f) - 30.0f ) );
        CHECK_EQUAL( 9.972473144531250e+02, hash.getMaxValue() );
        CHECK_EQUAL( 7381, doubleXSHash::hashFunction( hash.getMaxValue()) );
        CHECK_EQUAL( 7381, hash.getMaxIndex() );
        CHECK_EQUAL( 6447, hash.getHashIndex( hash.getMaxValue()) );
        CHECK_EQUAL(9.972473144531250e+02, hash.invHashFunction( hash.getNumIndices() -1  ));

        //printf( "Debug: hash.invHashFunction( 0 ) = %20.15e \n", hash.invHashFunction( 0 ) );
        //printf( "Debug: hash.invHashFunction( 1 ) = %20.15e \n", hash.invHashFunction( 1 ) );
        //printf( "Debug: hash.invHashFunction( 2 ) = %20.15e \n", hash.invHashFunction( 2 ) );
        //printf( "Debug: hash.invHashFunction( 3 ) = %20.15e \n", hash.invHashFunction( 3 ) );
        CHECK_CLOSE( 1.003441166963492e-11, hash.invHashFunction( 1 ), 1e-26 );
        CHECK_CLOSE( 1.008472038516173e-11, hash.invHashFunction( 2 ), 1e-26);
        CHECK_CLOSE( 1.013526155363431e-11, hash.invHashFunction( 3 ), 1e-26 );

        CHECK_EQUAL( 51656, hash.bytesize() );
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
        DoubleNuclearDataVec_t testValues = { 1e-13, 5e-13, 1e-11,  1.005e-11, 1e-10, 1.0, 1e6 };
        doubleXSHash hash(testValues);

        CHECK_EQUAL( 51672, hash.bytesize() );

        CHECK_EQUAL( 1, hash.getBinLo( 0 ) ); // 9.984373570970373e-12
        CHECK_EQUAL( 2, hash.getBinLo( 1 ) ); // 1.003441166963492e-11
        CHECK_EQUAL( 3, hash.getBinLo( 2 ) ); // 1.008472038516173e-11
        CHECK_EQUAL( 3, hash.getBinLo( 3 ) ); // 1.013526155363431e-11

        CHECK_EQUAL( 0, hash.getIndex( 1e-14 ).first );
        CHECK_EQUAL( 0, hash.getIndex( 1e-14 ).second );

        T energy = 1e-13;
        CHECK_EQUAL( 0, hash.getHashIndex(energy) );
        CHECK_EQUAL( 9.984372453092965e-12, hash.getMinValue() );
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

        energy = 1.0051e-11;
        CHECK_EQUAL( 1, hash.getHashIndex(energy) );
        CHECK_CLOSE( 1.003441166963492e-11, hash.invHashFunction(1), 1e-15 );
        CHECK_CLOSE( 1.008472038516173e-11, hash.invHashFunction(2), 1e-15 );
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

        CHECK_EQUAL( 6447, hash.getHashIndex( 1e3 ) );
        CHECK_EQUAL( 5, hash.getIndex( 1e3 ).first );
        CHECK_EQUAL( 7, hash.getIndex( 1e3 ).second );

        CHECK_EQUAL( 6447, hash.getHashIndex( 1e4 ) );
        CHECK_EQUAL( 5, hash.getIndex( 1e4 ).first );
        CHECK_EQUAL( 7, hash.getIndex( 1e4 ).second );

        energy = 1e5;
        CHECK_EQUAL( 6447, hash.getHashIndex(energy) );
        CHECK_EQUAL( 5, hash.getBinLo( 6447 ) );
        CHECK_EQUAL( 5, hash.getIndex( energy ).first );
        CHECK_EQUAL( 7, hash.getIndex( energy ).second );

        CHECK_EQUAL( 6447, hash.getHashIndex( 1e6 ) );
        CHECK_EQUAL( 5, hash.getIndex( 1e6 ).first );
        CHECK_EQUAL( 7, hash.getIndex( 1e6 ).second );

        CHECK_EQUAL( 6, hash.getIndex( 1e7 ).first );
        CHECK_EQUAL( 7, hash.getIndex( 1e7 ).second );
    }

    TEST( testLowerBin_float ) {
        using T = float;
        //                                        0      1      2           3      4    5    6
        FloatNuclearDataVec_t testValues = { 1e-13, 5e-13, 1e-11,  1.005e-11, 1e-10, 1.0, 1e6 };
        floatXSHash hash(testValues);

        CHECK_EQUAL( 51656, hash.bytesize() );

        CHECK_EQUAL( 1, hash.getBinLo( 0 ) ); // 9.984373570970373e-12
        CHECK_EQUAL( 2, hash.getBinLo( 1 ) ); // 1.003441166963492e-11
        CHECK_EQUAL( 3, hash.getBinLo( 2 ) ); // 1.008472038516173e-11
        CHECK_EQUAL( 3, hash.getBinLo( 3 ) ); // 1.013526155363431e-11

        CHECK_EQUAL( 0, hash.getIndex( 1e-14 ).first );
        CHECK_EQUAL( 0, hash.getIndex( 1e-14 ).second );

        T energy = 1e-13;
        CHECK_EQUAL( 0, hash.getHashIndex(energy) );
        CHECK_EQUAL( 9.984373570970373e-12, hash.getMinValue() );
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

        energy = 1.0051e-11;
        CHECK_EQUAL( 1, hash.getHashIndex(energy) );
        CHECK_CLOSE( 1.003441166963492e-11, hash.invHashFunction(1), 1e-15 );
        CHECK_CLOSE( 1.008472038516173e-11, hash.invHashFunction(2), 1e-15 );
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

        CHECK_EQUAL( 6447, hash.getHashIndex( 1e3 ) );
        CHECK_EQUAL( 5, hash.getIndex( 1e3 ).first );
        CHECK_EQUAL( 7, hash.getIndex( 1e3 ).second );

        CHECK_EQUAL( 6447, hash.getHashIndex( 1e4 ) );
        CHECK_EQUAL( 5, hash.getIndex( 1e4 ).first );
        CHECK_EQUAL( 7, hash.getIndex( 1e4 ).second );

        energy = 1e5;
        CHECK_EQUAL( 6447, hash.getHashIndex(energy) );
        CHECK_EQUAL( 5, hash.getBinLo( 6447 ) );
        CHECK_EQUAL( 5, hash.getIndex( energy ).first );
        CHECK_EQUAL( 7, hash.getIndex( energy ).second );

        CHECK_EQUAL( 6447, hash.getHashIndex( 1e6 ) );
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
           CHECK_EQUAL( 51672, hash.bytesize() );
    }


    TEST( energies_stop_before_end_of_hashTable ) {
         using T = double;
         //                                        0      1      2           3      4   5    6   7
         DoubleNuclearDataVec_t testValues = { 1e-13, 5e-13, 1e-11, 1.0009e-11, 1e-10, .1, 1.0, 20 };
         doubleXSHash hash(testValues);

         double energy = .5;
         CHECK_EQUAL( 4927, hash.getHashIndex(energy) );
         CHECK_EQUAL( 5, hash.getBinLo( 4927 ) );
         CHECK_EQUAL( 5, hash.getIndex( energy ).first );
         CHECK_EQUAL( 6, hash.getIndex( energy ).second );

         energy = 1.5;
         CHECK_EQUAL( 5147, hash.getHashIndex(energy) );
         CHECK_EQUAL( 6, hash.getBinLo( 5147 ) );
         CHECK_EQUAL( 6, hash.getIndex( energy ).first );
         CHECK_EQUAL( 7, hash.getIndex( energy ).second );

         energy = 20.0;
         CHECK_EQUAL( 5665, hash.getHashIndex(energy) );
         CHECK_EQUAL( 6, hash.getBinLo( 5665 ) );
         CHECK_CLOSE( 19.9854, hash.invHashFunction(5665), 1e-4 );
         CHECK_CLOSE( 20.0855, hash.invHashFunction(5666), 1e-4 );
         CHECK_EQUAL( 6, hash.getIndex( energy ).first );
         CHECK_EQUAL( 8, hash.getIndex( energy ).second );

         energy = 100.0;
         CHECK_EQUAL( 5987, hash.getHashIndex(energy) );
         CHECK_EQUAL( 7, hash.getBinLo( 5987 ) );
         CHECK_EQUAL( 7, hash.getIndex( energy ).first );
         CHECK_EQUAL( 8, hash.getIndex( energy ).second );

         energy = 1000.0;
         CHECK_EQUAL( 6447, hash.getHashIndex(energy) );
         CHECK_EQUAL( 7, hash.getBinLo( 6447 ) );
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
         CHECK_EQUAL( 19, hash.getHashIndex(energy) );
         CHECK_EQUAL( 0, hash.getBinLo( 19 ) );
         CHECK_EQUAL( 0, hash.getIndex( energy ).first );
         CHECK_EQUAL( 0, hash.getIndex( energy ).second );

         energy = 1.2e-11;
         CHECK_EQUAL( 36, hash.getHashIndex(energy) );
         CHECK_EQUAL( 0, hash.getBinLo( 36 ) );
         CHECK_EQUAL( 0, hash.getIndex( energy ).first );
         CHECK_EQUAL( 0, hash.getIndex( energy ).second );

         energy = 1.0e-10;
         CHECK_EQUAL( 460, hash.getHashIndex(energy) );
         CHECK_EQUAL( 0, hash.getBinLo( 360 ) );
         CHECK_EQUAL( 0, hash.getIndex( energy ).first );
         CHECK_EQUAL( 0, hash.getIndex( energy ).second );

         energy = 1.1e-9;
         CHECK_EQUAL( 940, hash.getHashIndex(energy) );
         CHECK_EQUAL( 0, hash.getBinLo( 940 ) );
         CHECK_EQUAL( 0, hash.getIndex( energy ).first );
         CHECK_EQUAL( 1, hash.getIndex( energy ).second );

         energy = 0.01;
         CHECK_EQUAL( 4144, hash.getHashIndex(energy) );
         CHECK_EQUAL( 0, hash.getBinLo( 4144 ) );
         CHECK_EQUAL( 0, hash.getIndex( energy ).first );
         CHECK_EQUAL( 1, hash.getIndex( energy ).second );

         energy = 0.15;
         CHECK_EQUAL( 4686, hash.getHashIndex(energy) );
         CHECK_EQUAL( 1, hash.getBinLo( 4686 ) );
         CHECK_EQUAL( 1, hash.getIndex( energy ).first );
         CHECK_EQUAL( 2, hash.getIndex( energy ).second );

     }

}

} // end namespace
