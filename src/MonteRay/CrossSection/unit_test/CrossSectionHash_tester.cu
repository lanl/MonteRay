#include <UnitTest++.h>

#include <vector>

#ifdef __CUDACC__
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#endif

#include "CrossSectionHash.hh"

namespace CrossSectionHash_tester_namespace {

using namespace MonteRay;

SUITE( CrossSectionHash_tester ) {
    typedef CrossSectionHash_t<double>::NuclearData_t DoubleNuclearData_t;
    typedef std::vector<double> DoubleNuclearDataVec_t;
    typedef CrossSectionHash_t<double>::Index_t DoubleIndex_t;

    typedef CrossSectionHash_t<float>::NuclearData_t FloatNuclearData_t;
    typedef std::vector<float> FloatNuclearDataVec_t;
    typedef CrossSectionHash_t<float>::Index_t FloatIndex_t;

    TEST( ctor_w_double ) {
        DoubleNuclearDataVec_t testValues = { 1.0, 2.0};
        CrossSectionHash_t<double> hash(testValues);
        CHECK_EQUAL( 5964, hash.size() );
    }


    TEST( ctor_w_float ) {
        FloatNuclearDataVec_t testValues = { 1.0, 2.0};
        CrossSectionHash_t<float> hash(testValues);
        CHECK_EQUAL( 5964, hash.size() );
    }

    TEST( getHashIndex_double ) {
        DoubleNuclearDataVec_t testValues = { 1.0, 2.0};
        CrossSectionHash_t<double> hash(testValues);
        CHECK_EQUAL( 9.947598300641403e-12, hash.getMinValue() );
        CHECK_EQUAL( 126255, CrossSectionHash_t<double>::hashFunction( hash.getMinValue()) );
        CHECK_EQUAL( 126255, hash.getMinIndex() );
        CHECK_EQUAL( 0, hash.getHashIndex( hash.getMinValue() ) );
        CHECK_EQUAL( 9.947598300641403e-12, hash.invHashFunction( 0 ) );

        CHECK_EQUAL( 1000, hash.getMaxValue() );
        CHECK_EQUAL( 132218, CrossSectionHash_t<double>::hashFunction( hash.getMaxValue()) );
        CHECK_EQUAL( 132218, hash.getMaxIndex() );
        CHECK_EQUAL( 5963, hash.getHashIndex( hash.getMaxValue()) );
        CHECK_EQUAL( 1000, hash.invHashFunction( hash.getNumIndices() -1  ));

        CHECK_EQUAL( 1.000444171950221e-11, hash.invHashFunction( 1 ) );
        CHECK_CLOSE( 1.006128513836302e-11, hash.invHashFunction( 2 ), 1e-26);
        CHECK_CLOSE( 1.011812855722383e-11, hash.invHashFunction( 3 ), 1e-26 );

        CHECK_EQUAL( 47816, hash.bytesize() );
    }

    TEST( getHashIndex_float ) {
        using T = float;
        FloatNuclearDataVec_t testValues = { 1.0, 2.0};
        CrossSectionHash_t<T> hash(testValues);
        CHECK_EQUAL( 9.947598300641403e-12, hash.getMinValue() );

        CHECK_EQUAL( 11567, CrossSectionHash_t<T>::hashFunction( hash.getMinValue()) );
        CHECK_EQUAL( 11567, hash.getMinIndex() );
        CHECK_EQUAL( 0, hash.getHashIndex( hash.getMinValue()) );
        CHECK_EQUAL( 9.947598300641403e-12, hash.invHashFunction( 0 ) );

        CHECK_EQUAL( 1000, hash.getMaxValue() );
        CHECK_EQUAL( 17530, CrossSectionHash_t<T>::hashFunction( hash.getMaxValue()) );
        CHECK_EQUAL( 17530, hash.getMaxIndex() );
        CHECK_EQUAL( 5963, hash.getHashIndex( hash.getMaxValue()) );
        CHECK_EQUAL( 1000, hash.invHashFunction( hash.getNumIndices() -1  ));

        CHECK_EQUAL( 1.000444171950221e-11, hash.invHashFunction( 1 ) );
        CHECK_CLOSE( 1.006128513836302e-11, hash.invHashFunction( 2 ), 1e-26);
        CHECK_CLOSE( 1.011812855722383e-11, hash.invHashFunction( 3 ), 1e-26 );
    }

    TEST( test_all_bin_eneries_are_valid_double ) {
        using T = double;
        DoubleNuclearDataVec_t testValues = { 1.0, 2.0};
        CrossSectionHash_t<T> hash(testValues);

        for( unsigned i=0; i< hash.size(); ++i ){
            T value = hash.invHashFunction( i );
            CHECK_EQUAL( true, std::isnormal(value) );
        }
    }

    TEST( test_all_bin_eneries_are_valid_float ) {
        using T = float;
        FloatNuclearDataVec_t testValues = { 1.0, 2.0};
        CrossSectionHash_t<T> hash(testValues);

        for( unsigned i=0; i< hash.size(); ++i ){
            T value = hash.invHashFunction( i );
            CHECK_EQUAL( true, std::isnormal(value) );
        }
    }

    TEST( testLowerBin_double ) {
        using T = double;
        //                                        0      1      2           3      4    5    6
        DoubleNuclearDataVec_t testValues = { 1e-13, 5e-13, 1e-11, 1.0005e-11, 1e-10, 1.0, 1e6 };
        CrossSectionHash_t<T> hash(testValues);

        CHECK_EQUAL( 47816, hash.bytesize() );

        CHECK_EQUAL( 1, hash.getBinLo( 0 ) ); // 9.947598300641403e-12
        CHECK_EQUAL( 2, hash.getBinLo( 1 ) ); // 1.000444171950221e-11
        CHECK_EQUAL( 3, hash.getBinLo( 2 ) ); // 1.006128513836302e-11
        CHECK_EQUAL( 3, hash.getBinLo( 3 ) ); // 1.011812855722383e-11

        CHECK_EQUAL( 0, hash.getIndex( 1e-14 ).first );
        CHECK_EQUAL( 0, hash.getIndex( 1e-14 ).second );

        T energy = 1e-13;
        CHECK_EQUAL( 0, hash.getHashIndex(energy) );
        CHECK_EQUAL( 9.947598300641403e-12, hash.getMinValue() );
        CHECK_EQUAL( 0, hash.getIndex( 1e-13 ).first );
        CHECK_EQUAL( 0, hash.getIndex( 1e-13 ).second );

        energy = 1e-12;
        CHECK_EQUAL( 0, hash.getHashIndex(energy) );
        CHECK_EQUAL( 1, hash.getBinLo( 0 ) ); // 9.947598300641403e-12
        CHECK_EQUAL( 2, hash.getBinLo( 1 ) ); // 1.000444171950221e-11
        CHECK_EQUAL( 1, hash.getIndex( 1e-12 ).first);
        CHECK_EQUAL( 3, hash.getIndex( 1e-12 ).second);

        CHECK_EQUAL( 1, hash.getIndex( 1e-11 ).first);
        CHECK_EQUAL( 3, hash.getIndex( 1e-11 ).second);

        energy = 1.00051e-11;
        CHECK_EQUAL( 1, hash.getHashIndex(energy) );
        CHECK_CLOSE( 1.000444171950221e-11, hash.invHashFunction(1), 1e-15 );
        CHECK_CLOSE( 1.006128513836302e-11, hash.invHashFunction(2), 1e-15 );
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

        CHECK_EQUAL( 5963, hash.getHashIndex( 1e3 ) );
        CHECK_EQUAL( 5, hash.getIndex( 1e3 ).first );
        CHECK_EQUAL( 7, hash.getIndex( 1e3 ).second );

        CHECK_EQUAL( 5963, hash.getHashIndex( 1e4 ) );
        CHECK_EQUAL( 5, hash.getIndex( 1e4 ).first );
        CHECK_EQUAL( 7, hash.getIndex( 1e4 ).second );

        energy = 1e5;
        CHECK_EQUAL( 5963, hash.getHashIndex(energy) );
        CHECK_EQUAL( 5, hash.getBinLo( 5963 ) );
        CHECK_EQUAL( 5, hash.getIndex( energy ).first );
        CHECK_EQUAL( 7, hash.getIndex( energy ).second );

        CHECK_EQUAL( 5963, hash.getHashIndex( 1e6 ) );
        CHECK_EQUAL( 5, hash.getIndex( 1e6 ).first );
        CHECK_EQUAL( 7, hash.getIndex( 1e6 ).second );


        CHECK_EQUAL( 6, hash.getIndex( 1e7 ).first );
        CHECK_EQUAL( 7, hash.getIndex( 1e7 ).second );

    }

    TEST( testLowerBin_float ) {
        using T = float;
        //                                       0      1      2           3      4    5    6
        FloatNuclearDataVec_t testValues = { 1e-13, 5e-13, 1e-11, 1.0005e-11, 1e-10, 1.0, 1e6 };
        CrossSectionHash_t<T> hash(testValues);

        CHECK_EQUAL( 47800, hash.bytesize() );

        CHECK_EQUAL( 1, hash.getBinLo( 0 ) ); // 9.947598300641403e-12
        CHECK_EQUAL( 2, hash.getBinLo( 1 ) ); // 1.000444171950221e-11
        CHECK_EQUAL( 3, hash.getBinLo( 2 ) ); // 1.006128513836302e-11
        CHECK_EQUAL( 3, hash.getBinLo( 3 ) ); // 1.011812855722383e-11

        CHECK_EQUAL( 0, hash.getIndex( 1e-14 ).first );
        CHECK_EQUAL( 0, hash.getIndex( 1e-14 ).second );

        T energy = 1e-13;
        CHECK_EQUAL( 0, hash.getHashIndex(energy) );
        CHECK_EQUAL( 9.947598300641403e-12, hash.getMinValue() );
        CHECK_EQUAL( 0, hash.getIndex( 1e-13 ).first );
        CHECK_EQUAL( 0, hash.getIndex( 1e-13 ).second );

        energy = 1e-12;
        CHECK_EQUAL( 0, hash.getHashIndex(energy) );
        CHECK_EQUAL( 1, hash.getBinLo( 0 ) ); // 9.947598300641403e-12
        CHECK_EQUAL( 2, hash.getBinLo( 1 ) ); // 1.000444171950221e-11
        CHECK_EQUAL( 1, hash.getIndex( 1e-12 ).first);
        CHECK_EQUAL( 3, hash.getIndex( 1e-12 ).second);

        CHECK_EQUAL( 1, hash.getIndex( 1e-11 ).first);
        CHECK_EQUAL( 3, hash.getIndex( 1e-11 ).second);

        energy = 1.00051e-11;
        CHECK_EQUAL( 1, hash.getHashIndex(energy) );
        CHECK_CLOSE( 1.000444171950221e-11, hash.invHashFunction(1), 1e-15 );
        CHECK_CLOSE( 1.006128513836302e-11, hash.invHashFunction(2), 1e-15 );
        CHECK_EQUAL( 2, hash.getIndex( energy ).first );
        CHECK_EQUAL( 4, hash.getIndex( energy ).second );

        CHECK_EQUAL( 5, hash.getIndex( 1e2 ).first );
        CHECK_EQUAL( 6, hash.getIndex( 1e2 ).second );

        CHECK_EQUAL( 5963, hash.getHashIndex( 1e3 ) );
        CHECK_EQUAL( 5964, hash.size() );
        CHECK_EQUAL( 5, hash.getIndex( 1e3 ).first );
        CHECK_EQUAL( 7, hash.getIndex( 1e3 ).second );

        CHECK_EQUAL( 5963, hash.getHashIndex( 1e4 ) );
        CHECK_EQUAL( 5, hash.getIndex( 1e4 ).first );
        CHECK_EQUAL( 7, hash.getIndex( 1e4 ).second );

        CHECK_EQUAL( 5963, hash.getHashIndex(1e5) );
        CHECK_EQUAL( 5, hash.getBinLo( 5963 ) );
        CHECK_EQUAL( 5, hash.getIndex( 1e5 ).first );
        CHECK_EQUAL( 7, hash.getIndex( 1e5 ).second );

        CHECK_EQUAL( 5963, hash.getHashIndex( 1e6 ) );
        CHECK_EQUAL( 5, hash.getIndex( 1e6 ).first );
        CHECK_EQUAL( 7, hash.getIndex( 1e6 ).second );

        CHECK_EQUAL( 6, hash.getIndex( 1e7 ).first );
        CHECK_EQUAL( 7, hash.getIndex( 1e7 ).second );

    }

    TEST( test_bytesize_with_larger_array ) {
           using T = double;
           DoubleNuclearDataVec_t testValues = { 1e-13, 5e-13, 1e-11, 1.0005e-11, 1e-10, 1e-9, 1e-8, 1e-5, 1.0, 1e6 };
           CrossSectionHash_t<T> hash(testValues);

           // doesn't change based on the array size- fixed size
           CHECK_EQUAL( 47816, hash.bytesize() );
    }

    TEST( energies_stop_before_end_of_hashTable ) {
         using T = double;
         //                                        0      1      2           3      4   5    6   7
         DoubleNuclearDataVec_t testValues = { 1e-13, 5e-13, 1e-11, 1.0005e-11, 1e-10, .1, 1.0, 20 };
         CrossSectionHash_t<T> hash(testValues);

         double energy = .5;
         CHECK_EQUAL( 4561, hash.getHashIndex(energy) );
         CHECK_EQUAL( 5, hash.getBinLo( 4561 ) );
         CHECK_EQUAL( 5, hash.getIndex( energy ).first );
         CHECK_EQUAL( 6, hash.getIndex( energy ).second );

         energy = 1.5;
         CHECK_EQUAL( 4753, hash.getHashIndex(energy) );
         CHECK_EQUAL( 6, hash.getBinLo( 4753 ) );
         CHECK_EQUAL( 6, hash.getIndex( energy ).first );
         CHECK_EQUAL( 7, hash.getIndex( energy ).second );

         energy = 20.0;
         CHECK_EQUAL( 5233, hash.getHashIndex(energy) );
         CHECK_EQUAL( 7, hash.getBinLo( 5233 ) );
         CHECK_EQUAL( 7, hash.getIndex( energy ).first );
         CHECK_EQUAL( 8, hash.getIndex( energy ).second );

         energy = 100.0;
         CHECK_EQUAL( 5529, hash.getHashIndex(energy) );
         CHECK_EQUAL( 7, hash.getBinLo( 5529 ) );
         CHECK_EQUAL( 7, hash.getIndex( energy ).first );
         CHECK_EQUAL( 8, hash.getIndex( energy ).second );

         energy = 1000.0;
         CHECK_EQUAL( 5963, hash.getHashIndex(energy) );
         CHECK_EQUAL( 7, hash.getBinLo( 5963 ) );
         CHECK_EQUAL( 7, hash.getIndex( energy ).first );
         CHECK_EQUAL( 8, hash.getIndex( energy ).second );

     }

    TEST( energies_start_after_beginning_of_hashTable ) {
         using T = double;
         //                                        0     1    2   3
         DoubleNuclearDataVec_t testValues = {   1e-9, .1, 1.0, 20 };
         CrossSectionHash_t<T> hash(testValues);

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
         CHECK_EQUAL( 18, hash.getHashIndex(energy) );
         CHECK_EQUAL( 0, hash.getBinLo( 18 ) );
         CHECK_EQUAL( 0, hash.getIndex( energy ).first );
         CHECK_EQUAL( 0, hash.getIndex( energy ).second );

         energy = 1.2e-11;
         CHECK_EQUAL( 36, hash.getHashIndex(energy) );
         CHECK_EQUAL( 0, hash.getBinLo( 36 ) );
         CHECK_EQUAL( 0, hash.getIndex( energy ).first );
         CHECK_EQUAL( 0, hash.getIndex( energy ).second );

         energy = 1.0e-10;
         CHECK_EQUAL( 428, hash.getHashIndex(energy) );
         CHECK_EQUAL( 0, hash.getBinLo( 428 ) );
         CHECK_EQUAL( 0, hash.getIndex( energy ).first );
         CHECK_EQUAL( 0, hash.getIndex( energy ).second );

         energy = 1.1e-9;
         CHECK_EQUAL( 872, hash.getHashIndex(energy) );
         CHECK_EQUAL( 0, hash.getBinLo( 872 ) );
         CHECK_EQUAL( 0, hash.getIndex( energy ).first );
         CHECK_EQUAL( 1, hash.getIndex( energy ).second );

         energy = 0.01;
         CHECK_EQUAL( 3828, hash.getHashIndex(energy) );
         CHECK_EQUAL( 0, hash.getBinLo( 3828 ) );
         CHECK_EQUAL( 0, hash.getIndex( energy ).first );
         CHECK_EQUAL( 1, hash.getIndex( energy ).second );

         energy = 0.15;
         CHECK_EQUAL( 4330, hash.getHashIndex(energy) );
         CHECK_EQUAL( 1, hash.getBinLo( 4330 ) );
         CHECK_EQUAL( 1, hash.getIndex( energy ).first );
         CHECK_EQUAL( 2, hash.getIndex( energy ).second );

     }

}

} // end namespace
