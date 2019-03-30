#include <UnitTest++.h>

#include <vector>
#include <sstream>
#include <random>

#ifdef __CUDACC__
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#endif

#include "CrossSectionHash.hh"
#include "BinarySearch.hh"

#include "MonteRay_timer.hh"

namespace CrossSectionHash_tester_namespace {

using namespace MonteRay;

SUITE( CrossSectionHash_tester ) {
    typedef CrossSectionHash<double>::NuclearData_t DoubleNuclearData_t;
    typedef CrossSectionHash<double>::NuclearDataVec_t DoubleNuclearDataVec_t;
    typedef CrossSectionHash<double>::Index_t DoubleIndex_t;

    typedef CrossSectionHash<float>::NuclearData_t FloatNuclearData_t;
    typedef CrossSectionHash<float>::NuclearDataVec_t FloatNuclearDataVec_t;
    typedef CrossSectionHash<float>::Index_t FloatIndex_t;

    TEST( ctor_w_double ) {
        DoubleNuclearDataVec_t testValues;
        CrossSectionHash<double> hash(testValues);
        CHECK_EQUAL( 5964, hash.size() );
    }

    TEST( ctor_w_float ) {
        FloatNuclearDataVec_t testValues;
        CrossSectionHash<float> hash(testValues);
        CHECK_EQUAL( 5964, hash.size() );
    }

    TEST( getHashIndex_double ) {
        DoubleNuclearDataVec_t testValues;
        CrossSectionHash<double> hash(testValues);
        CHECK_EQUAL( 9.947598300641403e-12, hash.minValue );
        CHECK_EQUAL( 126255, CrossSectionHash<double>::hashFunction( hash.minValue) );
        CHECK_EQUAL( 126255, hash.minIndex );
        CHECK_EQUAL( 0, hash.getHashIndex( hash.minValue) );
        CHECK_EQUAL( 9.947598300641403e-12, hash.invHashFunction( 0 ) );

        CHECK_EQUAL( 1000, hash.maxValue );
        CHECK_EQUAL( 132218, CrossSectionHash<double>::hashFunction( hash.maxValue) );
        CHECK_EQUAL( 132218, hash.maxIndex );
        CHECK_EQUAL( 5963, hash.getHashIndex( hash.maxValue) );
        CHECK_EQUAL( 1000, hash.invHashFunction( hash.numIndices -1  ));

        CHECK_EQUAL( 1.000444171950221e-11, hash.invHashFunction( 1 ) );
        CHECK_CLOSE( 1.006128513836302e-11, hash.invHashFunction( 2 ), 1e-26);
        CHECK_CLOSE( 1.011812855722383e-11, hash.invHashFunction( 3 ), 1e-26 );
    }

    TEST( getHashIndex_float ) {
        using T = float;
        FloatNuclearDataVec_t testValues;
        CrossSectionHash<T> hash(testValues);
        CHECK_EQUAL( 9.947598300641403e-12, hash.minValue );

        CHECK_EQUAL( 11567, CrossSectionHash<T>::hashFunction( hash.minValue) );
        CHECK_EQUAL( 11567, hash.minIndex );
        CHECK_EQUAL( 0, hash.getHashIndex( hash.minValue) );
        CHECK_EQUAL( 9.947598300641403e-12, hash.invHashFunction( 0 ) );

        CHECK_EQUAL( 1000, hash.maxValue );
        CHECK_EQUAL( 17530, CrossSectionHash<T>::hashFunction( hash.maxValue) );
        CHECK_EQUAL( 17530, hash.maxIndex );
        CHECK_EQUAL( 5963, hash.getHashIndex( hash.maxValue) );
        CHECK_EQUAL( 1000, hash.invHashFunction( hash.numIndices -1  ));

        CHECK_EQUAL( 1.000444171950221e-11, hash.invHashFunction( 1 ) );
        CHECK_CLOSE( 1.006128513836302e-11, hash.invHashFunction( 2 ), 1e-26);
        CHECK_CLOSE( 1.011812855722383e-11, hash.invHashFunction( 3 ), 1e-26 );
    }

    TEST( test_all_bin_eneries_are_valid_double ) {
        using T = double;
        DoubleNuclearDataVec_t testValues;
        CrossSectionHash<T> hash(testValues);

        for( unsigned i=0; i< hash.size(); ++i ){
            T value = hash.invHashFunction( i );
            CHECK_EQUAL( true, std::isnormal(value) );
        }
    }

    TEST( test_all_bin_eneries_are_valid_float ) {
        using T = float;
        FloatNuclearDataVec_t testValues;
        CrossSectionHash<T> hash(testValues);

        for( unsigned i=0; i< hash.size(); ++i ){
            T value = hash.invHashFunction( i );
            CHECK_EQUAL( true, std::isnormal(value) );
        }
    }

    TEST( testLowerBin_double ) {
        using T = double;
        //                                        0      1      2           3      4    5    6
        DoubleNuclearDataVec_t testValues = { 1e-13, 5e-13, 1e-11, 1.0005e-11, 1e-10, 1.0, 1e6 };
        CrossSectionHash<T> hash(testValues);

        CHECK_EQUAL( 1, hash.getBinLo( 0 ) ); // 9.947598300641403e-12
        CHECK_EQUAL( 2, hash.getBinLo( 1 ) ); // 1.000444171950221e-11
        CHECK_EQUAL( 3, hash.getBinLo( 2 ) ); // 1.006128513836302e-11
        CHECK_EQUAL( 3, hash.getBinLo( 3 ) ); // 1.011812855722383e-11

        CHECK_EQUAL( 0, hash.getIndex( 1e-14 ).first );
        CHECK_EQUAL( 0, hash.getIndex( 1e-14 ).second );

        T energy = 1e-13;
        CHECK_EQUAL( 0, hash.getHashIndex(energy) );
        CHECK_EQUAL( 9.947598300641403e-12, hash.minValue );
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
        unsigned index =  LowerBoundIndex(testValues.data()+hash.getIndex( energy ).first,
                                          hash.getIndex( energy ).second-hash.getIndex( energy ).first,
                                          energy );
        CHECK_EQUAL( 3, index+hash.getIndex( energy ).first);

        CHECK_EQUAL( 5, hash.getIndex( 1e2 ).first );
        CHECK_EQUAL( 6, hash.getIndex( 1e2 ).second );

        energy = 1e5;
        CHECK_EQUAL( 5963, hash.getHashIndex(energy) );
        CHECK_EQUAL( 5, hash.getBinLo( 5963 ) );
        CHECK_EQUAL( 5, hash.getIndex( energy ).first );
        CHECK_EQUAL( 6, hash.getIndex( energy ).second );

        CHECK_EQUAL( 6, hash.getIndex( 1e7 ).first );
        CHECK_EQUAL( 6, hash.getIndex( 1e7 ).second );

        CHECK_EQUAL( 5963, hash.getHashIndex( 1e3 ) );
        CHECK_EQUAL( 5, hash.getIndex( 1e3 ).first );
        CHECK_EQUAL( 6, hash.getIndex( 1e3 ).second );

        CHECK_EQUAL( 5963, hash.getHashIndex( 1e4 ) );
        CHECK_EQUAL( 5, hash.getIndex( 1e4 ).first );
        CHECK_EQUAL( 6, hash.getIndex( 1e4 ).second );

        CHECK_EQUAL( 5963, hash.getHashIndex( 1e6 ) );
        CHECK_EQUAL( 5, hash.getIndex( 1e6 ).first );
        CHECK_EQUAL( 6, hash.getIndex( 1e6 ).second );
    }

    TEST( testLowerBin_float ) {
        using T = float;
        FloatNuclearDataVec_t testValues = { 1e-13, 5e-13, 1e-11, 1.0005e-11, 1e-10, 1.0, 1e6 };
        CrossSectionHash<T> hash(testValues);

        CHECK_EQUAL( 1, hash.getBinLo( 0 ) ); // 9.947598300641403e-12
        CHECK_EQUAL( 2, hash.getBinLo( 1 ) ); // 1.000444171950221e-11
        CHECK_EQUAL( 3, hash.getBinLo( 2 ) ); // 1.006128513836302e-11
        CHECK_EQUAL( 3, hash.getBinLo( 3 ) ); // 1.011812855722383e-11

        CHECK_EQUAL( 0, hash.getIndex( 1e-14 ).first );
        CHECK_EQUAL( 0, hash.getIndex( 1e-14 ).second );

        T energy = 1e-13;
        CHECK_EQUAL( 0, hash.getHashIndex(energy) );
        CHECK_EQUAL( 9.947598300641403e-12, hash.minValue );
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

        energy = 1e5;
        CHECK_EQUAL( 5963, hash.getHashIndex(energy) );
        CHECK_EQUAL( 5, hash.getBinLo( 5963 ) );
        CHECK_EQUAL( 5, hash.getIndex( energy ).first );
        CHECK_EQUAL( 6, hash.getIndex( energy ).second );

        CHECK_EQUAL( 6, hash.getIndex( 1e7 ).first );
        CHECK_EQUAL( 6, hash.getIndex( 1e7 ).second );

        CHECK_EQUAL( 5963, hash.getHashIndex( 1e3 ) );
        CHECK_EQUAL( 5, hash.getIndex( 1e3 ).first );
        CHECK_EQUAL( 6, hash.getIndex( 1e3 ).second );

        CHECK_EQUAL( 5963, hash.getHashIndex( 1e4 ) );
        CHECK_EQUAL( 5, hash.getIndex( 1e4 ).first );
        CHECK_EQUAL( 6, hash.getIndex( 1e4 ).second );

        CHECK_EQUAL( 5963, hash.getHashIndex( 1e6 ) );
        CHECK_EQUAL( 5, hash.getIndex( 1e6 ).first );
        CHECK_EQUAL( 6, hash.getIndex( 1e6 ).second );
    }

}

} // end namespace
