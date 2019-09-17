#include <UnitTest++.h>

#include <iostream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <vector>

#include "LinearSearch.hh"

namespace LinearSearch_test{

using namespace MonteRay;

SUITE( UpperBound ) {
    double epsilon = 1.0e-7;
    TEST( int_values ) {
        std::vector<int> values = { 10, 10, 10, 20, 20, 20, 30, 30 };
        unsigned ref_index = std::distance( values.begin(), std::upper_bound(values.begin(), values.end(), 20) );
        CHECK_EQUAL(6, ref_index );
        unsigned index = UpperBoundIndexLinear( values.data(), 0, values.size()-1, 20 );
        CHECK_EQUAL(ref_index, index);
    }

    TEST( low_side ) {
        std::vector<int> values = { 10, 10, 10, 20, 20, 20, 30, 30 };
        unsigned ref_index = std::distance( values.begin(), std::upper_bound(values.begin(), values.end(), 0) );
        CHECK_EQUAL(0, ref_index );
        unsigned index = UpperBoundIndexLinear( values.data(), 0, values.size()-1, 0 );
        CHECK_EQUAL(ref_index, index);
    }

    TEST( high_side ) {
        std::vector<int> values = { 10, 10, 10, 20, 20, 20, 30, 30 };
        unsigned index = UpperBoundIndexLinear( values.data(), 0, values.size()-1, 100 );
        unsigned ref_index = std::distance( values.begin(), std::upper_bound(values.begin(), values.end(), 100) );
        CHECK_EQUAL(8, ref_index );
        CHECK_EQUAL(ref_index, index);
    }

    TEST( float_values ) {
        std::vector<float> values = { 0.0, 0.5, 1.0, 2.0, 3.0, 30.0 };
        float value = 0.25;
        unsigned ref_index = std::distance( values.begin(), std::upper_bound(values.begin(), values.end(), value) );
        CHECK_EQUAL(1, ref_index );
        unsigned index = UpperBoundIndexLinear( values.data(), 0, values.size()-1, value );
        CHECK_EQUAL(ref_index, index);
    }

    TEST( float_value_low_side ) {
        std::vector<float> values = { 0.0, 0.5, 1.0, 2.0, 3.0, 30.0 };
        float value = -.25;
        unsigned ref_index = std::distance( values.begin(), std::upper_bound(values.begin(), values.end(), value) );
        CHECK_EQUAL(0, ref_index );
        unsigned index = UpperBoundIndexLinear( values.data(), 0, values.size()-1, value );
        CHECK_EQUAL(ref_index, index);
    }

    TEST( float_value_low_side_equal ) {
        std::vector<float> values = { 0.0, 0.5, 1.0, 2.0, 3.0, 30.0 };
        float value = 0.0;
        unsigned ref_index = std::distance( values.begin(), std::upper_bound(values.begin(), values.end(), value) );
        CHECK_EQUAL(1, ref_index );
        unsigned index = UpperBoundIndexLinear( values.data(), 0, values.size()-1, value );
        CHECK_EQUAL(ref_index, index);
    }

    TEST( float_value_high_side ) {
        std::vector<float> values = { 0.0, 0.5, 1.0, 2.0, 3.0, 30.0 };
        float value = 35.0;
        unsigned ref_index = std::distance( values.begin(), std::upper_bound(values.begin(), values.end(), value) );
        CHECK_EQUAL(6, ref_index );
        unsigned index = UpperBoundIndexLinear( values.data(), 0, values.size()-1, value );
        CHECK_EQUAL(ref_index, index);
    }

    TEST( float_value_high_side_equal ) {
         std::vector<float> values = { 0.0, 0.5, 1.0, 2.0, 3.0, 30.0 };
         float value = 30.0;
         unsigned ref_index = std::distance( values.begin(), std::upper_bound(values.begin(), values.end(), value) );
         CHECK_EQUAL(6, ref_index );
         unsigned index = UpperBoundIndexLinear( values.data(), 0, values.size()-1, value );
         CHECK_EQUAL(ref_index, index);
     }

    TEST( float_equal ) {
        std::vector<float> values = { 0.0, 0.5, 1.0, 2.0, 3.0, 30.0 };
        float value = 1.0;
        unsigned ref_index = std::distance( values.begin(), std::upper_bound(values.begin(), values.end(), value) );
        CHECK_EQUAL(3, ref_index );
        unsigned index = UpperBoundIndexLinear( values.data(), 0, values.size()-1, value );
        CHECK_EQUAL(ref_index, index);
    }
}

SUITE( LowerBound ) {
    double epsilon = 1.0e-7;
    TEST( int_values ) {
        std::vector<int> values = { 10, 20, 30, 40, 50 };
        unsigned ref_index = std::distance( values.begin(), std::upper_bound(values.begin(), values.end(), 20) - 1 );
        CHECK_EQUAL(1, ref_index );
        unsigned index = LowerBoundIndexLinear( values.data(), 0, values.size()-1, 20 );
        CHECK_EQUAL(ref_index, index);
    }

    TEST( low_side ) {
        std::vector<int> values = { 10, 20, 30, 40, 50 };
        unsigned ref_index = std::distance( values.begin(), std::upper_bound(values.begin(), values.end(), 0) - 1 );
        unsigned index = LowerBoundIndexLinear( values.data(), 0, values.size()-1, 0 );
        CHECK_EQUAL(0, index);
    }

    TEST( high_side ) {
        std::vector<int> values = { 10, 20, 30, 40, 50 };
        unsigned index = LowerBoundIndexLinear( values.data(), 0, values.size()-1, 100 );
        unsigned ref_index = std::distance( values.begin(), std::upper_bound(values.begin(), values.end(), 100) - 1 );
        CHECK_EQUAL(4, ref_index );
        CHECK_EQUAL(ref_index, index);
    }

    TEST( float_values ) {
        std::vector<float> values = { 0.0, 0.5, 1.0, 2.0, 3.0, 30.0 };
        float value = 0.25;
        unsigned ref_index = std::distance( values.begin(), std::upper_bound(values.begin(), values.end(), value) - 1 );
        CHECK_EQUAL(0, ref_index );
        unsigned index = LowerBoundIndexLinear( values.data(), 0, values.size()-1, value );
        CHECK_EQUAL(ref_index, index);
    }

    TEST( float_equal ) {
        std::vector<float> values = { 0.0, 0.5, 1.0, 2.0, 3.0, 30.0 };
        float value = 1.0;
        unsigned ref_index = std::distance( values.begin(), std::upper_bound(values.begin(), values.end(), value) - 1);
        CHECK_EQUAL(2, ref_index );
        unsigned index = LowerBoundIndexLinear( values.data(), 0, values.size()-1, value );
        CHECK_EQUAL(ref_index, index);
    }

    TEST( float_value_low_side ) {
        std::vector<float> values = { 0.0, 0.5, 1.0, 2.0, 3.0, 30.0 };
        float value = -.25;
        unsigned index = LowerBoundIndexLinear( values.data(), 0, values.size()-1, value );
        CHECK_EQUAL(0, index);
    }

    TEST( float_value_high_side ) {
        std::vector<float> values = { 0.0, 0.5, 1.0, 2.0, 3.0, 30.0 };
        float value = 35.0;
        unsigned ref_index = std::distance( values.begin(), std::lower_bound(values.begin(), values.end(), value) - 1 );
        CHECK_EQUAL(5, ref_index );
        unsigned index = LowerBoundIndexLinear( values.data(), 0, values.size()-1, value );
        CHECK_EQUAL(ref_index, index);
    }
}

} // end namespace

