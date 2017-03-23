#include <UnitTest++.h>

#include "gpuAddTwoDoubles.hh"
#include "GPUUtilityFunctions.hh"

SUITE( GPUAtomicAdd_simple_tests ) {

    TEST( gpuAddTwoDoubles ) {
    	double A = 1.0;
    	double B = 2.0;
    	double C = MonteRay::gpuAddTwoDoubles(A,B);
        CHECK_CLOSE( 3.0, C, 1e-6 );
    }

    TEST( gpuAddTwoFloats ) {
    	float A = 1.0;
    	float B = 2.0;
    	float C = MonteRay::gpuAddTwoFloats(A,B);
        CHECK_CLOSE( 3.0f, C, 1e-6 );
    }
}
