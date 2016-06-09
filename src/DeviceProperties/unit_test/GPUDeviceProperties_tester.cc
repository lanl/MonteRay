#include <UnitTest++.h>

#include "gpuDeviceProperties.hh"
#include "gpuAddTwoDoubles.hh"

SUITE( gpu_device_properties_simple_tests ) {
    TEST( getNumberOfGPUS ) {
    	int i=MonteRay::getNumberOfGPUS();
        CHECK_EQUAL( 1, i );
    }

    TEST( gpuAddTwoDoubles ) {
    	double A = 1.0;
    	double B = 2.0;
    	double C = MonteRay::gpuAddTwoDoubles(A,B);
        CHECK_CLOSE( 3.0, C, 1e-6 );
    }
}
