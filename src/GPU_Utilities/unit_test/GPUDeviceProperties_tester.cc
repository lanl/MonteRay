#include <UnitTest++.h>

#include "gpuDeviceProperties.hh"
#include "gpuAddTwoDoubles.hh"
#include "GPUUtilityFunctions.hh"

SUITE( gpu_device_properties_simple_tests ) {
    TEST( getNumberOfGPUS ) {
    	int i=MonteRay::getNumberOfGPUS();
        CHECK( i > 0 );
    }
}
