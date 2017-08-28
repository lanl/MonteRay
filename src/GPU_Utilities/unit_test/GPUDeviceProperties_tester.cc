#include <UnitTest++.h>

#include <iostream>

#include "GPUUtilityFunctions.hh"
#include "gpuAddTwoDoubles.hh"
#include "GPUUtilityFunctions.hh"

SUITE( gpu_device_properties_simple_tests ) {
    TEST( getNumberOfGPUS ) {
    	int i=MonteRay::getNumberOfGPUS();
        std::cout << "MonteRay GPUDeviceProperties_tester -- number of GPUs = " << i << "\n";
        CHECK( i > 0 );
    }
}

