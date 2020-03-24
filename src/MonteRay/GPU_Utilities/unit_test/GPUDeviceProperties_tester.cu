#include <UnitTest++.h>

#include <iostream>

#include "GPUUtilityFunctions.hh"

SUITE( gpu_device_properties_simple_tests ) {
    TEST( getNumberOfGPUS ) {
    	int i=MonteRay::getNumberOfGPUS();
        std::cout << "Debug: MonteRay GPUDeviceProperties_tester -- number of GPUs = " << i << "\n";
#ifdef __CUDACC__
        CHECK( i > 0 );
#else
        CHECK( i == 0 );
#endif
    }
}

