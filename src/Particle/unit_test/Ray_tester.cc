#include <UnitTest++.h>

#include <iostream>

#include "MonteRayDefinitions.hh"
#include "GPUUtilityFunctions.hh"

#include "Ray.hh"

SUITE( Ray_simple_tests ) {

	using namespace MonteRay;

	TEST( ray_ctor ) {
		Ray_t<> ray;
		// std::cout << "Debug: running Ray_tester.cc -- ctor test\n";
		CHECK(true);
	}

	TEST( getN ) {
		Ray_t<> ray;
		CHECK_EQUAL( 1, ray.getN());
	}

	TEST( ParticleRay_t_getN ) {
		ParticleRay_t ray;
		CHECK_EQUAL( 1, ray.getN());
	}

	TEST( PointDetRay_t_getN ) {
		PointDetRay_t ray;
		CHECK_EQUAL( 3, ray.getN());
	}

}


