#include <UnitTest++.h>

#include "MonteRayConstants.hh"

namespace Constants_test{

using namespace std;
using namespace MonteRay;

SUITE( Test_Constants ) {
	TEST(inf){
		CHECK_EQUAL(inf, std::numeric_limits<double>::infinity() );
	}
	TEST(epsilon){
		CHECK_EQUAL(epsilon, std::numeric_limits<double>::epsilon() );
	}
}

} // end namespace
