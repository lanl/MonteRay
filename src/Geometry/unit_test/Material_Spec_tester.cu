#include "MonteRay_MaterialSpec.hh"
#include <UnitTest++.h>

namespace Material_Spec_tester{
using namespace MonteRay;
typedef MonteRay_MaterialSpec MaterialSpec;

SUITE( Material_Spec_Tester ) {

    MaterialSpec One  ( 3, 3.1 );
    MaterialSpec Two  ( 2, 3.1 );
    MaterialSpec Three( 3, 3.1 );
	MaterialSpec Four ( 2, 2.1 );

	TEST( Operator_Tests ){

		// Test All the possible combination of comparators for MaterialSpec

	    //Check if a MaterialSpec equals a MaterialSpec.
	    CHECK( One == Three );
		CHECK( One != Four );
		CHECK( Two != Four );
		CHECK( Three != Four);

		//Check if a MaterialSpec.ID equals an int (id).
		CHECK( 3 == One ); //Example: id(3) equals One.ID(3)
		CHECK( One == 3 );
		CHECK( 1 != Two ); //Example: id(1) not equal to Two.ID(2)
		CHECK( Two != 1 );
	}
}
} // end namespace
