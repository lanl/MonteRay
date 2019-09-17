#include <UnitTest++.h>

SUITE( simple_Test ) {
    TEST( test ) {
        int i=5;
        CHECK_EQUAL( 5, i );
    }
}
