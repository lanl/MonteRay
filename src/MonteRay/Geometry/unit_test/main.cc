#include "MonteRay_unittest.hh"
#include <iostream>

int 
main( int argc, char* argv[] ) {

    std::ostream& out = std::cout;
    UnitTest::TestReporterOstream<std::ostream> reporter( out );
    
    return RunTests( argc, argv, reporter );
}
