#include "MonteRay_unittest.hh"

int
main( int argc, char* argv[] ) {
    UnitTest::TestReporterStdout reporter;
    return RunTests( argc, argv, reporter );
}
