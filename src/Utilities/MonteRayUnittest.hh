#ifndef MONTERAYUNITTEST_HH_
#define MONTERAYUNITTEST_HH_

#include <UnitTest++.h>

//The following headers used in the main.cc
#include <TestReporterStdout.h>

#include "MonteRayTestReporterOstream.hh"


int
RunAllTests(UnitTest::TestReporter& reporter, char const* suiteName=NULL, int const maxTestTimeInMs=0 ) {
    UnitTest::TestRunner runner( reporter );
    return runner.RunTestsIf( UnitTest::Test::GetTestList(), suiteName, UnitTest::True(), maxTestTimeInMs);
}

int
RunBySuites(int argc, char* argv[], UnitTest::TestReporter& reporter){

    int error = 0;
    for(int i = 1; i < argc; ++i) {
        error += RunAllTests(
                 reporter,
                 argv[i]);
    }
    return error;
}

int
RunTests(int argc, char* argv[], UnitTest::TestReporter& reporter) {

    bool UseSuites = (argc > 1);

    if(UseSuites) {
      return RunBySuites( argc, argv, reporter );
    }
    return RunAllTests(reporter);
}

#endif /* UNITTEST_HH_*/
