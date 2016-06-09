#ifndef UNITTEST_TESTREPORTER_OSTREAM_H
#define UNITTEST_TESTREPORTER_OSTREAM_H

#include "TestReporter.h"

namespace UnitTest {

/// A templated class for use with UnitTest++ TestReporting. \n

/// This templated class allows for creation of a TestReporter object that handles\n a type meant to be an ostream. \n
/// This class is based on UnitTest::TestReporterOutput class in UnitTest++. \n
///
template <typename T>
class TestReporterOstream : public TestReporter
{
private:
    T& out;
    virtual void ReportTestStart( TestDetails const& ) {}
    virtual void ReportTestFinish(TestDetails const&, float ) {}

    virtual void ReportFailure(TestDetails const& details, char const* failure) {
        out << details.filename << ":"<<details.lineNumber<<": error: Failure in "<<details.testName<<": "<<failure<<"\n";
    }
    virtual void ReportSummary(int totalTestCount, int failedTestCount, int failureCount, float secondsElapsed) {
        if( failureCount > 0 ) {
            out << "FAILURE: "<<failedTestCount<<" out of "<<totalTestCount<<" tests failed ("<<failureCount<<" failures).\n";
        } else {
            out << "Success: "<<totalTestCount<<" tests passed.\n";
        }
        out <<"Test time "<<secondsElapsed<<" seconds.\n";
    }

public:
    TestReporterOstream( T& os ) : out( os ) {}
};

}

#endif 
