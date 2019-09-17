#include <UnitTest++.h>

#include "MonteRayMultiStream.hh"

#include <sstream>
#include <iomanip>
#include <cstring>

using namespace std;

SUITE( verifyMultiStream ) {
	using namespace MonteRay;

    TEST( FlushFail ) {
        char name[] = "FlushFailTest";

        MultiStream out;
        out.addFile( name );

        char msg[] = "hello world\n";
        out << msg;

        ifstream testFile( name );

        static const int bufferSize = 256;
        char buff[ bufferSize ];
        /// This should fail since the output file created previously hasn't been flushed yet.
        testFile.getline( buff, bufferSize );

        CHECK( strlen( buff ) == 0 );
        CHECK( testFile.eof() );

    }
    TEST( FlushPass ) {
        char name[] = "FlushPassTest";

        MultiStream out;
        out.addFile( name );

        char msg[] = "hello world\n";
        out << msg;
        out << flush;

        ifstream testFile( name );

        static const int bufferSize = 256;
        char buff[ bufferSize ] = { '\0' };
        testFile.getline( buff, bufferSize );

        CHECK( strlen( buff ) > 0 );
        CHECK( !testFile.eof() );
        // Note: getline removes the \n character by default
        CHECK_ARRAY_EQUAL( msg, buff, strlen( msg )-1 );

    }
    TEST( CoutFilesTest ) {
        stringstream termOut;
        MultiStream out(termOut);
        out.addFile("CoutFileTest1");
        out.addFile("CoutFileTest2");

        char msg[] = "hello world";
        out << msg << endl;

        char wnewline[] = "hello world\n";
        CHECK_EQUAL(string( wnewline ), termOut.str() );

        ifstream testFile( "CoutFileTest1" );

        static const int bufferSize = 256;
        char buff[ bufferSize ] = { '\0' };
        testFile.getline( buff, bufferSize );

        // Note: getline removes the \n character by default
        CHECK_ARRAY_EQUAL( msg, buff, strlen( msg ) );


        ifstream testFile2( "CoutFileTest2" );

        static const int bufferSize2 = 256;
        char buff2[ bufferSize2 ] = { '\0' };
        testFile2.getline( buff2, bufferSize2 );

        // Note: getline removes the \n character by default
        CHECK_ARRAY_EQUAL( msg, buff2, strlen( msg ) );

    }
    TEST( verify_resetScreen ) {
        stringstream testOut;
        MultiStream out(testOut);
        out << 1.234 << " world";
        CHECK_EQUAL( string("1.234 world"), testOut.str() );
        out.unsetScreen();
        out << 432.1 << " hello";
        CHECK_EQUAL( string("1.234 world"), testOut.str() );
    }
    TEST( RealStringChain ) {
        stringstream testOut;
        MultiStream out(testOut);
        out << 1.234 << " world";
        CHECK_EQUAL( string("1.234 world"), testOut.str() );
    }
    TEST( with_std_endl ) {
        stringstream testOut;
        MultiStream out(testOut);
        out << "Test Method endl";
        out << std::endl;
        CHECK_EQUAL(string("Test Method endl\n"), testOut.str() );
    }

    int sumOf( int a, int b ) { return a+b; }
    TEST( simpleOrderOfOperations ) {
        //Checking order of operations by doing the function first before
        //directing to ouput.
        stringstream testOut;
        MultiStream out( testOut );
        out << sumOf( 2, 3 );
        CHECK_EQUAL( string( "5" ), testOut.str() );
    }
    TEST( StringBuffer ) {
        std::stringstream testBuffer;
        MultiStream out( testBuffer );
        out << "String" << "Buffer" << "\n";
        CHECK_EQUAL( std::string("StringBuffer\n"), testBuffer.str() );
    }
    TEST( wEndl_StringBuffer ) {
        std::stringstream testBuffer;
        MultiStream out( testBuffer );
        out << "wEndl_" << "StringBuffer" << endl << endl;
        CHECK_EQUAL( std::string("wEndl_StringBuffer\n\n"), testBuffer.str() );
    }
    TEST( SpacingFormat ) {
        std::stringstream testBuffer;
        MultiStream out( testBuffer );
        out <<"*"<< setw(15) << "SpacingFormat" << endl;
        CHECK_EQUAL( std::string("*  SpacingFormat\n"), testBuffer.str() );
    }
    TEST( PrecisionFormat ) {
        std::stringstream testBuffer;
        MultiStream out( testBuffer );
        double value = 1.999108739487;
        out << setprecision(9) << value
            << " "
            << setprecision(6) << value;
        CHECK_EQUAL( std::string("1.99910874 1.99911"), testBuffer.str() );
    }
    TEST( ChangingFloatField ) {
        std::stringstream testBuffer;
        MultiStream out( testBuffer );
        double value = 19.99108739487;
        out << setiosflags( ios::scientific )<< setprecision(6) << value
        	<< " "
        	<< resetiosflags( ios::scientific )
        	<< setiosflags( ios::fixed )<< setprecision(6) << value;
        CHECK_EQUAL( std::string("1.999109e+01 19.991087"), testBuffer.str() );
    }

}
