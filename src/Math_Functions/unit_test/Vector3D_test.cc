#include <UnitTest++.h>

#include <iostream>
#include <sstream>
#include <cstring>

#include "MonteRayVector3D.hh"

#include "MonteRayMultiStream.hh"

namespace Vector3D_test{

using namespace std;
using namespace MonteRay;

SUITE( Test_Vector3D ) {
	double epsilon = 1.0e-7;
	TEST( DefaultCtor ) {
		//CHECK(false);
		Vector3D<double> temp;
		CHECK_CLOSE( temp[0], 0.0, epsilon );
		CHECK( temp[ 0 ] == temp[ 1 ] && temp[ 0 ] == temp[ 2 ]);
	}
#ifndef NDEBUG
	TEST( IndexingOutOfBounds ) {
		Vector3D<double> t1( 15.0 );
		CHECK_THROW( t1[ 5000 ], std::exception );
	}
#endif
	TEST( SingleValueCtor ) {
		Vector3D<int> temp( 1 );
		CHECK( temp[ 0 ] == 1 && temp[ 1 ] == 1 && temp[ 2 ] == 1 );
	}
	TEST( MultipleValueCtor ) {
		Vector3D<int> temp( 1, 10, 100 );
		CHECK( temp[ 0 ] == 1 && temp[ 1 ] == 10 && temp[ 2 ] == 100 );
	}
	TEST( ArrayInitialization ) {
		int SomeInts[] = { 0, 1, 2 };
		Vector3D<int> temp( SomeInts );
		CHECK( temp[0] == 0 );
		CHECK( temp[1] == 1 );
		CHECK( temp[2] == 2 );
	}
	TEST( CopyCtor ) {
		Vector3D<int> t1( 1 );
		Vector3D<int> t2( t1 );
		CHECK( t2 == t1 );
	}
	TEST( AssignmentOperator ) {
		Vector3D<int> t1( 1 );
		Vector3D<int> t2 = t1;
		CHECK( t1 == t2 );
	}
	TEST( VectorEquals ) {
		Vector3D<int> t1( 1 );
		Vector3D<int> t2( 1 );
		CHECK( t1 == t2 );
	}
	TEST( VectorNotEquals ) {
		Vector3D<int> t1( 1 );
		Vector3D<int> t2( 2 );
		CHECK( t1 != t2 );
	}
	TEST( VectorNotEquals_bySingleValue ) {
		Vector3D<int> t1( 0, 0, 1 );
		Vector3D<int> t2( 0, 0, 0 );
		CHECK( t1 != t2 );
	}
	TEST( Magnitude ) {
		Vector3D<double> t1( 1.0, 1.0, 1.0 );
		CHECK_CLOSE( sqrt( 3.0 ), t1.magnitude(), epsilon );
	}
	TEST( Normalize ) {
		Vector3D<double> t1( 1.0, 2.0, 3.0 );
		t1.normalize();
		CHECK_CLOSE( 3.0 / sqrt( 14.0 ), t1[ 2 ], epsilon );
		CHECK_CLOSE( 1.0, t1.magnitude(), epsilon );
	}
	TEST( MultiplicationBySingleValue ) {
		Vector3D<int> t1( 2 );
		CHECK( (t1*3)[0] == 6 );
	}
	TEST( VectorAdditionAssignment ) {
		Vector3D<int> t1( 2 ), t2( 4 );
		t1 += t2;
		CHECK( t1[0] == 6 );
	}
	TEST( VectorVectorAddition ) {
		Vector3D<double> t1( 1, 2, 3 ), t2( 3, 4, 5 );
		CHECK_CLOSE( 4.0, (t1+t2)[0], epsilon );
	}
	TEST( MultiplyAssignSingleValue ) {
		Vector3D<int> t1(2);
		t1 *= 10;
		CHECK( t1 == Vector3D<int>( 20 ) );
	}
	TEST( UpdateDistanceCalculation ) {
		Vector3D<double> Pos( 1, 2, 3 ), Direc( 2, 3, 5 );
		double distance = Direc.magnitude();
		Direc.normalize();
		Pos += Direc * distance;
		CHECK_CLOSE(3.0, Pos[0], epsilon );
	}
	TEST( DotProductCalculation ) {
		Vector3D<double> t1( 2 ), t2( 3 );
		CHECK_CLOSE( 18.0, dot(t1,t2), epsilon );
	}
	TEST( DotProductCalculation2 ) {
		Vector3D<double> t1( 1, 2, 3 ), t2( 3, 4, 5 );
		CHECK_CLOSE( 26.0, dot(t1,t2), epsilon );
	}
	TEST( CrossProduct1 ) {
		Vector3D<double> t1( 1, 0, 0 ), t2( 0, 1, 0 );
		Vector3D<double> result = cross(t1,t2);
 		CHECK_CLOSE( 0.0, result[0], epsilon );
 		CHECK_CLOSE( 0.0, result[1], epsilon );
 		CHECK_CLOSE( 1.0, result[2], epsilon );
	}
	TEST( CrossProduct2 ) {
		Vector3D<double> t1( 0, 1, 0 ), t2( 0, 0, 1 );
		Vector3D<double> result = cross(t1,t2);
 		CHECK_CLOSE( 1.0, result[0], epsilon );
 		CHECK_CLOSE( 0.0, result[1], epsilon );
 		CHECK_CLOSE( 0.0, result[2], epsilon );
	}
	TEST( CrossProduct3 ) {
		Vector3D<double> t1( 0, 0, 1 ), t2( 1, 0, 0 );
		Vector3D<double> result = cross(t1,t2);
 		CHECK_CLOSE( 0.0, result[0], epsilon );
 		CHECK_CLOSE( 1.0, result[1], epsilon );
 		CHECK_CLOSE( 0.0, result[2], epsilon );
	}
}

SUITE(Vector3D_TestOfMultiStream) {

    TEST(Something) {
        Vector3D<int> temp( 1, 10, 100 );
        CHECK( temp[ 0 ] == 1 && temp[ 1 ] == 10 && temp[ 2 ] == 100 );

        stringstream termOut;
        MultiStream out(termOut);
        out.addFile("Vector3d.out");

        out << "Debug: temp = *" << temp << "*" << endl;

        CHECK_EQUAL(string("Debug: temp = *( 1, 10, 100 )*\n"), termOut.str() );

        ifstream testFile( "Vector3d.out" );
        char buff[ 50 ] = { '\0' };
        testFile.getline( buff, 50 );
        // Note: getline removes the \n character by default
        CHECK_ARRAY_EQUAL( "Debug: temp = *( 1, 10, 100 )*", buff, strlen( buff ) );

    }
}

SUITE(Test_Vector3D_Math_Operations){
    TEST( getDistanceDirection_PosU ) {
        float expectedDistance = std::sqrt( 3.0f*3.0f );

        MonteRay::Vector3D<gpuRayFloat_t> pos(0.0, 0.0, 0.0);
        MonteRay::Vector3D<gpuRayFloat_t> pos2(3.0, 0.0, 0.0);

        auto distanceAndDirection = getDistanceDirection(pos, pos2);
        auto& distance = std::get<0>(distanceAndDirection);
        auto& dir = std::get<1>(distanceAndDirection);

        CHECK_CLOSE( expectedDistance, distance, 1e-6 );
        CHECK_CLOSE( 1.0, dir[0], 1e-6 );
    }

    TEST( getDistanceDirection_NegU ) {
        float expectedDistance = std::sqrt( 3.0f*3.0f );

        MonteRay::Vector3D<gpuRayFloat_t> pos(0.0, 0.0, 0.0);
        MonteRay::Vector3D<gpuRayFloat_t> pos2(-3.0, 0.0, 0.0);

        auto distanceAndDirection = getDistanceDirection(pos, pos2);
        auto& distance = std::get<0>(distanceAndDirection);
        auto& dir = std::get<1>(distanceAndDirection);
 
        CHECK_CLOSE( expectedDistance, distance, 1e-6 );
        CHECK_CLOSE( -1.0, dir[0], 1e-6 );
    }

    TEST( getDistanceDirection_PosV ) {
        float expectedDistance = std::sqrt( 3.0f*3.0f );

        MonteRay::Vector3D<gpuRayFloat_t> pos(0.0, 0.0, 0.0);
        MonteRay::Vector3D<gpuRayFloat_t> pos2(0.0, 3.0, 0.0);

        auto distanceAndDirection = getDistanceDirection(pos, pos2);
        auto& distance = std::get<0>(distanceAndDirection);
        auto& dir = std::get<1>(distanceAndDirection);

        CHECK_CLOSE( expectedDistance, distance, 1e-6 );
        CHECK_CLOSE( 1.0, dir[1], 1e-6 );
    }

    TEST( getDistanceDirection_NegV ) {
        float expectedDistance = std::sqrt( 3.0f*3.0f );

        MonteRay::Vector3D<gpuRayFloat_t> pos(0.0, 0.0, 0.0);
        MonteRay::Vector3D<gpuRayFloat_t> pos2(0.0, -3.0, 0.0);

        auto distanceAndDirection = getDistanceDirection(pos, pos2);
        auto& distance = std::get<0>(distanceAndDirection);
        auto& dir = std::get<1>(distanceAndDirection);

        CHECK_CLOSE( expectedDistance, distance, 1e-6 );
        CHECK_CLOSE( -1.0, dir[1], 1e-6 );
    }

    TEST( getDistanceDirection_PosW ) {
        float expectedDistance = std::sqrt( 3.0f*3.0f );

        MonteRay::Vector3D<gpuRayFloat_t> pos(0.0, 0.0, 0.0);
        MonteRay::Vector3D<gpuRayFloat_t> pos2(0.0, 0.0, 3.0);

        auto distanceAndDirection = getDistanceDirection(pos, pos2);
        auto& distance = std::get<0>(distanceAndDirection);
        auto& dir = std::get<1>(distanceAndDirection);

        CHECK_CLOSE( expectedDistance, distance, 1e-6 );
        CHECK_CLOSE( 1.0, dir[2], 1e-6 );
    }

    TEST( getDistanceDirection_NegW ) {
        float expectedDistance = std::sqrt( 3.0f*3.0f );

        MonteRay::Vector3D<gpuRayFloat_t> pos(0.0, 0.0, 0.0);
        MonteRay::Vector3D<gpuRayFloat_t> pos2(0.0, 0.0, -3.0);

        auto distanceAndDirection = getDistanceDirection(pos, pos2);
        auto& distance = std::get<0>(distanceAndDirection);
        auto& dir = std::get<1>(distanceAndDirection);

        CHECK_CLOSE( expectedDistance, distance, 1e-6 );
        CHECK_CLOSE( -1.0, dir[2], 1e-6 );
    }

    TEST( getDistanceDirection_PosUV ) {
        float expectedDistance = std::sqrt( (3.0f*3.0f)*2 );

        MonteRay::Vector3D<gpuRayFloat_t> pos(0.0, 0.0, 0.0);
        MonteRay::Vector3D<gpuRayFloat_t> pos2(3.0, 3.0, 0.0);

        auto distanceAndDirection = getDistanceDirection(pos, pos2);
        auto& distance = std::get<0>(distanceAndDirection);
        auto& dir = std::get<1>(distanceAndDirection);

        CHECK_CLOSE( expectedDistance, distance, 1e-6 );
        CHECK_CLOSE( 1.0/sqrt(2.0), dir[0], 1e-6 );
        CHECK_CLOSE( 1.0/sqrt(2.0), dir[1], 1e-6 );
    }
}

} // end namespace

