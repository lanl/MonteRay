#include <UnitTest++.h>

#include "ManagedMemory_test_helper.hh"

SUITE( ManagedMemory_tester ) {
	using namespace MonteRay;

	TEST( ManagedMemory_ctor ) {
		testClass test;
		//CHECK(false);

		CHECK_EQUAL( 16, sizeof(testClass) );
	}

	TEST( ManagedMemory_ctor_with_new ) {
		testClass* test = new testClass(4);
		CHECK_EQUAL(4, test->N);
	}

	TEST_FIXTURE(ManagedMemoryTestHelper, add_vectors ) {
		testClass* A = new testClass(4);
		testClass* B = new testClass(4);
		testClass* C = new testClass(4);

		A->multiple = 10.0;
		A->elements[0] = 1.0;
		A->elements[1] = 2.0;
		A->elements[2] = 3.0;
		A->elements[3] = 4.0;

		B->multiple = 100.0;
		B->elements[0] = 10.0;
		B->elements[1] = 20.0;
		B->elements[2] = 30.0;
		B->elements[3] = 40.0;

		launchSumVectors(A,B,C);

		CHECK_EQUAL( 4,C->N);
		CHECK_CLOSE( 1.0, C->multiple, 1e-6);
		CHECK_CLOSE( 10.0*1.0+100.0*10.0, C->elements[0], 1e-6);
		CHECK_CLOSE( 2020.0, C->elements[1], 1e-6);
		CHECK_CLOSE( 3030.0, C->elements[2], 1e-6);
		CHECK_CLOSE( 4040.0, C->elements[3], 1e-6);
	}

	TEST_FIXTURE(ManagedMemoryTestHelper, add_vectors_w_copyToCPU ) {
		testClass* A = new testClass(4);
		testClass* B = new testClass(4);
		testClass* C = new testClass(4);

		A->multiple = 10.0;
		A->elements[0] = 1.0;
		A->elements[1] = 2.0;
		A->elements[2] = 3.0;
		A->elements[3] = 4.0;

		B->multiple = 100.0;
		B->elements[0] = 10.0;
		B->elements[1] = 20.0;
		B->elements[2] = 30.0;
		B->elements[3] = 40.0;

		A->copyToGPU();
		B->copyToGPU();
		C->copyToGPU();

		launchSumVectors(A,B,C);

		C->copyToCPU();

		CHECK_EQUAL( 4,C->N);
		CHECK_CLOSE( 1.0, C->multiple, 1e-6);
		CHECK_CLOSE( 10.0*1.0+100.0*10.0, C->elements[0], 1e-6);
		CHECK_CLOSE( 2020.0, C->elements[1], 1e-6);
		CHECK_CLOSE( 3030.0, C->elements[2], 1e-6);
		CHECK_CLOSE( 4040.0, C->elements[3], 1e-6);
	}

}
