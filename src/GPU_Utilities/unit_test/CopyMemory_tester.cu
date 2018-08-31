#include <UnitTest++.h>

#include "MonteRayCopyMemory.hh"

SUITE( CopyMemory_tester ) {

	using namespace MonteRay;

	class testCopyClass : public CopyMemoryBase<testCopyClass> {

	public:
		using Base = CopyMemoryBase<testCopyClass>;
		testCopyClass() : CopyMemoryBase<testCopyClass>() {
			init();
		}
		~testCopyClass(){
			if( Base::isCudaIntermediate ) {
				// MonteRayDeviceFree( x );
			} else {
				// MonteRayHostFree( x, Base::isManagedMemory );
			}
		}

		std::string className(){ return std::string("testCopyClass");}

		void init() {
			A = 10;
			B = 20;
			C = 30;
		}

		void copyToGPU(void) {
			//std::cout << "Debug: testCopyClass::copyToGPU \n";
			Base::copyToGPU();
		}

		void copy(const testCopyClass* rhs) {
			if( debug ) {
				std::cout << "Debug: testCopyClass::operator= (const RayList_t<N>& rhs) \n";
			}

			if( isCudaIntermediate && rhs->isCudaIntermediate ) {
				throw std::runtime_error("RayList_t::operator= -- can NOT copy CUDA intermediate to CUDA intermediate.");
			}

			if( !isCudaIntermediate && !rhs->isCudaIntermediate ) {
				throw std::runtime_error("RayList_t::operator= -- can NOT copy CUDA non-intermediate to CUDA non-intermediate.");
			}

			A = rhs->A;
			B = rhs->B;
			C = rhs->C;
		}

		unsigned A;
		unsigned B;
		unsigned C;
	};

	TEST( CopyMemory_ctor ) {
		testCopyClass* test = new testCopyClass();

		CHECK_EQUAL( 40, sizeof(testCopyClass) );

		delete test;
	}

	TEST( CopyMemory_initialize ) {
		testCopyClass* test = new testCopyClass();

		CHECK_EQUAL( 10U, test->A );
		CHECK_EQUAL( 10U, test->intermediatePtr->A );
		CHECK_EQUAL( 20U, test->B );
		CHECK_EQUAL( 20U, test->intermediatePtr->B );

		delete test;
	}

	CUDA_CALLABLE_KERNEL void kernelSumAandB(testCopyClass* A, testCopyClass* B, testCopyClass* C) {
	    C->A += A->A + B->A;
	    C->B += A->B + B->B;
	    return;
	}

	TEST( CopyMemory_copyToGPU ) {
		testCopyClass* test1 = new testCopyClass();
		testCopyClass* test2 = new testCopyClass();
		testCopyClass* test3 = new testCopyClass();

		test1->A = 10.0;
		test1->B = 20.0;
		test2->A = 100.0;
		test2->B = 200.0;
		test3->A = 1000.0;
		test3->B = 2000.0;

#ifdef __CUDACC__
		test1->copyToGPU();
		test2->copyToGPU();
		test3->copyToGPU();
		kernelSumAandB<<<1,1>>>(test1->devicePtr,test2->devicePtr,test3->devicePtr);
		test3->copyToCPU();
#else
		kernelSumAandB(test1,test2,test3);
#endif

		CHECK_EQUAL(1110, test3->A);
		CHECK_EQUAL(2220, test3->B);

		delete test1;
		delete test2;
		delete test3;
	}
#if true
	class testClassWithArray : public CopyMemoryBase<testClassWithArray> {
	public:
		using Base = MonteRay::CopyMemoryBase<testClassWithArray> ;
		testClassWithArray(unsigned num = 1, double mult = 1.0) : CopyMemoryBase() {
			N = num;
			multiple = mult;
			elements = (gpuFloatType_t*) MONTERAYHOSTALLOC( N * sizeof( gpuFloatType_t ), false, std::string("testClassWithArray") );

			for( unsigned i=0; i<N; ++i) {
				elements[i] = 0.0;
			}
		}
		~testClassWithArray(){
			if( Base::isCudaIntermediate ) {
#ifdef __CUDACC__
				MonteRayDeviceFree( elements );
#endif
			} else {
				MonteRayHostFree( elements, Base::isManagedMemory );
			}
		}

		std::string className(){ return std::string("testClassWithArray");}

		void init() {
			multiple = 0.0;
			N = 0;
			elements = NULL;
		}

		gpuFloatType_t multiple;
		unsigned N;
		gpuFloatType_t* elements;

		void copy(const testClassWithArray* rhs) {
#ifdef __CUDACC__
			if( N != 0 && (N != rhs->N) ) {
				std::cout << "Error: testClassWithArray::copy -- can't change size after initialization.\n";
				std::cout << "Error: testClassWithArray::copy -- N = " << N << " \n";
				std::cout << "Error: testClassWithArray::copy -- rhs->N = " << rhs->N << " \n";
				std::cout << "Error: testClassWithArray::copy -- isCudaIntermediate = " << isCudaIntermediate << " \n";
				std::cout << "Error: testClassWithArray::copy -- rhs->isCudaIntermediate = " << rhs->isCudaIntermediate << " \n";
				throw std::runtime_error("testClassWithArray::copy -- can't change size after initialization.");
			}

			if( isCudaIntermediate ) {
				// host to device
				if( N == 0 ) {
					elements = (gpuFloatType_t*) MONTERAYDEVICEALLOC( rhs->N*sizeof(gpuFloatType_t), std::string("device - testClassWithArray::elements") );
				}
				MonteRayMemcpy( elements, rhs->elements, rhs->N*sizeof(gpuFloatType_t), cudaMemcpyHostToDevice );
			} else {
				// device to host
				MonteRayMemcpy( elements, rhs->elements, rhs->N*sizeof(gpuFloatType_t), cudaMemcpyDeviceToHost );
			}

			multiple = rhs->multiple;
			N = rhs->N;
#else
			throw std::runtime_error("testClassWithArray::copy -- can NOT copy between host and device without CUDA.");
#endif
		}
	};

CUDA_CALLABLE_KERNEL void kernelSumVectors2(testClassWithArray* A, testClassWithArray* B, testClassWithArray* C) {
    for( unsigned i=0; i<A->N; ++i) {
    	gpuFloatType_t elementA = A->elements[i] * A->multiple;
    	gpuFloatType_t elementB = B->elements[i] * B->multiple;
    	gpuFloatType_t elementC = elementA + elementB;
    	C->elements[i] = elementC;
    }
    C->N = A->N;
    C->multiple = 1.0;
    return;
}

TEST( add_vectors_w_copyToCPU ) {
	testClassWithArray* A = new testClassWithArray(4);
	testClassWithArray* B = new testClassWithArray(4);
	testClassWithArray* C = new testClassWithArray(4);

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

#ifdef __CUDACC__
	A->copyToGPU();
	B->copyToGPU();
	C->copyToGPU();

	kernelSumVectors2<<<1,1>>>(A->devicePtr,B->devicePtr,C->devicePtr);

	C->copyToCPU();
#else
	kernelSumVectors2(A,B,C);
#endif

	CHECK_EQUAL( 4,C->N);
	CHECK_CLOSE( 1.0, C->multiple, 1e-6);
	CHECK_CLOSE( 10.0*1.0+100.0*10.0, C->elements[0], 1e-6);
	CHECK_CLOSE( 2020.0, C->elements[1], 1e-6);
	CHECK_CLOSE( 3030.0, C->elements[2], 1e-6);
	CHECK_CLOSE( 4040.0, C->elements[3], 1e-6);

	delete A;
	delete B;
	delete C;
}

}
#endif
