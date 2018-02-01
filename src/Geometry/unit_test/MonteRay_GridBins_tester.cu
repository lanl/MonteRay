#include <UnitTest++.h>

#include "MonteRayGridBins.hh"
#include "GPUSync.hh"

using namespace MonteRay;

SUITE( MonteRay_GridBins_Tester ) {

	TEST( ctor ) {
		CHECK(true);
        std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins() );
    }

	TEST( ctor_takes_min_and_max ) {
        std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins(-10, 10, 20 ) );
        CHECK_CLOSE( -10.0, pGridInfo->getMinVertex(), 1e-11 );
        CHECK_CLOSE( 10.0, pGridInfo->getMaxVertex(), 1e-11 );
        CHECK_CLOSE( 1.0, pGridInfo->delta, 1e-11 );
	}

	TEST( initialize ) {
        std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins() );
        pGridInfo->initialize( -10, 10, 20);
        CHECK_CLOSE( -10.0, pGridInfo->getMinVertex(), 1e-11 );
        CHECK_CLOSE( 10.0, pGridInfo->getMaxVertex(), 1e-11 );
        CHECK_CLOSE( 20.0, pGridInfo->getNumBins(), 1e-11 );
        CHECK_CLOSE( -10.0, pGridInfo->vertices[0], 1e-11);
        CHECK_CLOSE( -9.0, pGridInfo->vertices[1], 1e-11);
	}

	TEST( ctor_with_vector ) {
        std::vector<gpuFloatType_t> vertices = {
        		-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
        		  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10
        };

        std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins(vertices) );

        CHECK_CLOSE( -10.0, pGridInfo->getMinVertex(), 1e-11 );
        CHECK_CLOSE( 10.0, pGridInfo->getMaxVertex(), 1e-11 );
        CHECK_CLOSE( 20.0, pGridInfo->getNumBins(), 1e-11 );
        CHECK_CLOSE( -10.0, pGridInfo->vertices[0], 1e-11);
        CHECK_CLOSE( -9.0, pGridInfo->vertices[1], 1e-11);
	}


	template<typename T>
	class resultClass :  public CopyMemoryBase<resultClass<T>> {
		public:
				using Base = CopyMemoryBase<resultClass>;

				resultClass() : CopyMemoryBase<resultClass<T>>() {
					init();
				}

				~resultClass(){}

				std::string className(){ return std::string("resultClass");}

				void init() {
					v = T(0);
				}

				void copyToGPU(void) {
					//std::cout << "Debug: resultClass::copyToGPU \n";
					Base::copyToGPU();
				}

				void copyToCPU(void) {
					//std::cout << "Debug: resultClass::copyToCPU \n";
					Base::copyToCPU();
				}

				void copy(const resultClass* rhs) {
					if( this->debug ) {
						std::cout << "Debug: 1- resultClass::copy(const resultClass* rhs) \n";
					}

					if( this->isCudaIntermediate && rhs->isCudaIntermediate ) {
						throw std::runtime_error("resultClass::copy -- can NOT copy CUDA intermediate to CUDA intermediate.");
					}

					if( !this->isCudaIntermediate && !rhs->isCudaIntermediate ) {
						throw std::runtime_error("resultClass::copy -- can NOT copy CUDA non-intermediate to CUDA non-intermediate.");
					}

					v = rhs->v;
				}

				T v;
	};

	// kernal call
	CUDA_CALLABLE_KERNEL void kernelGetNumBins(MonteRayGridBins* pGridBins, resultClass<gpuFloatType_t>* pResult) {
		pResult->v = pGridBins->getNumBins();
		printf( "kernelGetNumBins -- value = %d\n",  pResult->v );
		return;
	}

	TEST( kernelGetNumVertices ) {
		 std::vector<gpuFloatType_t> vertices = {
		        		-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
		        		  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10
		        };

		MonteRayGridBins* pGridInfo = new MonteRayGridBins(vertices);
		resultClass<gpuFloatType_t>* pResult = new resultClass<gpuFloatType_t>();
//
#ifdef __CUDACC__
		pGridInfo->copyToGPU();
		pResult->copyToGPU();
		CHECK_EQUAL(0, pResult->v );

		GPUSync sync1;
		sync1.sync();
		kernelGetNumBins<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr);

		GPUSync sync2;
		sync2.sync();
		gpuErrchk( cudaPeekAtLastError() );

		pResult->copyToCPU();
#else
		kernelGetNumBins( pGridInfo, pResult);
#endif
		CHECK_EQUAL(20, pResult->v );

		delete pGridInfo;
		delete pResult;
	}


    TEST( getLinearIndex_NonequalSpacing_off_neg_side ) {
    	std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins(-10, 10, 20 ) );
        CHECK_EQUAL( -1, pGridInfo->getLinearIndex( -10.5 ) );
    }
    TEST( getLinearIndex_NonequalSpacing_off_pos_side ) {
    	std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins(-10, 10, 20 ) );
        CHECK_EQUAL( 20, pGridInfo->getLinearIndex( 10.5 ) );
    }
    TEST( getLinearIndex_NonequalSpacing_first_index ) {
    	std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins(-10, 10, 20 ) );
        CHECK_EQUAL( 0, pGridInfo->getLinearIndex( -9.5 ) );
    }
    TEST( getLinearIndex_NonequalSpacing_second_index ) {
    	std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins(-10, 10, 20 ) );
        CHECK_EQUAL( 1, pGridInfo->getLinearIndex( -8.5  ) );
    }
    TEST( getLinearIndex_NonequalSpacing_last_index ) {
    	std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins(-10, 10, 20 ) );
        CHECK_EQUAL( 19, pGridInfo->getLinearIndex( 9.5 ) );
    }

    TEST( isIndexOutside_neg_side ) {
    	std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins(-10, 10, 20 ) );
        CHECK_EQUAL( true, pGridInfo->isIndexOutside( -1 ) );
    }
    TEST( isIndexOutside_pos_side ) {
    	std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins(-10, 10, 20 ) );
        CHECK_EQUAL( true, pGridInfo->isIndexOutside( 20 ) );
    }
    TEST( isIndexOutside_false_start ) {
    	std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins(-10, 10, 20 ) );
        CHECK_EQUAL( false, pGridInfo->isIndexOutside( 0 ) );
    }
    TEST( isIndexOutside_false_1 ) {
    	std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins(-10, 10, 20 ) );
        CHECK_EQUAL( false, pGridInfo->isIndexOutside( 1 ) );
    }
    TEST( isIndexOutside_false_end ) {
    	std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins(-10, 10, 20 ) );
        CHECK_EQUAL( false, pGridInfo->isIndexOutside( 19 ) );
    }

    TEST( getRadialIndexFromR ) {
         std::vector<gpuFloatType_t> Rverts = { 1.0, 2.0, 3.0 };

         std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins( Rverts ) );
         pGridInfo->initialize( Rverts );
         pGridInfo->modifyForRadial();

         CHECK_EQUAL(   0, pGridInfo->getRadialIndexFromR( 0.5 ) );
         CHECK_EQUAL(   1, pGridInfo->getRadialIndexFromR( 1.5 ) );
         CHECK_EQUAL(   2, pGridInfo->getRadialIndexFromR( 2.5 ) );
         CHECK_EQUAL(   3, pGridInfo->getRadialIndexFromR( 3.5 ) );
         CHECK_EQUAL(   3, pGridInfo->getRadialIndexFromR( 30.5 ) );
     }

    TEST( getRadialIndexFromRSq ) {
        std::vector<gpuFloatType_t> Rverts = { 1.0, 2.0, 3.0 };

        std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins( Rverts ) );
        pGridInfo->initialize( Rverts );
        pGridInfo->modifyForRadial();

        CHECK_EQUAL(   0, pGridInfo->getRadialIndexFromRSq( 0.5*0.5 ) );
        CHECK_EQUAL(   1, pGridInfo->getRadialIndexFromRSq( 1.5*1.5 ) );
        CHECK_EQUAL(   2, pGridInfo->getRadialIndexFromRSq( 2.5*2.5 ) );
        CHECK_EQUAL(   3, pGridInfo->getRadialIndexFromRSq( 3.5*3.4 ) );
        CHECK_EQUAL(   3, pGridInfo->getRadialIndexFromRSq( 30.5*30.5 ) );
    }

	// kernal call
	CUDA_CALLABLE_KERNEL void kernelGetLinearIndex(MonteRayGridBins* pGridBins, resultClass<int>* pResult, gpuFloatType_t r) {
		pResult->v = pGridBins->getLinearIndex(r);
		//printf( "kernelGetNumBins -- value = %d\n",  pResult->v );
		return;
	}

	int launchKernelGetLinearIndex( gpuFloatType_t r ) {

		std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins(-10, 10, 20 ) );
	    std::unique_ptr<resultClass<int>> pResult = std::unique_ptr<resultClass<int>>( new resultClass<int>() );
#ifdef __CUDACC__
	    pGridInfo->copyToGPU();
		pResult->copyToGPU();
		CHECK_EQUAL(0, pResult->v );

		GPUSync sync1;
		sync1.sync();

		kernelGetLinearIndex<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr, r);

		GPUSync sync2;
		sync2.sync();
		gpuErrchk( cudaPeekAtLastError() );

		pResult->copyToCPU();
#else
		kernelGetLinearIndex( pGridInfo, pResult, r);
#endif
		return pResult->v;
	}

	TEST( kernelGetLinearIndex ) {
		CHECK_EQUAL(-1, launchKernelGetLinearIndex(-10.5) );
		CHECK_EQUAL( 20, launchKernelGetLinearIndex( 10.5) );
		CHECK_EQUAL( 0, launchKernelGetLinearIndex( -9.5) );
		CHECK_EQUAL( 1, launchKernelGetLinearIndex( -8.5 ) );
		CHECK_EQUAL( 19, launchKernelGetLinearIndex( 9.5 ) );
	}

	// kernal call
	CUDA_CALLABLE_KERNEL void isIndexOutside(MonteRayGridBins* pGridBins, resultClass<bool>* pResult, int i) {
		pResult->v = pGridBins->isIndexOutside(i);
		return;
	}

	bool launchKernelGetIsIndexOutside( int i ) {

		std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins(-10, 10, 20 ) );
	    std::unique_ptr<resultClass<bool>> pResult = std::unique_ptr<resultClass<bool>>( new resultClass<bool>() );
#ifdef __CUDACC__
	    pGridInfo->copyToGPU();
		pResult->copyToGPU();
		CHECK_EQUAL(0, pResult->v );

		GPUSync sync1;
		sync1.sync();

		isIndexOutside<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr, i);

		GPUSync sync2;
		sync2.sync();
		gpuErrchk( cudaPeekAtLastError() );

		pResult->copyToCPU();
#else
		isIndexOutside( pGridInfo, pResult, i);
#endif
		return pResult->v;
	}

	TEST( kernelIsIndexOutside ) {
        CHECK_EQUAL( true, launchKernelGetIsIndexOutside( -1 ) );
        CHECK_EQUAL( true, launchKernelGetIsIndexOutside( 20 ) );
        CHECK_EQUAL( false, launchKernelGetIsIndexOutside( 0 ) );
        CHECK_EQUAL( false, launchKernelGetIsIndexOutside( 1 ) );
        CHECK_EQUAL( false, launchKernelGetIsIndexOutside( 19 ) );
    }

	// kernal call
	CUDA_CALLABLE_KERNEL void kernelGetRadialIndexFromR(MonteRayGridBins* pGridBins, resultClass<int>* pResult, gpuFloatType_t r) {
		pResult->v = pGridBins->getRadialIndexFromR(r);
		return;
	}

	int launchKernelGetRadialIndexFromR( gpuFloatType_t r ) {
        std::vector<gpuFloatType_t> Rverts= { 1.0, 2.0, 3.0 };
		std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins() );
		pGridInfo->initialize( Rverts );
		pGridInfo->modifyForRadial();

	    std::unique_ptr<resultClass<int>> pResult = std::unique_ptr<resultClass<int>>( new resultClass<int>() );
#ifdef __CUDACC__
	    pGridInfo->copyToGPU();
		pResult->copyToGPU();
		CHECK_EQUAL(0, pResult->v );

		GPUSync sync1;
		sync1.sync();

		kernelGetRadialIndexFromR<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr, r);

		GPUSync sync2;
		sync2.sync();
		gpuErrchk( cudaPeekAtLastError() );

		pResult->copyToCPU();
#else
		kernelGetRadialIndexFromR( pGridInfo, pResult, r);
#endif
		return pResult->v;
	}

	TEST( kernelGetRadialIndexFromR ) {
        CHECK_EQUAL(   0, launchKernelGetRadialIndexFromR( 0.5 ) );
        CHECK_EQUAL(   1, launchKernelGetRadialIndexFromR( 1.5 ) );
        CHECK_EQUAL(   2, launchKernelGetRadialIndexFromR( 2.5 ) );
        CHECK_EQUAL(   3, launchKernelGetRadialIndexFromR( 3.5 ) );
        CHECK_EQUAL(   3, launchKernelGetRadialIndexFromR( 30.5 ) );
	}

	// kernal call
	CUDA_CALLABLE_KERNEL void kernelGetRadialIndexFromRSq(MonteRayGridBins* pGridBins, resultClass<int>* pResult, gpuFloatType_t rSq) {
		pResult->v = pGridBins->getRadialIndexFromRSq(rSq);
		return;
	}

	int launchKernelGetRadialIndexFromRSq( gpuFloatType_t rSq ) {
        std::vector<gpuFloatType_t> Rverts= { 1.0, 2.0, 3.0 };
		std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins() );
		pGridInfo->initialize( Rverts );
		pGridInfo->modifyForRadial();

	    std::unique_ptr<resultClass<int>> pResult = std::unique_ptr<resultClass<int>>( new resultClass<int>() );
#ifdef __CUDACC__
	    pGridInfo->copyToGPU();
		pResult->copyToGPU();
		CHECK_EQUAL(0, pResult->v );

		GPUSync sync1;
		sync1.sync();

		kernelGetRadialIndexFromRSq<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr, rSq);

		GPUSync sync2;
		sync2.sync();
		gpuErrchk( cudaPeekAtLastError() );

		pResult->copyToCPU();
#else
		kernelGetRadialIndexFromRSq( pGridInfo, pResult, rSq);
#endif
		return pResult->v;
	}

	TEST( kernelGetRadialIndexFromRSq ) {
        CHECK_EQUAL(   0, launchKernelGetRadialIndexFromRSq( 0.5*0.5 ) );
        CHECK_EQUAL(   1, launchKernelGetRadialIndexFromRSq( 1.5*1.5 ) );
        CHECK_EQUAL(   2, launchKernelGetRadialIndexFromRSq( 2.5*2.5 ) );
        CHECK_EQUAL(   3, launchKernelGetRadialIndexFromRSq( 3.5*3.5 ) );
        CHECK_EQUAL(   3, launchKernelGetRadialIndexFromRSq( 30.5*30.5 ) );
	}

}
