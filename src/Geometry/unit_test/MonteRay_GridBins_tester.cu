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


	class resultClass :  public CopyMemoryBase<resultClass> {
		public:
				using Base = CopyMemoryBase<resultClass>;

				resultClass() : CopyMemoryBase<resultClass>() {
					init();
				}

				~resultClass(){}

				std::string className(){ return std::string("resultClass");}

				void init() {
					v = 0;
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
					if( debug ) {
						std::cout << "Debug: resultClass::copy(const resultClass* rhs) \n";
					}

					if( isCudaIntermediate && rhs->isCudaIntermediate ) {
						throw std::runtime_error("resultClass::copy -- can NOT copy CUDA intermediate to CUDA intermediate.");
					}

					if( !isCudaIntermediate && !rhs->isCudaIntermediate ) {
						throw std::runtime_error("resultClass::copy -- can NOT copy CUDA non-intermediate to CUDA non-intermediate.");
					}

					v = rhs->v;
				}

				unsigned v;
	};

	// kernal call
	CUDA_CALLABLE_KERNEL void kernelGetNumBins(MonteRayGridBins* pGridBins, resultClass* pResult) {
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
		resultClass* pResult = new resultClass();
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

    TEST( getLinearIndex_equalSpacing_off_neg_side ) {
    	std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins(-10, 10, 20 ) );
        CHECK_EQUAL( -1, pGridInfo->getLinearIndex( -10.5, true ) );
    }
    TEST( getLinearIndex_equalSpacing_off_pos_side ) {
    	std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins(-10, 10, 20 ) );
        CHECK_EQUAL( 20, pGridInfo->getLinearIndex( 10.5, true ) );
    }
    TEST( getLinearIndex_equalSpacing_first_index ) {
    	std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins(-10, 10, 20 ) );
        CHECK_EQUAL( 0, pGridInfo->getLinearIndex( -9.5, true ) );
    }
    TEST( getLinearIndex_equalSpacing_second_index ) {
    	std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins(-10, 10, 20 ) );
        CHECK_EQUAL( 1, pGridInfo->getLinearIndex( -8.5, true ) );
    }
    TEST( getLinearIndex_equalSpacing_last_index ) {
    	std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins(-10, 10, 20 ) );
        CHECK_EQUAL( 19, pGridInfo->getLinearIndex( 9.5, true ) );
    }

    TEST( getLinearIndex_NonequalSpacing_off_neg_side ) {
    	std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins(-10, 10, 20 ) );
        CHECK_EQUAL( -1, pGridInfo->getLinearIndex( -10.5, false ) );
    }
    TEST( getLinearIndex_NonequalSpacing_off_pos_side ) {
    	std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins(-10, 10, 20 ) );
        CHECK_EQUAL( 20, pGridInfo->getLinearIndex( 10.5, false ) );
    }
    TEST( getLinearIndex_NonequalSpacing_first_index ) {
    	std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins(-10, 10, 20 ) );
        CHECK_EQUAL( 0, pGridInfo->getLinearIndex( -9.5, false ) );
    }
    TEST( getLinearIndex_NonequalSpacing_second_index ) {
    	std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins(-10, 10, 20 ) );
        CHECK_EQUAL( 1, pGridInfo->getLinearIndex( -8.5, false ) );
    }
    TEST( getLinearIndex_NonequalSpacing_last_index ) {
    	std::unique_ptr<MonteRayGridBins> pGridInfo = std::unique_ptr<MonteRayGridBins>( new MonteRayGridBins(-10, 10, 20 ) );
        CHECK_EQUAL( 19, pGridInfo->getLinearIndex( 9.5, false ) );
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

}
