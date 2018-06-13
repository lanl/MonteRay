#include <UnitTest++.h>

#include <memory>
#include <vector>
#include <array>

#include "MonteRay_SphericalGrid.hh"
#include "MonteRay_SpatialGrid.hh"
#include "GPUSync.hh"
#include "MonteRayVector3D.hh"
#include "MonteRayConstants.hh"

using namespace MonteRay;

namespace MonteRay_SphericalGrid_on_GPU_tester{

#if true

SUITE( MonteRay_SphericalGrid_GPU_basic_tests ) {
	using Grid_t = MonteRay_SphericalGrid;
	using GridBins_t = MonteRay_GridBins;
	using GridBins_t = Grid_t::GridBins_t;
	using pGridInfo_t = GridBins_t*;
	using pArrayOfpGridInfo_t = Grid_t::pArrayOfpGridInfo_t;

    typedef MonteRay::Vector3D<gpuRayFloat_t> Position_t;

    class gridTestData {
    public:
        enum coord {R,DIM};
        gridTestData(){
            std::vector<gpuRayFloat_t> vertices = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

            pGridInfo[0] = new GridBins_t();

            pGridInfo[0]->initialize( vertices );

        }
        ~gridTestData(){
        	delete pGridInfo[0];
        }

        pArrayOfpGridInfo_t pGridInfo;
    };

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
	CUDA_CALLABLE_KERNEL void kernelSphericalGridGetNumBins(Grid_t** pGrid, resultClass<unsigned>* pResult, unsigned d) {
		pResult->v = (*pGrid)->getNumBins(d);
	}

    TEST( getNumBins_on_GPU ) {
        enum coord {R,DIM};
    	gridTestData data;

    	resultClass<unsigned>* pResult = new resultClass<unsigned>();
    	pResult->copyToGPU();

    	std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(1,data.pGridInfo));
    	pGrid->copyToGPU();


//    	Grid_t** devicePtr;
//    	data.pGridInfo[X]->copyToGPU();
//    	data.pGridInfo[Y]->copyToGPU();
//    	data.pGridInfo[Z]->copyToGPU();
//    	createDeviceInstance<<<1,1>>>( devicePtr, data.pGridInfo[X]->devicePtr, data.pGridInfo[Y]->devicePtr, data.pGridInfo[Z]->devicePtr );

    	GPUSync sync1;
		sync1.sync();

    	gpuErrchk( cudaPeekAtLastError() );

    	//printf( "Debug: devicePtr = %d\n", devicePtr );

    	kernelSphericalGridGetNumBins<<<1,1>>>( pGrid->ptrDevicePtr, pResult->devicePtr, 0);
    	gpuErrchk( cudaPeekAtLastError() );
    	pResult->copyToCPU();
    	CHECK_EQUAL( 10, pResult->v );
    	pResult->v = 0;

		kernelSphericalGridGetNumBins<<<1,1>>>( pGrid->ptrDevicePtr, pResult->devicePtr, 1);
		gpuErrchk( cudaPeekAtLastError() );
		pResult->copyToCPU();
    	CHECK_EQUAL( 0, pResult->v );
    	pResult->v = 0;

		kernelSphericalGridGetNumBins<<<1,1>>>( pGrid->ptrDevicePtr, pResult->devicePtr, 2);
		gpuErrchk( cudaPeekAtLastError() );
		pResult->copyToCPU();
    	CHECK_EQUAL( 0, pResult->v );

    	delete pResult;
    }

//	// kernal call
	CUDA_CALLABLE_KERNEL void kernelSphericalGridGetIndex(Grid_t** pGrid, resultClass<unsigned>* pResult, Position_t pos) {
		//printf("Debug: kernelSphericalGridGetIndex -- calling pGrid->getIndex(pos)\n");
		unsigned index = (*pGrid)->getIndex(pos);
		pResult->v = index;
	}

    TEST( getIndex ) {
        gridTestData data;
        std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(1,data.pGridInfo));
        pGrid->copyToGPU();

    	resultClass<unsigned>* pResult = new resultClass<unsigned>();
    	pResult->copyToGPU();

        Position_t pos1( -0.5, -0.5, -0.5 );
        Position_t pos2( -1.5,  0.0,  0.0 );
        Position_t pos3(  2.5,  0.0,  0.0 );
        Position_t pos4(  0.0, -3.5,  0.0 );
        Position_t pos5(  0.0,  4.5,  0.0 );
        Position_t pos6(  0.0,  0.0, -5.5 );
        Position_t pos7(  0.0,  0.0,  6.5 );
        Position_t pos8(  5.5,  5.5,  5.5 );
        Position_t pos9( 10.0, 10.0, 10.0 );

        kernelSphericalGridGetIndex<<<1,1>>>( pGrid->ptrDevicePtr, pResult->devicePtr, pos1);
		gpuErrchk( cudaPeekAtLastError() );
		pResult->copyToCPU();
    	CHECK_EQUAL( 0, pResult->v );
    	pResult->v = 0;

        kernelSphericalGridGetIndex<<<1,1>>>( pGrid->ptrDevicePtr, pResult->devicePtr, pos2);
		gpuErrchk( cudaPeekAtLastError() );
		pResult->copyToCPU();
    	CHECK_EQUAL( 1, pResult->v );
    	pResult->v = 0;

        kernelSphericalGridGetIndex<<<1,1>>>( pGrid->ptrDevicePtr, pResult->devicePtr, pos3);
		gpuErrchk( cudaPeekAtLastError() );
		pResult->copyToCPU();
    	CHECK_EQUAL( 2, pResult->v );
    	pResult->v = 0;

        kernelSphericalGridGetIndex<<<1,1>>>( pGrid->ptrDevicePtr, pResult->devicePtr, pos4);
		gpuErrchk( cudaPeekAtLastError() );
		pResult->copyToCPU();
    	CHECK_EQUAL( 3, pResult->v );
    	pResult->v = 0;

        kernelSphericalGridGetIndex<<<1,1>>>( pGrid->ptrDevicePtr, pResult->devicePtr, pos5);
		gpuErrchk( cudaPeekAtLastError() );
		pResult->copyToCPU();
    	CHECK_EQUAL( 4, pResult->v );
    	pResult->v = 0;

        kernelSphericalGridGetIndex<<<1,1>>>( pGrid->ptrDevicePtr, pResult->devicePtr, pos6);
		gpuErrchk( cudaPeekAtLastError() );
		pResult->copyToCPU();
    	CHECK_EQUAL( 5, pResult->v );
    	pResult->v = 0;

        kernelSphericalGridGetIndex<<<1,1>>>( pGrid->ptrDevicePtr, pResult->devicePtr, pos7);
		gpuErrchk( cudaPeekAtLastError() );
		pResult->copyToCPU();
    	CHECK_EQUAL( 6, pResult->v );
    	pResult->v = 0;

        kernelSphericalGridGetIndex<<<1,1>>>( pGrid->ptrDevicePtr, pResult->devicePtr, pos8);
		gpuErrchk( cudaPeekAtLastError() );
		pResult->copyToCPU();
    	CHECK_EQUAL( 9, pResult->v );
    	pResult->v = 0;

        kernelSphericalGridGetIndex<<<1,1>>>( pGrid->ptrDevicePtr, pResult->devicePtr, pos9);
		gpuErrchk( cudaPeekAtLastError() );
		pResult->copyToCPU();
    	CHECK_EQUAL( UINT_MAX, pResult->v );
    	pResult->v = 0;
//
        delete pResult;
    }

   	CUDA_CALLABLE_KERNEL void kernelGetRadialIndexFromR(Grid_t** pPtrGrid, resultClass<int>* pResult, gpuRayFloat_t R ) {
   		pResult->v = (*pPtrGrid)->getRadialIndexFromR(R);
	}

   	CUDA_CALLABLE_KERNEL void kernelGetRadialIndexFromRSq(Grid_t** pPtrGrid, resultClass<int>* pResult, gpuRayFloat_t RSq ) {
   		pResult->v = (*pPtrGrid)->getRadialIndexFromRSq(RSq);
	}

   	CUDA_CALLABLE_KERNEL void kernelSphericalGridIsIndexOutside(Grid_t** pGrid, resultClass<bool>* pResult, unsigned d, int index) {
   		pResult->v = (*pGrid)->isIndexOutside(d,index);
	}

   	CUDA_CALLABLE_KERNEL void kernelSphericalGridIsOutside(Grid_t** pGrid, resultClass<bool>* pResult, int i, int j, int k ) {
   		int indices[] = {i,j,k};
   		pResult->v = (*pGrid)->isOutside(indices);
	}

   	class SphericalGridGPUTester {
   	public:
   		SphericalGridGPUTester(){
   			pGrid = std::unique_ptr<Grid_t>( new Grid_t(1,data.pGridInfo));
   			pGrid->copyToGPU();
   		}
   		~SphericalGridGPUTester(){}

   	   	int getRadialIndexFromR( gpuRayFloat_t R ) {
   	   		using result_t = resultClass<int>;
   	   	    std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   	    pResult->copyToGPU();

   	   	    kernelGetRadialIndexFromR<<<1,1>>>( pGrid->ptrDevicePtr, pResult->devicePtr, R );
   	   		gpuErrchk( cudaPeekAtLastError() );
   	   		pResult->copyToCPU();
   	   		return pResult->v;
   	   	}

   	   	int getRadialIndexFromRSq( gpuRayFloat_t RSq ) {
   	   		using result_t = resultClass<int>;
   	   	    std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   	    pResult->copyToGPU();

   	   	    kernelGetRadialIndexFromRSq<<<1,1>>>( pGrid->ptrDevicePtr, pResult->devicePtr, RSq );
   	   		gpuErrchk( cudaPeekAtLastError() );
   	   		pResult->copyToCPU();
   	   		return pResult->v;
   	   	}

   	   	unsigned getIndex( Position_t pos) {
   	   		using result_t = resultClass<unsigned>;
   	   	    std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   	    pResult->copyToGPU();

   	   	    kernelSphericalGridGetIndex<<<1,1>>>( pGrid->ptrDevicePtr, pResult->devicePtr, pos);
   	   		gpuErrchk( cudaPeekAtLastError() );
   	   		pResult->copyToCPU();
   	   		return pResult->v;
   	   	}

   	   	bool isIndexOutside( unsigned d, int index ) {
   	   		using result_t = resultClass<bool>;
   	   	    std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   	    pResult->copyToGPU();

   	   	    kernelSphericalGridIsIndexOutside<<<1,1>>>( pGrid->ptrDevicePtr, pResult->devicePtr, d, index);
   	   		gpuErrchk( cudaPeekAtLastError() );
   	   		pResult->copyToCPU();
   	   		return pResult->v;
   	   	}

   	   	bool isOutside( const int indices[]) {
   	   		using result_t = resultClass<bool>;
   	   	    std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   	    pResult->copyToGPU();

   	   	    kernelSphericalGridIsOutside<<<1,1>>>( pGrid->ptrDevicePtr, pResult->devicePtr, indices[0], indices[1], indices[2]);
   	   		gpuErrchk( cudaPeekAtLastError() );
   	   		pResult->copyToCPU();
   	   		return pResult->v;
   	   	}

   	   	gridTestData data;
   	   	std::unique_ptr<Grid_t> pGrid;
   	};

   	TEST_FIXTURE( SphericalGridGPUTester, getRadialIndexFromR_outside ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(1,data.pGridInfo));
        CHECK_EQUAL( 10, getRadialIndexFromR( 10.5 ) );
    }
   	TEST_FIXTURE( SphericalGridGPUTester, getRadialIndexFromR_inside ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(1,data.pGridInfo));
        CHECK_EQUAL( 9, getRadialIndexFromR( 9.5 ) );
    }
   	TEST_FIXTURE( SphericalGridGPUTester, getRadialIndexFromR_insideOnVertex ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(1,data.pGridInfo));
        CHECK_EQUAL( 9, getRadialIndexFromR( 9.0 ) );
    }
   	TEST_FIXTURE( SphericalGridGPUTester, getRadialIndexFromR_center ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(1,data.pGridInfo));
        CHECK_EQUAL( 0, getRadialIndexFromR( 0.0 ) );
    }

    TEST_FIXTURE( SphericalGridGPUTester, getRadialIndexFromRSq_outside ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(1,data.pGridInfo));
        CHECK_EQUAL( 10, getRadialIndexFromRSq( 10.5*10.5 ) );
    }
    TEST_FIXTURE( SphericalGridGPUTester, getRadialIndexFromRSq_inside ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(1,data.pGridInfo));
        CHECK_EQUAL( 9, getRadialIndexFromRSq( 9.5*9.5 ) );
    }
    TEST_FIXTURE( SphericalGridGPUTester, getRadialIndexFromRSq_insideOnVertex ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(1,data.pGridInfo));
        CHECK_EQUAL( 9, getRadialIndexFromRSq( 9.0*9.0 ) );
    }
    TEST_FIXTURE( SphericalGridGPUTester, getRadialIndexFromRSq_center ) {
    	gridTestData data;
    	std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(1,data.pGridInfo));
        CHECK_EQUAL( 0, getRadialIndexFromRSq( 0.0 ) );
    }

    TEST_FIXTURE(SphericalGridGPUTester, isOutside_index ) {
    	CHECK_EQUAL( true, isIndexOutside(0, 10) );
    }

    TEST_FIXTURE(SphericalGridGPUTester, isOutside_Radius_false ) {
    	CHECK_EQUAL( false, isIndexOutside(0, 9) );
    }

     CUDA_CALLABLE_KERNEL void kernelSphericalGridGetVolume(Grid_t** pGrid, resultClass<gpuRayFloat_t>* pResult, unsigned i ) {
    	 pResult->v = (*pGrid)->getVolume(i);
     }

     gpuRayFloat_t getVolume(Grid_t& grid, unsigned i ) {
    	 using result_t = resultClass<gpuRayFloat_t>;
    	 std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
    	 pResult->copyToGPU();

    	 kernelSphericalGridGetVolume<<<1,1>>>( grid.ptrDevicePtr, pResult->devicePtr, i);
    	 gpuErrchk( cudaPeekAtLastError() );
    	 pResult->copyToCPU();
    	 return pResult->v;
     }

     TEST( getVolume ) {
     	pGridInfo_t* pGridInfo = new pGridInfo_t[3];
 		pGridInfo[0] = new GridBins_t();
 		pGridInfo[1] = new GridBins_t();
 		pGridInfo[2] = new GridBins_t();

     	std::vector<gpuRayFloat_t> vertices = { 1.0, 2.0, 3.0 };

     	pGridInfo[0]->initialize( vertices );

     	Grid_t grid(1,pGridInfo);
     	grid.copyToGPU();

        CHECK_CLOSE( (1.0)*(4.0/3.0)*pi, getVolume(grid, 0), 1e-5 );
        CHECK_CLOSE( (8.0-1.0)*(4.0/3.0)*pi, getVolume(grid, 1), 1e-4 );

     	delete pGridInfo[0];
     	delete pGridInfo[1];
     	delete pGridInfo[2];

     	delete[] pGridInfo;
     }

}

#endif

} // end namespace
