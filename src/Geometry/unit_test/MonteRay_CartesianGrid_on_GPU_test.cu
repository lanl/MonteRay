#include <UnitTest++.h>

#include <memory>
#include <vector>
#include <array>

#include "MonteRay_CartesianGrid.hh"
#include "MonteRay_SpatialGrid.hh"
#include "GPUSync.hh"
#include "MonteRayVector3D.hh"

using namespace MonteRay;

namespace MonteRay_CartesianGrid_on_GPU_tester{

#if true

SUITE( MonteRay_CartesianGrid_GPU_basic_tests ) {
	using Grid_t = MonteRay_CartesianGrid;
	using GridBins_t = MonteRay_GridBins;
	using GridBins_t = Grid_t::GridBins_t;
	using pGridInfo_t = GridBins_t*;
	using pArrayOfpGridInfo_t = Grid_t::pArrayOfpGridInfo_t;

    typedef MonteRay::Vector3D<gpuFloatType_t> Position_t;

    class gridTestData {
    public:
        enum coord {X,Y,Z,DIM};
        gridTestData(){

        	std::vector<gpuFloatType_t> vertices = { -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
        			0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10 };

        	pGridInfo[X] = new GridBins_t();
        	pGridInfo[Y] = new GridBins_t();
        	pGridInfo[Z] = new GridBins_t();

        	pGridInfo[X]->initialize( vertices );
        	pGridInfo[Y]->initialize( vertices );
        	pGridInfo[Z]->initialize( vertices );


//        	pGridInfo[X]->copyToGPU();
//        	pGridInfo[Y]->copyToGPU();
//        	pGridInfo[Z]->copyToGPU();
        }
        ~gridTestData(){
        	delete pGridInfo[X];
        	delete pGridInfo[Y];
        	delete pGridInfo[Z];

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
	CUDA_CALLABLE_KERNEL void kernelCartesianGridGetNumBins(Grid_t** pCart, resultClass<unsigned>* pResult, unsigned d) {
		pResult->v = (*pCart)->getNumBins(d);
	}

    TEST( getNumBins_on_GPU ) {
    	enum coord {X,Y,Z,DimMax};
    	gridTestData data;

    	resultClass<unsigned>* pResult = new resultClass<unsigned>();
    	pResult->copyToGPU();

    	std::unique_ptr<Grid_t> pCart = std::unique_ptr<Grid_t>( new Grid_t(3,data.pGridInfo));
    	pCart->copyToGPU();


//    	Grid_t** devicePtr;
//    	data.pGridInfo[X]->copyToGPU();
//    	data.pGridInfo[Y]->copyToGPU();
//    	data.pGridInfo[Z]->copyToGPU();
//    	createDeviceInstance<<<1,1>>>( devicePtr, data.pGridInfo[X]->devicePtr, data.pGridInfo[Y]->devicePtr, data.pGridInfo[Z]->devicePtr );

    	GPUSync sync1;
		sync1.sync();

    	gpuErrchk( cudaPeekAtLastError() );

    	//printf( "Debug: devicePtr = %d\n", devicePtr );

    	kernelCartesianGridGetNumBins<<<1,1>>>( pCart->devicePtr, pResult->devicePtr, 0);
    	gpuErrchk( cudaPeekAtLastError() );
    	pResult->copyToCPU();
    	CHECK_EQUAL( 20, pResult->v );
    	pResult->v = 0;


		kernelCartesianGridGetNumBins<<<1,1>>>( pCart->devicePtr, pResult->devicePtr, 0);
		gpuErrchk( cudaPeekAtLastError() );
		pResult->copyToCPU();
    	CHECK_EQUAL( 20, pResult->v );
    	pResult->v = 0;

		kernelCartesianGridGetNumBins<<<1,1>>>( pCart->devicePtr, pResult->devicePtr, 1);
		gpuErrchk( cudaPeekAtLastError() );
		pResult->copyToCPU();
    	CHECK_EQUAL( 20, pResult->v );
    	pResult->v = 0;

		kernelCartesianGridGetNumBins<<<1,1>>>( pCart->devicePtr, pResult->devicePtr, 2);
		gpuErrchk( cudaPeekAtLastError() );
		pResult->copyToCPU();
    	CHECK_EQUAL( 20, pResult->v );

    	delete pResult;
    }

//	// kernal call
	CUDA_CALLABLE_KERNEL void kernelCartesianGridGetIndex(Grid_t** pCart, resultClass<unsigned>* pResult, Position_t pos) {
		//printf("Debug: kernelCartesianGridGetIndex -- calling pCart->getIndex(pos)\n");
		unsigned index = (*pCart)->getIndex(pos);
		pResult->v = index;
	}

    TEST( getIndex ) {
        gridTestData data;
        std::unique_ptr<Grid_t> pCart = std::unique_ptr<Grid_t>( new Grid_t(3,data.pGridInfo));
        pCart->copyToGPU();

    	resultClass<unsigned>* pResult = new resultClass<unsigned>();
    	pResult->copyToGPU();

        Position_t pos1( -9.5, -9.5, -9.5 );
        Position_t pos2( -8.5, -9.5, -9.5 );
        Position_t pos3( -9.5, -8.5, -9.5 );
        Position_t pos4( -9.5, -9.5, -8.5 );
        Position_t pos5( -9.5, -7.5, -9.5 );

        kernelCartesianGridGetIndex<<<1,1>>>( pCart->devicePtr, pResult->devicePtr, pos1);
		gpuErrchk( cudaPeekAtLastError() );
		pResult->copyToCPU();
    	CHECK_EQUAL( 0, pResult->v );
    	pResult->v = 0;

        kernelCartesianGridGetIndex<<<1,1>>>( pCart->devicePtr, pResult->devicePtr, pos2);
		gpuErrchk( cudaPeekAtLastError() );
		pResult->copyToCPU();
    	CHECK_EQUAL( 1, pResult->v );
    	pResult->v = 0;

        kernelCartesianGridGetIndex<<<1,1>>>( pCart->devicePtr, pResult->devicePtr, pos3);
		gpuErrchk( cudaPeekAtLastError() );
		pResult->copyToCPU();
    	CHECK_EQUAL( 20, pResult->v );
    	pResult->v = 0;

        kernelCartesianGridGetIndex<<<1,1>>>( pCart->devicePtr, pResult->devicePtr, pos5);
		gpuErrchk( cudaPeekAtLastError() );
		pResult->copyToCPU();
    	CHECK_EQUAL( 40, pResult->v );
    	pResult->v = 0;

        kernelCartesianGridGetIndex<<<1,1>>>( pCart->devicePtr, pResult->devicePtr, pos4);
		gpuErrchk( cudaPeekAtLastError() );
		pResult->copyToCPU();
    	CHECK_EQUAL( 400, pResult->v );
    	pResult->v = 0;

////        CHECK_EQUAL(   0, pCart->getIndex( pos1 ) );
////        CHECK_EQUAL(   1, pCart->getIndex( pos2 ) );
////        CHECK_EQUAL(  20, pCart->getIndex( pos3 ) );
////        CHECK_EQUAL(  40, pCart->getIndex( pos5 ) );
////        CHECK_EQUAL( 400, pCart->getIndex( pos4 ) );
//
        delete pResult;
    }

   	CUDA_CALLABLE_KERNEL void kernelGetDimIndex(Grid_t** pCart, resultClass<int>* pResult, unsigned d, gpuFloatType_t pos) {
   		pResult->v = (*pCart)->getDimIndex(d,pos);
	}

   	CUDA_CALLABLE_KERNEL void kernelCartesianGridIsIndexOutside(Grid_t** pCart, resultClass<bool>* pResult, unsigned d, int index) {
   		pResult->v = (*pCart)->isIndexOutside(d,index);
	}

   	CUDA_CALLABLE_KERNEL void kernelCartesianGridIsOutside(Grid_t** pCart, resultClass<bool>* pResult, int i, int j, int k ) {
   		int indices[] = {i,j,k};
   		pResult->v = (*pCart)->isOutside(indices);
	}

   	class CartesianGridGPUTester {
   	public:
   		CartesianGridGPUTester(){
   			pCart = std::unique_ptr<Grid_t>( new Grid_t(3,data.pGridInfo));
   			pCart->copyToGPU();
   		}
   		~CartesianGridGPUTester(){}

   	   	int getDimIndex( unsigned d, gpuFloatType_t pos) {
   	   		using result_t = resultClass<int>;
   	   	    std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   	    pResult->copyToGPU();

   	   	    kernelGetDimIndex<<<1,1>>>( pCart->devicePtr, pResult->devicePtr, d, pos);
   	   		gpuErrchk( cudaPeekAtLastError() );
   	   		pResult->copyToCPU();
   	   		return pResult->v;
   	   	}

   	   	unsigned getIndex( Position_t pos) {
   	   		using result_t = resultClass<unsigned>;
   	   	    std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   	    pResult->copyToGPU();

   	   	    kernelCartesianGridGetIndex<<<1,1>>>( pCart->devicePtr, pResult->devicePtr, pos);
   	   		gpuErrchk( cudaPeekAtLastError() );
   	   		pResult->copyToCPU();
   	   		return pResult->v;
   	   	}

   	   	bool isIndexOutside( unsigned d, int index ) {
   	   		using result_t = resultClass<bool>;
   	   	    std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   	    pResult->copyToGPU();

   	   	    kernelCartesianGridIsIndexOutside<<<1,1>>>( pCart->devicePtr, pResult->devicePtr, d, index);
   	   		gpuErrchk( cudaPeekAtLastError() );
   	   		pResult->copyToCPU();
   	   		return pResult->v;
   	   	}

   	   	bool isOutside( const int indices[]) {
   	   		using result_t = resultClass<bool>;
   	   	    std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   	    pResult->copyToGPU();

   	   	    kernelCartesianGridIsOutside<<<1,1>>>( pCart->devicePtr, pResult->devicePtr, indices[0], indices[1], indices[2]);
   	   		gpuErrchk( cudaPeekAtLastError() );
   	   		pResult->copyToCPU();
   	   		return pResult->v;
   	   	}

   	   	gridTestData data;
   	   	std::unique_ptr<Grid_t> pCart;
   	};


    TEST_FIXTURE(CartesianGridGPUTester, getDimIndex_negX ) {
    	CHECK_EQUAL( -1, getDimIndex( 0, -10.5 ) );
    }

    TEST_FIXTURE(CartesianGridGPUTester, getDimIndex_posX ) {
        CHECK_EQUAL( 20, getDimIndex( 0, 10.5 ) );
    }
    TEST_FIXTURE(CartesianGridGPUTester, getDimIndex_inside_negSide_X ) {
        CHECK_EQUAL( 0, getDimIndex( 0, -9.5 ) );
    }
    TEST_FIXTURE(CartesianGridGPUTester, getDimIndex_inside_posSide_X ) {
        CHECK_EQUAL( 19, getDimIndex( 0, 9.5 ) );
    }

    TEST_FIXTURE(CartesianGridGPUTester, getDimIndex_negY ) {
        CHECK_EQUAL( -1, getDimIndex( 1, -10.5 ) );
    }
    TEST_FIXTURE(CartesianGridGPUTester, getDimIndex_posY ) {
        CHECK_EQUAL( 20, getDimIndex( 1, 10.5 ) );
    }
    TEST_FIXTURE(CartesianGridGPUTester, getDimIndex_inside_negSide_Y ) {
        CHECK_EQUAL( 0, getDimIndex( 1, -9.5 ) );
    }
    TEST_FIXTURE(CartesianGridGPUTester, getDimIndex_inside_posSide_Y ) {
        CHECK_EQUAL( 19, getDimIndex( 1, 9.5 ) );
    }

    TEST_FIXTURE(CartesianGridGPUTester, getDimIndex_negZ ) {
        CHECK_EQUAL( -1, getDimIndex( 2, -10.5 ) );
    }
    TEST_FIXTURE(CartesianGridGPUTester, getDimIndex_posZ ) {
        CHECK_EQUAL( 20, getDimIndex( 2, 10.5 ) );
    }
    TEST_FIXTURE(CartesianGridGPUTester,getDimIndex_inside_negSide_Z ) {
        CHECK_EQUAL( 0, getDimIndex( 2, -9.5 ) );
    }
    TEST_FIXTURE(CartesianGridGPUTester,getDimIndex_inside_posSide_Z ) {
        CHECK_EQUAL( 19,getDimIndex( 2, 9.5 ) );
    }

    TEST_FIXTURE(CartesianGridGPUTester, PositionOutOfBoundsToGrid ) {

        Position_t posNegX( -10.5, -9.5, -9.5 );
        Position_t posPosX(  10.5, -9.5, -9.5 );

        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, getIndex( posNegX ) );
        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, getIndex( posPosX ) );

        Position_t posNegY( -9.5, -10.5, -9.5 );
        Position_t posPosY( -9.5,  10.5, -9.5 );

        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, getIndex( posNegY ) );
        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, getIndex( posPosY ) );

        Position_t posNegZ( -9.5, -9.5, -10.5 );
        Position_t posPosZ( -9.5, -9.5,  10.5 );

        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, getIndex( posNegZ ) );
        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, getIndex( posPosZ ) );
    }

    TEST_FIXTURE(CartesianGridGPUTester, PositionOnTheBoundsToGrid_WeDefineOutside ) {

        Position_t posNegX( -10.0, -9.5, -9.5 );
        Position_t posPosX(  10.0, -9.5, -9.5 );

        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, getIndex( posNegX ) );
        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, getIndex( posPosX ) );

        Position_t posNegY( -9.5, -10.0, -9.5 );
        Position_t posPosY( -9.5,  10.0, -9.5 );

        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, getIndex( posNegY ) );
        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, getIndex( posPosY ) );

        Position_t posNegZ( -9.5, -9.5, -10.0 );
        Position_t posPosZ( -9.5, -9.5,  10.0 );

        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, getIndex( posNegZ ) );
        CHECK_EQUAL( MonteRay_SpatialGrid::OUTSIDE_MESH, getIndex( posPosZ ) );
    }

    TEST_FIXTURE(CartesianGridGPUTester, isIndexOutside_negX ) {
    	CHECK_EQUAL( true, isIndexOutside(0, -1) );
    }

    TEST_FIXTURE(CartesianGridGPUTester, isIndexOutside_posX ) {
    	CHECK_EQUAL( true, isIndexOutside(0, 20) );
    }

    TEST_FIXTURE(CartesianGridGPUTester, isIndexOutside_false_negEnd ) {
    	CHECK_EQUAL( false, isIndexOutside(0, 0) );
    }

    TEST_FIXTURE(CartesianGridGPUTester, isIndexOutside_false_posEnd ) {
    	CHECK_EQUAL( false, isIndexOutside(0, 19) );
    }

     TEST_FIXTURE(CartesianGridGPUTester, isOutside_negX ) {
         int indices[] = {-1,0,0};
         CHECK_EQUAL( true, isOutside( indices ) );
     }

     TEST_FIXTURE(CartesianGridGPUTester, isOutside_posX ) {
         int indices[] = {20,0,0};
         CHECK_EQUAL( true, isOutside( indices ) );
     }

     TEST_FIXTURE(CartesianGridGPUTester, isOutside_negY ) {
         int indices[] = {0,-1,0};
         CHECK_EQUAL( true, isOutside( indices ) );
     }
     TEST_FIXTURE(CartesianGridGPUTester, isOutside_posY ) {
         int indices[] = {0,20,0};
         CHECK_EQUAL( true, isOutside( indices ) );
     }
     TEST_FIXTURE(CartesianGridGPUTester, isOutside_negZ ) {
         int indices[] = {0,0,-1};
         CHECK_EQUAL( true, isOutside( indices ) );
     }
     TEST_FIXTURE(CartesianGridGPUTester, isOutside_posZ ) {
         int indices[] = {0,0,20};
         CHECK_EQUAL( true, isOutside( indices ) );
     }
     TEST_FIXTURE(CartesianGridGPUTester, isOutside_false1 ) {
         int indices[] = {19,0,0};
         CHECK_EQUAL( false, isOutside( indices ) );
     }
     TEST_FIXTURE(CartesianGridGPUTester, isOutside_false2 ) {
         int indices[] = {0,0,0};
         CHECK_EQUAL( false, isOutside( indices ) );
     }


     CUDA_CALLABLE_KERNEL void kernelCartesianGridGetVolume(Grid_t** pCart, resultClass<gpuFloatType_t>* pResult, unsigned i ) {
    	 pResult->v = (*pCart)->getVolume(i);
     }

     gpuFloatType_t getVolume(Grid_t& grid, unsigned i ) {
    	 using result_t = resultClass<gpuFloatType_t>;
    	 std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
    	 pResult->copyToGPU();

    	 kernelCartesianGridGetVolume<<<1,1>>>( grid.devicePtr, pResult->devicePtr, i);
    	 gpuErrchk( cudaPeekAtLastError() );
    	 pResult->copyToCPU();
    	 return pResult->v;
     }

     TEST( getVolume ) {
     	pGridInfo_t* pGridInfo = new pGridInfo_t[3];
 		pGridInfo[0] = new GridBins_t();
 		pGridInfo[1] = new GridBins_t();
 		pGridInfo[2] = new GridBins_t();

     	std::vector<gpuFloatType_t> vertices = {-3, -1, 0};

     	pGridInfo[0]->initialize( vertices );
     	pGridInfo[1]->initialize( vertices );
     	pGridInfo[2]->initialize( vertices );

     	Grid_t cart(3,pGridInfo);
     	cart.copyToGPU();

     	CHECK_CLOSE( 8.0, getVolume(cart,0), 1e-11 );
     	CHECK_CLOSE( 4.0, getVolume(cart,1), 1e-11 );
     	CHECK_CLOSE( 4.0, getVolume(cart,2), 1e-11 );
     	CHECK_CLOSE( 2.0, getVolume(cart,3), 1e-11 );
     	CHECK_CLOSE( 4.0, getVolume(cart,4), 1e-11 );
     	CHECK_CLOSE( 2.0, getVolume(cart,5), 1e-11 );
     	CHECK_CLOSE( 2.0, getVolume(cart,6), 1e-11 );
     	CHECK_CLOSE( 1.0, getVolume(cart,7), 1e-11 );

     	delete pGridInfo[0];
     	delete pGridInfo[1];
     	delete pGridInfo[2];

     	delete[] pGridInfo;
     }

}

#endif

} // end namespace
