#include <UnitTest++.h>

#include <memory>
#include <vector>
#include <array>

#include "MonteRay_CylindricalGrid.hh"
#include "MonteRay_SpatialGrid.hh"
#include "GPUSync.hh"
#include "MonteRayVector3D.hh"
#include "MonteRayConstants.hh"
#include "GPUUtilityFunctions.hh"
#include "MonteRayCopyMemory.t.hh"

using namespace MonteRay;

namespace MonteRay_CylindricalGrid_on_GPU_tester{

SUITE( MonteRay_CylindricalGrid_GPU_basic_tests ) {
#ifdef __CUDACC__
    using Grid_t = MonteRay_CylindricalGrid;
    using GridBins_t = MonteRay_GridBins;
    using GridBins_t = Grid_t::GridBins_t;
    using pGridInfo_t = GridBins_t*;
    using pArrayOfpGridInfo_t = Grid_t::pArrayOfpGridInfo_t;

    typedef MonteRay::Vector3D<gpuRayFloat_t> Position_t;

    const unsigned OUTSIDE_MESH = MonteRay_SpatialGrid::OUTSIDE_MESH;

    enum coord {R=0,Z=1,Theta=2,DIM=3};

    class gridTestData {
    public:
        gridTestData(){
            std::vector<gpuRayFloat_t> Rverts = { 1.3, 2.2, 5.0 };

            pGridInfo[R] = new GridBins_t();
            pGridInfo[Z] = new GridBins_t();

            pGridInfo[R]->initialize( Rverts );
            pGridInfo[Z]->initialize( -10.1, 10.2, 20 );

        }
        ~gridTestData(){
            delete pGridInfo[R];
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
            //v = T(0);
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

#ifdef DEBUG
            if( this->debug ) {
                std::cout << "Debug: 1- resultClass::copy(const resultClass* rhs) \n";
            }
#endif

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
#ifdef __CUDACC__
    CUDA_CALLABLE_KERNEL void kernelCylindricalGridGetNumBins(Grid_t** pGrid, resultClass<unsigned>* pResult, unsigned d) {
        pResult->v = (*pGrid)->getNumBins(d);
    }
#else
    void kernelCylindricalGridGetNumBins(Grid_t Grid, resultClass<unsigned>* pResult, unsigned d) {
        pResult->v = Grid.getNumBins(d);
    }
#endif

    unsigned launchGetNumBins(Grid_t& Grid, unsigned d ) {
        using T = unsigned;

        resultClass<T>* pResult = new resultClass<T>();

#ifdef __CUDACC__
        pResult->copyToGPU();

        GPUSync sync1;
        sync1.sync();

        gpuErrchk( cudaPeekAtLastError() );

        kernelCylindricalGridGetNumBins<<<1,1>>>( Grid.ptrDevicePtr, pResult->devicePtr, d);

        GPUSync sync2;
        sync2.sync();

        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelCylindricalGridGetNumBins( Grid, pResult, d);
#endif

        T value = pResult->v;

        delete pResult;
        return value;
    }

    TEST( getNumBins_on_GPU ) {
        //gpuReset();
        gridTestData data;

        std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(2,data.pGridInfo));
#ifdef __CUDACC__
        pGrid->copyToGPU();

        GPUSync sync1;
        sync1.sync();

        gpuErrchk( cudaPeekAtLastError() );
#endif

        CHECK_EQUAL(  3, launchGetNumBins( *pGrid, 0) );
        CHECK_EQUAL( 20, launchGetNumBins( *pGrid, 1) );
        //CHECK_EQUAL(  1, launchGetNumBins( *pGrid, 2) );

    }

    // kernal call
#ifdef __CUDACC__
    CUDA_CALLABLE_KERNEL void kernelCylindricalGridGetRVertex(Grid_t** pGrid, resultClass<gpuFloatType_t>* pResult, unsigned i) {
        pResult->v = (*pGrid)->getRVertex(i);
    }
#else
    void kernelCylindricalGridGetRVertex(Grid_t Grid, resultClass<gpuFloatType_t>* pResult, unsigned i) {
        pResult->v = Grid.getRVertex(i);
    }
#endif

#ifdef __CUDACC__
    CUDA_CALLABLE_KERNEL void kernelCylindricalGridGetZVertex(Grid_t** pGrid, resultClass<gpuFloatType_t>* pResult, unsigned i) {
        pResult->v = (*pGrid)->getZVertex(i);
    }
#else
    void kernelCylindricalGridGetZVertex(Grid_t Grid, resultClass<gpuFloatType_t>* pResult, unsigned i) {
        pResult->v = Grid.getZVertex(i);
    }
#endif

    gpuFloatType_t launchGetRVertex(Grid_t& Grid, unsigned i ) {
        using T = gpuFloatType_t;

        resultClass<T>* pResult = new resultClass<T>();
#ifdef __CUDACC__
        pResult->copyToGPU();

        GPUSync sync1;
        sync1.sync();

        gpuErrchk( cudaPeekAtLastError() );

        kernelCylindricalGridGetRVertex<<<1,1>>>( Grid.ptrDevicePtr, pResult->devicePtr, i);

        GPUSync sync2;
        sync2.sync();

        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelCylindricalGridGetRVertex( Grid, pResult, i);
#endif

        T value = pResult->v;

        delete pResult;
        return value;
    }

    gpuFloatType_t launchGetZVertex(Grid_t& Grid, unsigned i ) {
        using T = gpuFloatType_t;

        resultClass<T>* pResult = new resultClass<T>();
#ifdef __CUDACC__
        pResult->copyToGPU();

        GPUSync sync1;
        sync1.sync();

        gpuErrchk( cudaPeekAtLastError() );

        kernelCylindricalGridGetZVertex<<<1,1>>>( Grid.ptrDevicePtr, pResult->devicePtr, i);

        GPUSync sync2;
        sync2.sync();

        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelCylindricalGridGetZVertex( Grid, pResult, i);
#endif

        T value = pResult->v;

        delete pResult;
        return value;
    }

    TEST( special_case_remove_zero_R_entry ) {
        //gpuReset();
        std::vector<double> Rverts = { 0.0, 1.5, 2.0 };
        std::vector<double> Zverts = { -10, -5, 0, 5, 10 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[R] = new GridBins_t();
        pGridInfo[Z] = new GridBins_t();
        pGridInfo[R]->initialize(  Rverts );
        pGridInfo[Z]->initialize(  Zverts );

        std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(2,pGridInfo));
#ifdef __CUDACC__
        pGrid->copyToGPU();

        GPUSync sync1;
        sync1.sync();

        gpuErrchk( cudaPeekAtLastError() );
#endif

        CHECK_CLOSE(   1.5, launchGetRVertex( *pGrid, 0), 1e-5);
        CHECK_CLOSE(   2.0, launchGetRVertex( *pGrid, 1), 1e-5);
        CHECK_CLOSE( -10.0, launchGetZVertex( *pGrid, 0), 1e-5);
        CHECK_CLOSE(   5.0, launchGetZVertex( *pGrid, 3), 1e-5);

        delete pGridInfo[R];
        delete pGridInfo[Z];
    }

    // kernal call
#ifdef __CUDACC__
    CUDA_CALLABLE_KERNEL void kernelCylindricalGridConvertFromCartesian(Grid_t** pGrid, resultClass<Vector3D<gpuFloatType_t>>* pResult, Vector3D<gpuFloatType_t> pos) {
        pResult->v = (*pGrid)->convertFromCartesian(pos);
    }
#else
    void kernelCylindricalGridConvertFromCartesian(Grid_t Grid, resultClass<Vector3D<gpuFloatType_t>>* pResult, Vector3D<gpuFloatType_t> pos) {
        pResult->v = Grid.convertFromCartesian(pos);
    }
#endif

    Vector3D<gpuFloatType_t> launchConvertFromCartesian(Grid_t& Grid, Vector3D<gpuFloatType_t> pos ) {
        using T = Vector3D<gpuFloatType_t>;

        resultClass<T>* pResult = new resultClass<T>();
#ifdef __CUDACC__
        pResult->copyToGPU();

        GPUSync sync1;
        sync1.sync();

        gpuErrchk( cudaPeekAtLastError() );

        kernelCylindricalGridConvertFromCartesian<<<1,1>>>( Grid.ptrDevicePtr, pResult->devicePtr, pos);

        GPUSync sync2;
        sync2.sync();

        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelCylindricalGridConvertFromCartesian( Grid, pResult, pos);
#endif

        T value = pResult->v;

        delete pResult;
        return value;
    }

    TEST( convertFromCartesian ) {
        //gpuReset();
        std::vector<double> Rverts = { 0.0, 1.5, 2.0 };
        std::vector<double> Zverts = { -10, -5, 0, 5, 10 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[R] = new GridBins_t();
        pGridInfo[Z] = new GridBins_t();
        pGridInfo[R]->initialize(  Rverts );
        pGridInfo[Z]->initialize(  Zverts );

        std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(2,pGridInfo));
#ifdef __CUDACC__
        pGrid->copyToGPU();

        GPUSync sync1;
        sync1.sync();

        gpuErrchk( cudaPeekAtLastError() );
#endif

        Vector3D<gpuRayFloat_t> pos = launchConvertFromCartesian( *pGrid, Vector3D<gpuRayFloat_t>( 1.0, 0.0, 5.0) );
        CHECK_CLOSE( 1.0, pos[0], 1e-11);
        CHECK_CLOSE( 5.0, pos[1], 1e-11);
        CHECK_CLOSE( 0.0, pos[2], 1e-11);

        pos = launchConvertFromCartesian( *pGrid, Vector3D<gpuRayFloat_t>( 1.0, 1.0, 5.0) );
        CHECK_CLOSE( std::sqrt(2.0), pos[0], 1e-5);
        CHECK_CLOSE( 5.0, pos[1], 1e-11);
        CHECK_CLOSE( 0.0, pos[2], 1e-11);

        delete pGridInfo[R];
        delete pGridInfo[Z];
    }

    // kernal call
#ifdef __CUDACC__
    CUDA_CALLABLE_KERNEL void kernelCylindricalGridGetRadialIndexFromR(Grid_t** pGrid, resultClass<unsigned>* pResult, gpuFloatType_t pos) {
        pResult->v = (*pGrid)->getRadialIndexFromR(pos);
    }
#else
    void kernelCylindricalGridGetRadialIndexFromR(Grid_t Grid, resultClass<unsigned>* pResult, gpuFloatType_t pos) {
        pResult->v = Grid.getRadialIndexFromR(pos);
    }
#endif

    unsigned launchGetRadialIndexFromR(Grid_t& Grid, gpuFloatType_t r ) {
        using T = unsigned;
        resultClass<T>* pResult = new resultClass<T>();

#ifdef __CUDACC__

        pResult->copyToGPU();

        GPUSync sync1;
        sync1.sync();

        gpuErrchk( cudaPeekAtLastError() );

        kernelCylindricalGridGetRadialIndexFromR<<<1,1>>>( Grid.ptrDevicePtr, pResult->devicePtr, r);

        GPUSync sync2;
        sync2.sync();

        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelCylindricalGridGetRadialIndexFromR( Grid, pResult, r);
#endif
        T value = pResult->v;

        delete pResult;
        return value;
    }

    TEST( getRadialIndexFromR ) {
        //gpuReset();
        std::vector<double> Rverts = { 1.0, 2.0, 3.0 };
        std::vector<double> Zverts = { -10, 0, 10 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[R] = new GridBins_t();
        pGridInfo[Z] = new GridBins_t();
        pGridInfo[R]->initialize(  Rverts );
        pGridInfo[Z]->initialize(  Zverts );

        std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(2,pGridInfo));
#ifdef __CUDACC__
        pGrid->copyToGPU();

        GPUSync sync1;
        sync1.sync();

        gpuErrchk( cudaPeekAtLastError() );
#endif

        //printf( "Debug: devicePtr = %d\n", devicePtr );

        CHECK_EQUAL( 0, launchGetRadialIndexFromR( *pGrid, 0.5) );
        CHECK_EQUAL( 1, launchGetRadialIndexFromR( *pGrid, 1.5) );
        CHECK_EQUAL( 2, launchGetRadialIndexFromR( *pGrid, 2.5) );
        CHECK_EQUAL( 3, launchGetRadialIndexFromR( *pGrid, 3.5) );
        CHECK_EQUAL( 3, launchGetRadialIndexFromR( *pGrid, 30.5) );

        delete pGridInfo[R];
        delete pGridInfo[Z];
    }

    // kernal call
#ifdef __CUDACC__
    CUDA_CALLABLE_KERNEL void kernelCylindricalGridGetRadialIndexFromRSq(Grid_t** pGrid, resultClass<unsigned>* pResult, gpuFloatType_t rSq) {
        pResult->v = (*pGrid)->getRadialIndexFromRSq(rSq);
    }
#else
    void kernelCylindricalGridGetRadialIndexFromRSq(Grid_t Grid, resultClass<unsigned>* pResult, gpuFloatType_t rSq) {
        pResult->v = Grid.getRadialIndexFromRSq(rSq);
    }
#endif

    unsigned launchGetRadialIndexFromRSq(Grid_t& Grid, gpuFloatType_t rSq ) {
        using T = unsigned;

        resultClass<T>* pResult = new resultClass<T>();
#ifdef __CUDACC__
        pResult->copyToGPU();

        GPUSync sync1;
        sync1.sync();

        gpuErrchk( cudaPeekAtLastError() );

        kernelCylindricalGridGetRadialIndexFromRSq<<<1,1>>>( Grid.ptrDevicePtr, pResult->devicePtr, rSq);

        GPUSync sync2;
        sync2.sync();

        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelCylindricalGridGetRadialIndexFromRSq( Grid, pResult, rSq);
#endif
        T value = pResult->v;

        delete pResult;
        return value;
    }

    TEST( getRadialIndexFromRSq ) {
        //gpuReset();
        std::vector<double> Rverts = { 1.0, 2.0, 3.0 };
        std::vector<double> Zverts = { -10, 0, 10 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[R] = new GridBins_t();
        pGridInfo[Z] = new GridBins_t();
        pGridInfo[R]->initialize(  Rverts );
        pGridInfo[Z]->initialize(  Zverts );

        std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(2,pGridInfo));
#ifdef __CUDACC__
        pGrid->copyToGPU();

        GPUSync sync1;
        sync1.sync();

        gpuErrchk( cudaPeekAtLastError() );
#endif

        CHECK_EQUAL( 0, launchGetRadialIndexFromRSq( *pGrid, 0.5*0.5) );
        CHECK_EQUAL( 1, launchGetRadialIndexFromRSq( *pGrid, 1.5*1.5) );
        CHECK_EQUAL( 2, launchGetRadialIndexFromRSq( *pGrid, 2.5*2.5) );
        CHECK_EQUAL( 3, launchGetRadialIndexFromRSq( *pGrid, 3.5*3.5) );
        CHECK_EQUAL( 3, launchGetRadialIndexFromRSq( *pGrid, 30.5*30.5) );

        delete pGridInfo[R];
        delete pGridInfo[Z];
    }

    // kernal call
#ifdef __CUDACC__
    CUDA_CALLABLE_KERNEL void kernelCylindricalGridGetAxialIndex(Grid_t** pGrid, resultClass<int>* pResult, gpuFloatType_t z) {
        pResult->v = (*pGrid)->getAxialIndex(z);
    }
#else
    void kernelCylindricalGridGetAxialIndex(Grid_t Grid, resultClass<int>* pResult, gpuFloatType_t z) {
        pResult->v = Grid.getAxialIndex(z);
    }
#endif

    int launchGetAxialIndex(Grid_t& Grid, gpuFloatType_t z ) {
        using T = int;

        resultClass<T>* pResult = new resultClass<T>();
#ifdef __CUDACC__
        pResult->copyToGPU();

        GPUSync sync1;
        sync1.sync();

        gpuErrchk( cudaPeekAtLastError() );

        kernelCylindricalGridGetAxialIndex<<<1,1>>>( Grid.ptrDevicePtr, pResult->devicePtr, z);

        GPUSync sync2;
        sync2.sync();

        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelCylindricalGridGetAxialIndex( Grid, pResult, z);
#endif
        T value = pResult->v;

        delete pResult;
        return value;
    }

    TEST( getAxialIndex ) {
        //gpuReset();
        std::vector<double> Rverts = { 1.0, 2.0, 3.0 };
        std::vector<double> Zverts = { -10, 0, 10 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[R] = new GridBins_t();
        pGridInfo[Z] = new GridBins_t();
        pGridInfo[R]->initialize(  Rverts );
        pGridInfo[Z]->initialize(  Zverts );

        std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(2,pGridInfo));
#ifdef __CUDACC__
        pGrid->copyToGPU();

        GPUSync sync1;
        sync1.sync();

        gpuErrchk( cudaPeekAtLastError() );
#endif

        //printf( "Debug: devicePtr = %d\n", devicePtr );

        CHECK_EQUAL( -1, launchGetAxialIndex( *pGrid, -100.5 ));
        CHECK_EQUAL( -1, launchGetAxialIndex( *pGrid,  -10.5 ));
        CHECK_EQUAL(  0, launchGetAxialIndex( *pGrid,   -9.5 ));
        CHECK_EQUAL(  1, launchGetAxialIndex( *pGrid,    9.5 ));
        CHECK_EQUAL(  2, launchGetAxialIndex( *pGrid,   10.5 ));
        CHECK_EQUAL(  2, launchGetAxialIndex( *pGrid,  100.5 ));

        delete pGridInfo[R];
        delete pGridInfo[Z];
    }

    // kernal call
#ifdef __CUDACC__
    CUDA_CALLABLE_KERNEL void kernelCylindricalIsIndexOutside(Grid_t** pGrid, resultClass<bool>* pResult, unsigned d, int i) {
        pResult->v = (*pGrid)->isIndexOutside(d,i);
    }
#else
    void kernelCylindricalIsIndexOutside(Grid_t Grid, resultClass<bool>* pResult, unsigned d, int i) {
        pResult->v = Grid.isIndexOutside(d,i);
    }
#endif

    inline bool launchIsIndexOutside(Grid_t& Grid, unsigned d, int i ) {
        resultClass<bool>* pResult = new resultClass<bool>();
#ifdef __CUDACC__
        pResult->copyToGPU();

        GPUSync sync1;
        sync1.sync();

        gpuErrchk( cudaPeekAtLastError() );

        kernelCylindricalIsIndexOutside<<<1,1>>>( Grid.ptrDevicePtr, pResult->devicePtr, d, i);

        GPUSync sync2;
        sync2.sync();

        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelCylindricalIsIndexOutside( Grid, pResult, d, i);
#endif

        bool value = pResult->v;

        delete pResult;
        return value;
    }

    TEST( isIndexOutside_R ) {
        //gpuReset();
        std::vector<double> Rverts = { 1.0, 2.0, 3.0 };
        std::vector<double> Zverts = { -10, 0, 10 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[R] = new GridBins_t();
        pGridInfo[Z] = new GridBins_t();
        pGridInfo[R]->initialize(  Rverts );
        pGridInfo[Z]->initialize(  Zverts );

        std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(2,pGridInfo));
#ifdef __CUDACC__
        pGrid->copyToGPU();

        GPUSync sync1;
        sync1.sync();

        gpuErrchk( cudaPeekAtLastError() );
#endif

        //printf( "Debug: devicePtr = %d\n", devicePtr );

        CHECK_EQUAL( false, launchIsIndexOutside( *pGrid, R, 0) );
        CHECK_EQUAL( false, launchIsIndexOutside( *pGrid, R, 1) );
        CHECK_EQUAL( false, launchIsIndexOutside( *pGrid, R, 2) );
        CHECK_EQUAL( true,  launchIsIndexOutside( *pGrid, R, 3) );

        delete pGridInfo[R];
        delete pGridInfo[Z];
    }

    // kernel call
#ifdef __CUDACC__
    CUDA_CALLABLE_KERNEL void kernelCylindricalGridGetIndex(Grid_t** pGrid, resultClass<unsigned>* pResult, Position_t pos) {
        //printf("Debug: kernelCylindricalGridGetIndex -- calling pGrid->getIndex(pos)\n");
        unsigned index = (*pGrid)->getIndex(pos);
        pResult->v = index;
    }
#else
    void kernelCylindricalGridGetIndex(Grid_t Grid, resultClass<unsigned>* pResult, Position_t pos) {
        unsigned index = Grid.getIndex(pos);
        pResult->v = index;
    }
#endif


    inline unsigned launchGetIndex(Grid_t& Grid, Position_t pos ) {
        resultClass<unsigned>* pResult = new resultClass<unsigned>();
#ifdef __CUDACC__
        pResult->copyToGPU();

        GPUSync sync1;
        sync1.sync();

        gpuErrchk( cudaPeekAtLastError() );

        kernelCylindricalGridGetIndex<<<1,1>>>( Grid.ptrDevicePtr, pResult->devicePtr, pos);

        GPUSync sync2;
        sync2.sync();

        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelCylindricalGridGetIndex( Grid, pResult, pos);
#endif

        unsigned value = pResult->v;

        delete pResult;
        return value;
    }

    TEST( getIndex ) {
        //gpuReset();
        std::vector<double> Rverts = { 1.0, 2.0, 3.0 };
        std::vector<double> Zverts = { -10, 0, 10 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[R] = new GridBins_t();
        pGridInfo[Z] = new GridBins_t();
        pGridInfo[R]->initialize(  Rverts );
        pGridInfo[Z]->initialize(  Zverts );

        std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(2,pGridInfo));
#ifdef __CUDACC__
        pGrid->copyToGPU();

        GPUSync sync1;
        sync1.sync();

        gpuErrchk( cudaPeekAtLastError() );
#endif

        CHECK_EQUAL( 0, launchGetIndex( *pGrid,            Position_t(0.5,  0.0, -9.5) ) );
        CHECK_EQUAL( 1, launchGetIndex( *pGrid,            Position_t(1.5,  0.0, -9.5) ) );
        CHECK_EQUAL( 2, launchGetIndex( *pGrid,            Position_t(2.5,  0.0, -9.5) ) );
        CHECK_EQUAL( OUTSIDE_MESH, launchGetIndex( *pGrid, Position_t(3.5,  0.0, -9.5) ) );

        CHECK_EQUAL( 3, launchGetIndex( *pGrid,            Position_t(0.5,  0.0, 9.5) ) );
        CHECK_EQUAL( 4, launchGetIndex( *pGrid,            Position_t(1.5,  0.0, 9.5) ) );
        CHECK_EQUAL( 5, launchGetIndex( *pGrid,            Position_t(2.5,  0.0, 9.5) ) );
        CHECK_EQUAL( OUTSIDE_MESH, launchGetIndex( *pGrid, Position_t(3.5,  0.0, 9.5) ) );

        CHECK_EQUAL( 3, launchGetIndex( *pGrid,            Position_t(0.0,  0.5,  9.5) ) );
        CHECK_EQUAL( 4, launchGetIndex( *pGrid,            Position_t(0.0,  1.5,  9.5) ) );
        CHECK_EQUAL( 5, launchGetIndex( *pGrid,            Position_t(0.0,  2.5,  9.5) ) );
        CHECK_EQUAL( OUTSIDE_MESH, launchGetIndex( *pGrid, Position_t(0.0,  3.5,  9.5) ) );

        CHECK_EQUAL( OUTSIDE_MESH, launchGetIndex( *pGrid, Position_t(0.0,  0.5,  10.5) ) );
        CHECK_EQUAL( OUTSIDE_MESH, launchGetIndex( *pGrid, Position_t(0.0,  1.5,  10.5) ) );
        CHECK_EQUAL( OUTSIDE_MESH, launchGetIndex( *pGrid, Position_t(0.0,  2.5,  10.5) ) );
        CHECK_EQUAL( OUTSIDE_MESH, launchGetIndex( *pGrid, Position_t(0.0,  3.5,  10.5) ) );

        delete pGridInfo[R];
        delete pGridInfo[Z];
    }

    // kernel call
#ifdef __CUDACC__
    CUDA_CALLABLE_KERNEL void kernelCylindricalIsOutside(Grid_t** pGrid, resultClass<bool>* pResult, const int i, const int j, const int k) {
        //printf("Debug: kernelCylindricalIsOutside -- calling pGrid->isOutside(pos)\n");
        int indices[] = {i,j,k};
        pResult->v = (*pGrid)->isOutside(indices);
    }
#else
    void kernelCylindricalIsOutside(Grid_t Grid, resultClass<bool>* pResult, const int i, const int j, const int k) {
        int indices[] = {i,j,k};
        pResult->v = Grid.isOutside(indices);
    }
#endif


    inline bool launchIsOutside(Grid_t& Grid, const int i, const int j, const int k ) {
        resultClass<bool>* pResult = new resultClass<bool>();
#ifdef __CUDACC__
        pResult->copyToGPU();

        GPUSync sync1;
        sync1.sync();

        gpuErrchk( cudaPeekAtLastError() );

        kernelCylindricalIsOutside<<<1,1>>>( Grid.ptrDevicePtr, pResult->devicePtr, i, j, k);

        GPUSync sync2;
        sync2.sync();

        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelCylindricalIsOutside( Grid, pResult, i, j, k);
#endif

        bool value = pResult->v;

        delete pResult;
        return value;
    }

    TEST( isOutside ) {
        //gpuReset();
        std::vector<double> Rverts = { 1.0, 2.0, 3.0 };
        std::vector<double> Zverts = { -10, 0, 10 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[R] = new GridBins_t();
        pGridInfo[Z] = new GridBins_t();
        pGridInfo[R]->initialize(  Rverts );
        pGridInfo[Z]->initialize(  Zverts );

        std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(2,pGridInfo));
#ifdef __CUDACC__
        pGrid->copyToGPU();

        GPUSync sync1;
        sync1.sync();

        gpuErrchk( cudaPeekAtLastError() );
#endif

        CHECK_EQUAL( true,  launchIsOutside( *pGrid, 3, 0, 0 ) );  // pos radius
        CHECK_EQUAL( false, launchIsOutside( *pGrid, 2, 0, 0 ) );  // radius is not outside
        CHECK_EQUAL( false, launchIsOutside( *pGrid, 0, 0, 0 ) );  // radius is not outside

        CHECK_EQUAL( true,  launchIsOutside( *pGrid, 0, -1, 0 ) );  // outside - neg Z
        CHECK_EQUAL( false, launchIsOutside( *pGrid, 0,  1, 0 ) );  // not outside in Z
        CHECK_EQUAL( true,  launchIsOutside( *pGrid, 0,  2, 0 ) );  // outside - pos Z

        delete pGridInfo[R];
        delete pGridInfo[Z];
    }

    // kernel call
#ifdef __CUDACC__
    CUDA_CALLABLE_KERNEL void kernelCylindricalCalcIJK(Grid_t** pGrid, resultClass<uint3>* pResult, unsigned index) {
        //printf("Debug: kernelCylindricalIsOutside -- calling pGrid->isOutside(pos)\n");
        pResult->v = (*pGrid)->calcIJK(index);
    }
#else
    void kernelCylindricalCalcIJK(Grid_t Grid, resultClass<uint3>* pResult, unsigned index) {
        pResult->v = Grid.calcIJK(index);
    }
#endif


    inline uint3 launchCalcIJK(Grid_t& Grid, unsigned index ) {
        resultClass<uint3>* pResult = new resultClass<uint3>();
#ifdef __CUDACC__
        pResult->copyToGPU();

        GPUSync sync1;
        sync1.sync();

        gpuErrchk( cudaPeekAtLastError() );

        kernelCylindricalCalcIJK<<<1,1>>>( Grid.ptrDevicePtr, pResult->devicePtr, index);

        GPUSync sync2;
        sync2.sync();

        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelCylindricalCalcIJK( Grid, pResult, index);
#endif

        uint3 value = pResult->v;

        delete pResult;
        return value;
    }

    TEST( calcIJK ) {
        //gpuReset();
        std::vector<double> Rverts = { 1.0, 2.0, 3.0 };
        std::vector<double> Zverts = { -10, 0, 10 };

        pArrayOfpGridInfo_t pGridInfo;
        pGridInfo[R] = new GridBins_t();
        pGridInfo[Z] = new GridBins_t();
        pGridInfo[R]->initialize(  Rverts );
        pGridInfo[Z]->initialize(  Zverts );

        std::unique_ptr<Grid_t> pGrid = std::unique_ptr<Grid_t>( new Grid_t(2,pGridInfo));
#ifdef __CUDACC__
        pGrid->copyToGPU();

        GPUSync sync1;
        sync1.sync();

        gpuErrchk( cudaPeekAtLastError() );
#endif

        uint3 indices = launchCalcIJK( *pGrid, 0);
        CHECK_EQUAL( 0, indices.x );
        CHECK_EQUAL( 0, indices.y );
        CHECK_EQUAL( 0, indices.z );

        indices = launchCalcIJK( *pGrid, 1);
        CHECK_EQUAL( 1, indices.x );
        CHECK_EQUAL( 0, indices.y );
        CHECK_EQUAL( 0, indices.z );

        indices = launchCalcIJK( *pGrid, 2);
        CHECK_EQUAL( 2, indices.x );
        CHECK_EQUAL( 0, indices.y );
        CHECK_EQUAL( 0, indices.z );

        indices = launchCalcIJK( *pGrid, 3);
        CHECK_EQUAL( 0, indices.x );
        CHECK_EQUAL( 1, indices.y );
        CHECK_EQUAL( 0, indices.z );

        indices = launchCalcIJK( *pGrid, 4);
        CHECK_EQUAL( 1, indices.x );
        CHECK_EQUAL( 1, indices.y );
        CHECK_EQUAL( 0, indices.z );

        indices = launchCalcIJK( *pGrid, 5);
        CHECK_EQUAL( 2, indices.x );
        CHECK_EQUAL( 1, indices.y );
        CHECK_EQUAL( 0, indices.z );

        delete pGridInfo[R];
        delete pGridInfo[Z];
    }
#endif
}

} // end namespace
