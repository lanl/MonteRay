#include <UnitTest++.h>

#include "GridBins.hh"
#include "HashBins.hh"
#include "MonteRay_SingleValueCopyMemory.t.hh"
#include "GPUUtilityFunctions.hh"

using namespace MonteRay;

SUITE( GridBins_gpu_Tester ) {
#ifdef __CUDACC__

    template<typename T>
    using resultClass = MonteRay_SingleValueCopyMemory<T>;

    CUDA_CALLABLE_KERNEL void kernelGetMaxNumVertices(GridBins* pGridBins, resultClass<unsigned>* pResult) {
        pResult->v = pGridBins->getMaxNumVertices();
    }

    CUDA_CALLABLE_KERNEL void kernelGetNumVertices(GridBins* pGridBins, unsigned dim, resultClass<unsigned>* pResult) {
        pResult->v = pGridBins->getNumVertices(dim);
    }

    CUDA_CALLABLE_KERNEL void kernelGetNumBins(GridBins* pGridBins, unsigned dim, resultClass<unsigned>* pResult) {
        pResult->v = pGridBins->getNumBins(dim);
    }

    CUDA_CALLABLE_KERNEL void kernelGetVertex(GridBins* pGridBins, unsigned dim, unsigned index, resultClass<gpuFloatType_t>* pResult) {
        pResult->v = pGridBins->getVertex(dim,index);
    }

    CUDA_CALLABLE_KERNEL void kernelIsRegular(GridBins* pGridBins, unsigned dim, resultClass<bool>* pResult) {
        pResult->v = pGridBins->isRegular(dim);
    }

    CUDA_CALLABLE_KERNEL void kernelGetOffset(GridBins* pGridBins, const unsigned dim, resultClass<unsigned>* pResult) {
        pResult->v = pGridBins->getOffset(dim);
    }

    CUDA_CALLABLE_KERNEL void kernelGetHashNEdges(GridBins* pGridBins, const unsigned dim, resultClass<unsigned>* pResult) {
        pResult->v = pGridBins->getHashPtr(dim)->getNEdges();
    }

    CUDA_CALLABLE_KERNEL void kernelGetDimIndex(GridBins* pGridBins, const unsigned dim, gpuFloatType_t pos, resultClass<unsigned>* pResult) {
        pResult->v = pGridBins->getDimIndex(dim, pos);
    }


    CUDA_CALLABLE_KERNEL void kernelGetHashLowerUpperBins(GridBins* pGridBins, const unsigned dim, gpuFloatType_t pos, resultClass<unsigned>* pResult1, resultClass<unsigned>* pResult2) {
        pGridBins->getHashLowerUpperBins(dim, pos, pResult1->v, pResult2->v);
    }

    class GridBinsGPUHelper{
    public:
        GridBinsGPUHelper(){
            pGridBins = std::unique_ptr<GridBins>( new GridBins() );
        }
        ~GridBinsGPUHelper(){}

        void copyToGPU(){
            pGridBins->copyToGPU();
        }

        unsigned getMaxNumVertices() const {
            using result_t = resultClass<unsigned>;
            std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
            pResult->copyToGPU();

            kernelGetMaxNumVertices<<<1,1>>>( pGridBins->devicePtr, pResult->devicePtr);
            gpuErrchk( cudaPeekAtLastError() );
            pResult->copyToCPU();
            return pResult->v;
        }

        unsigned getOffset( const unsigned dim ) const {
            using result_t = resultClass<unsigned>;
            std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
            pResult->copyToGPU();

            kernelGetOffset<<<1,1>>>( pGridBins->devicePtr, dim, pResult->devicePtr);
            gpuErrchk( cudaPeekAtLastError() );
            pResult->copyToCPU();
            return pResult->v;
        }

        unsigned getNumVertices( const unsigned dim ) const {
            using result_t = resultClass<unsigned>;
            std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
            pResult->copyToGPU();

            kernelGetNumVertices<<<1,1>>>( pGridBins->devicePtr, dim, pResult->devicePtr);
            gpuErrchk( cudaPeekAtLastError() );
            pResult->copyToCPU();
            return pResult->v;
        }

        unsigned getNumBins( const unsigned dim ) const {
            using result_t = resultClass<unsigned>;
            std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
            pResult->copyToGPU();

            kernelGetNumBins<<<1,1>>>( pGridBins->devicePtr, dim, pResult->devicePtr);
            gpuErrchk( cudaPeekAtLastError() );
            pResult->copyToCPU();
            return pResult->v;
        }

        gpuFloatType_t getVertex( const unsigned dim, const unsigned index ) const {
            using result_t = resultClass<gpuFloatType_t>;
            std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
            pResult->copyToGPU();

            kernelGetVertex<<<1,1>>>( pGridBins->devicePtr, dim, index, pResult->devicePtr);
            gpuErrchk( cudaPeekAtLastError() );
            pResult->copyToCPU();
            return pResult->v;
        }

        bool isRegular( const unsigned dim ) {
            using result_t = resultClass<bool>;
            std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
            pResult->copyToGPU();

            kernelIsRegular<<<1,1>>>( pGridBins->devicePtr, dim, pResult->devicePtr);
            gpuErrchk( cudaPeekAtLastError() );
            pResult->copyToCPU();
            return pResult->v;
        }

        void setVertices(const unsigned dim, gpuFloatType_t min, gpuFloatType_t max, unsigned nBins ) {
            pGridBins->setVertices( dim, min, max, nBins);
        }

        unsigned getHashNEdges( const unsigned dim) {
            using result_t = resultClass<unsigned>;
            std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
            pResult->copyToGPU();

            kernelGetHashNEdges<<<1,1>>>( pGridBins->devicePtr, dim, pResult->devicePtr);
            gpuErrchk( cudaPeekAtLastError() );
            pResult->copyToCPU();
            return pResult->v;
        }

        void setDefaultHashSize( const unsigned N) {
            pGridBins->setDefaultHashSize( N );
        }

        template<typename T>
        void setVertices( const unsigned dim, std::vector<T> vertices ) {
            pGridBins->setVertices<T>( dim, vertices );
        }

        unsigned getDimIndex( const unsigned dim, gpuFloatType_t pos ) const {
            using result_t = resultClass<unsigned>;
            std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
            pResult->copyToGPU();

            kernelGetDimIndex<<<1,1>>>( pGridBins->devicePtr, dim, pos, pResult->devicePtr);
            gpuErrchk( cudaPeekAtLastError() );
            pResult->copyToCPU();
            return pResult->v;
        }

        void getHashLowerUpperBins( const unsigned dim, gpuFloatType_t pos, unsigned& lower, unsigned& upper){
            using result_t = resultClass<unsigned>;

            std::unique_ptr<result_t> pResult1 = std::unique_ptr<result_t> ( new result_t() );
            std::unique_ptr<result_t> pResult2 = std::unique_ptr<result_t> ( new result_t() );

            pResult1->copyToGPU();
            pResult2->copyToGPU();

            kernelGetHashLowerUpperBins<<<1,1>>>( pGridBins->devicePtr, dim, pos, pResult1->devicePtr, pResult2->devicePtr);
            gpuErrchk( cudaPeekAtLastError() );

            pResult1->copyToCPU();
            pResult2->copyToCPU();

            lower = pResult1->v;
            upper = pResult2->v;
        }

        std::unique_ptr<GridBins> pGridBins;
    };


    TEST(setup) {
        //gpuReset();
    }

    TEST_FIXTURE(GridBinsGPUHelper, ctor_class_getMaxNumVertices ) {
        copyToGPU();
        CHECK_EQUAL( 1000, getMaxNumVertices() );
        CHECK_EQUAL( 0,      getOffset(0) );
        CHECK_EQUAL( 1000,   getOffset(1) );
        CHECK_EQUAL( 1000*2, getOffset(2) );
    }

    TEST_FIXTURE(GridBinsGPUHelper, class_setVertices ) {

        setVertices(0,  0.0, 10.0, 10);
        setVertices(1, 20.0, 30.0, 10);
        setVertices(2, 40.0, 50.0, 10);
        copyToGPU();

        CHECK_EQUAL(11, getNumVertices(0) );
        CHECK_EQUAL(10, getNumBins(0) );

        CHECK_EQUAL(11, getNumVertices(1) );
        CHECK_EQUAL(10, getNumBins(1) );

        CHECK_EQUAL(11, getNumVertices(2) );
        CHECK_EQUAL(10, getNumBins(2) );

        CHECK_CLOSE( 0.0, getVertex(0, 0), 1e-11 );
        CHECK_CLOSE( 1.0, getVertex(0, 1), 1e-11 );
        CHECK_CLOSE( 10.0, getVertex(0, 10), 1e-11 );

        CHECK_CLOSE( 20.0, getVertex(1, 0), 1e-11 );
        CHECK_CLOSE( 21.0, getVertex(1, 1), 1e-11 );
        CHECK_CLOSE( 30.0, getVertex(1, 10), 1e-11 );

        CHECK_CLOSE( 40.0, getVertex(2, 0), 1e-11 );
        CHECK_CLOSE( 41.0, getVertex(2, 1), 1e-11 );
        CHECK_CLOSE( 50.0, getVertex(2, 10), 1e-11 );

        CHECK_EQUAL( true, isRegular(0) );
        CHECK_EQUAL( true, isRegular(1) );
        CHECK_EQUAL( true, isRegular(2) );
    }

    TEST_FIXTURE(GridBinsGPUHelper, hash_setup ) {
        setVertices(0,  0.0, 10.0, 10);
        copyToGPU();

        CHECK_EQUAL( 8000, getHashNEdges(0) );
    }

    TEST_FIXTURE(GridBinsGPUHelper, hash_getHashLowerUpperBins_and_getDimIndex ) {
        setDefaultHashSize( 201 );
        std::vector<gpuFloatType_t> vertices {-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
            0,  1,  2,  3.1,  4,  5,  6,  7,  8,  9,  10};

        setVertices(0, vertices );
        copyToGPU();

        unsigned lower_bin;
        unsigned upper_bin;

        gpuFloatType_t pos = -10.5;
        CHECK_EQUAL( -1, getDimIndex( 0, pos) );

        pos = -9.5;
        getHashLowerUpperBins(0, pos, lower_bin, upper_bin);
        CHECK_EQUAL( 0, getDimIndex( 0, pos) );
        CHECK_EQUAL( 0, lower_bin);
        CHECK_EQUAL( 0, upper_bin );

        pos = -8.5;
        getHashLowerUpperBins(0, pos, lower_bin, upper_bin);
        CHECK_EQUAL( 1, getDimIndex( 0, pos) );
        CHECK_EQUAL( 1, lower_bin);
        CHECK_EQUAL( 1, upper_bin );

        pos = 8.5;
        getHashLowerUpperBins(0, pos, lower_bin, upper_bin);
        CHECK_EQUAL( 18, getDimIndex( 0, pos) );
        CHECK_EQUAL( 18, lower_bin);
        CHECK_EQUAL( 18, upper_bin );

        pos = 9.5;
        getHashLowerUpperBins(0, pos, lower_bin, upper_bin);
        CHECK_EQUAL( 19, getDimIndex( 0, pos) );
        CHECK_EQUAL( 19, lower_bin);
        CHECK_EQUAL( 19, upper_bin );

        pos = 10.5;
        CHECK_EQUAL( 20, getDimIndex( 0, pos) );
    }

    TEST_FIXTURE(GridBinsGPUHelper, hash_getHashLowerUpperBins_and_getDimIndex_coarse ) {
        setDefaultHashSize( 5 );
        std::vector<gpuFloatType_t> vertices {-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
            0,  1,  2,  3.1,  4,  5,  6,  7,  8,  9,  10};

        setVertices(0, vertices );
        copyToGPU();

        unsigned lower_bin;
        unsigned upper_bin;

        gpuFloatType_t pos = -10.5;
        CHECK_EQUAL( -1, getDimIndex( 0, pos) );

        pos = -9.5;
        getHashLowerUpperBins(0, pos, lower_bin, upper_bin);
        CHECK_EQUAL( 0, getDimIndex( 0, pos) );
        CHECK_EQUAL( 0, lower_bin);
        CHECK_EQUAL( 5, upper_bin );

        pos = -8.5;
        getHashLowerUpperBins(0, pos, lower_bin, upper_bin);
        CHECK_EQUAL( 1, getDimIndex( 0, pos) );
        CHECK_EQUAL( 0, lower_bin);
        CHECK_EQUAL( 5, upper_bin );

        pos = 8.5;
        getHashLowerUpperBins(0, pos, lower_bin, upper_bin);
        CHECK_EQUAL( 18, getDimIndex( 0, pos) );
        CHECK_EQUAL( 15, lower_bin);
        CHECK_EQUAL( 20, upper_bin );

        pos = 9.5;
        getHashLowerUpperBins(0, pos, lower_bin, upper_bin);
        CHECK_EQUAL( 19, getDimIndex( 0, pos) );
        CHECK_EQUAL( 15, lower_bin);
        CHECK_EQUAL( 20, upper_bin );

        pos = 10.5;
        CHECK_EQUAL( 20, getDimIndex( 0, pos) );
    }

    TEST( cleanup ) {
        //gpuReset();
    }
#endif
}
