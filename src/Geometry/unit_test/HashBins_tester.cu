#include <UnitTest++.h>

#include "HashBins.hh"
#include "MonteRay_SingleValueCopyMemory.t.hh"
#include "GPUUtilityFunctions.hh"

using namespace MonteRay;

SUITE( HashBins_Tester ) {

    TEST( ctor ) {
        //gpuReset();

        unsigned N= 10;
        gpuFloatType_t values[N];

        values[0] = 0.0;
        values[1] = 1.5;
        values[2] = 2.0;
        values[3] = 2.5;
        values[4] = 4.0;
        values[5] = 6.0;
        values[6] = 6.1;
        values[7] = 6.15;
        values[8] = 8.0;
        values[9] = 10.0;

        HashBins hash(values,N);
        CHECK(true);
    }

    TEST( min_max ) {
        unsigned N= 10;
        gpuFloatType_t values[N];

        values[0] = -1.0;
        values[1] = -0.5;
        values[2] = 0.0;
        values[3] = 1.5;
        values[4] = 4.0;
        values[5] = 6.0;
        values[6] = 6.1;
        values[7] = 6.15;
        values[8] = 8.0;
        values[9] = 10.0;

        HashBins hash(values,N,12);
        CHECK_CLOSE( -1.0, hash.getMin(), 1e-6 );
        CHECK_CLOSE( 10.0, hash.getMax(), 1e-6 );
        CHECK_CLOSE( 1.0, hash.getDelta(), 1e-6 );
    }

    TEST( getBinEdge ) {
        unsigned N= 10;
        gpuFloatType_t values[N];

        values[0] = -1.0;
        values[1] = -0.5;
        values[2] = 0.0;
        values[3] = 1.5;
        values[4] = 4.0;
        values[5] = 6.0;
        values[6] = 6.1;
        values[7] = 6.15;
        values[8] = 8.0;
        values[9] = 10.0;

        HashBins hash(values,N,12);
        CHECK_CLOSE( -1.0, hash.getBinEdge(0), 1e-6 );
        CHECK_CLOSE(  0.0, hash.getBinEdge(1), 1e-6 );
        CHECK_CLOSE(  1.0, hash.getBinEdge(2), 1e-6 );
        CHECK_CLOSE(  2.0, hash.getBinEdge(3), 1e-6 );
        CHECK_CLOSE(  9.0, hash.getBinEdge(10), 1e-6 );
        CHECK_CLOSE( 10.0, hash.getBinEdge(11), 1e-6 );
    }

    TEST( getBinBound ) {
        unsigned N= 10;
        gpuFloatType_t values[N];

        values[0] = -1.0;
        values[1] = -0.5;
        values[2] = 0.1;
        values[3] = 0.2;
        values[4] = 2.0;
        values[5] = 6.0;
        values[6] = 6.1;
        values[7] = 6.15;
        values[8] = 8.0;
        values[9] = 10.0;

        HashBins hash(values,N,12);

        CHECK_CLOSE( -1.0, hash.getBinEdge(0), 1e-6 );
        CHECK_EQUAL( 0, hash.getBinBound(0) );

        CHECK_CLOSE( 0.0, hash.getBinEdge(1), 1e-6 );
        CHECK_EQUAL( 1, hash.getBinBound(1) );

        CHECK_CLOSE( 1.0, hash.getBinEdge(2), 1e-6 );
        CHECK_EQUAL( 3, hash.getBinBound(2) );
    }

    TEST( getLowerUpperBins1 ) {
        unsigned N= 10;
        gpuFloatType_t values[N];

        values[0] = -1.0;
        values[1] = -0.5;
        values[2] = 0.1;
        values[3] = 0.2;
        values[4] = 3.0;
        values[5] = 6.0;
        values[6] = 6.1;
        values[7] = 6.15;
        values[8] = 8.0;
        values[9] = 10.0;

        HashBins hash(values,N,12);

        unsigned lower_bin;
        unsigned upper_bin;

        hash.getLowerUpperBins(-0.5, lower_bin, upper_bin);
        CHECK_EQUAL( 0, lower_bin );
        CHECK_EQUAL( 1, upper_bin );

        hash.getLowerUpperBins(0.0, lower_bin, upper_bin);
        CHECK_EQUAL( 1, lower_bin );
        CHECK_EQUAL( 3, upper_bin );

        hash.getLowerUpperBins(1.0, lower_bin, upper_bin);
        CHECK_EQUAL( 3, lower_bin );
        CHECK_EQUAL( 3, upper_bin );

        hash.getLowerUpperBins(9.9, lower_bin, upper_bin);
        CHECK_EQUAL( 8, lower_bin );
        CHECK_EQUAL( 9, upper_bin );
    }

    template<typename T>
    using resultClass = MonteRay_SingleValueCopyMemory<T>;

    CUDA_CALLABLE_KERNEL void kernelGetNEdges(HashBins* pHashBins, resultClass<unsigned>* pResult) {
        pResult->v = pHashBins->getNEdges();
    }


    CUDA_CALLABLE_KERNEL void kernelGetLowerUpperBins(HashBins* pHashBins, gpuFloatType_t value, resultClass<unsigned>* pLower, resultClass<unsigned>* pUpper) {
        pHashBins->getLowerUpperBins(value, pLower->v, pUpper->v);
    }

    class HashBinGPUhelper {
    public:
        HashBinGPUhelper(gpuFloatType_t *vertices, unsigned nVertices, unsigned nHashBinEdges = 8000){
            pHashBins = std::unique_ptr<HashBins>( new HashBins(vertices, nVertices, nHashBinEdges) );
        }

        ~HashBinGPUhelper(){}

        void copyToGPU(){
            pHashBins->copyToGPU();
        }

        unsigned getNEdges() {
            using result_t = resultClass<unsigned>;
            std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );

#ifdef __CUDACC__
            pResult->copyToGPU();
            kernelGetNEdges<<<1,1>>>( pHashBins->devicePtr, pResult->devicePtr);
            gpuErrchk( cudaPeekAtLastError() );
            pResult->copyToCPU();
#else
            kernelGetNEdges( pHashBins.get(), pResult.get());
#endif

            return pResult->v;
        }


        void getLowerUpperBins(gpuFloatType_t value, unsigned& lower, unsigned& upper) {
            using result_t = resultClass<unsigned>;
            std::unique_ptr<result_t> pResult1 = std::unique_ptr<result_t> ( new result_t() );
            std::unique_ptr<result_t> pResult2 = std::unique_ptr<result_t> ( new result_t() );
            pResult1->copyToGPU();
            pResult2->copyToGPU();

#ifdef __CUDACC__
            kernelGetLowerUpperBins<<<1,1>>>( pHashBins->devicePtr, value, pResult1->devicePtr, pResult2->devicePtr);
            gpuErrchk( cudaPeekAtLastError() );
#else
            kernelGetLowerUpperBins( pHashBins.get(), value, pResult1.get(), pResult2.get());
#endif

            pResult1->copyToCPU();
            pResult2->copyToCPU();

            lower = pResult1->v;
            upper = pResult2->v;
        }
        std::unique_ptr<HashBins> pHashBins;
    };

    TEST( device_getNEdges ) {
        unsigned N= 10;
        gpuFloatType_t values[N];

        values[0] = -1.0;
        values[1] = -0.5;
        values[2] = 0.1;
        values[3] = 0.2;
        values[4] = 3.0;
        values[5] = 6.0;
        values[6] = 6.1;
        values[7] = 6.15;
        values[8] = 8.0;
        values[9] = 10.0;

        HashBinGPUhelper hash(values,N,12);
        hash.copyToGPU();

        CHECK_EQUAL(12, hash.getNEdges() );
    }

    TEST( device_getLowerUpperBins1 ) {
        unsigned N= 10;
        gpuFloatType_t values[N];

        values[0] = -1.0;
        values[1] = -0.5;
        values[2] = 0.1;
        values[3] = 0.2;
        values[4] = 3.0;
        values[5] = 6.0;
        values[6] = 6.1;
        values[7] = 6.15;
        values[8] = 8.0;
        values[9] = 10.0;

        HashBinGPUhelper hash(values,N,12);
        hash.copyToGPU();

        unsigned lower_bin;
        unsigned upper_bin;

        hash.getLowerUpperBins(-0.5, lower_bin, upper_bin);
        CHECK_EQUAL( 0, lower_bin );
        CHECK_EQUAL( 1, upper_bin );

        hash.getLowerUpperBins(0.0, lower_bin, upper_bin);
        CHECK_EQUAL( 1, lower_bin );
        CHECK_EQUAL( 3, upper_bin );

        hash.getLowerUpperBins(1.0, lower_bin, upper_bin);
        CHECK_EQUAL( 3, lower_bin );
        CHECK_EQUAL( 3, upper_bin );

        hash.getLowerUpperBins(9.9, lower_bin, upper_bin);
        CHECK_EQUAL( 8, lower_bin );
        CHECK_EQUAL( 9, upper_bin );
    }

    TEST( cleanup ) {
        //gpuReset();
    }


}
