#ifndef MONTERAYMEMORY_TEST_HELPER_HH_
#define MONTERAYMEMORY_TEST_HELPER_HH_

#include <stdexcept>

#include "MonteRayConstants.hh"
#include "MonteRayManagedMemory.hh"
#include "MonteRayDefinitions.hh"

using namespace MonteRay;

class testClass : public ManagedMemoryBase {
public:
    CUDAHOST_CALLABLE_MEMBER testClass( unsigned num = 1, gpuFloatType_t mult = 1.0 ) : ManagedMemoryBase() {
        N = num;
        multiple = mult;
        elements = (gpuFloatType_t*) MonteRayHostAlloc( N * sizeof( gpuFloatType_t ));

        for( unsigned i=0; i<N; ++i) {
            elements[i] = 0.0;
        }
    }

    CUDAHOST_CALLABLE_MEMBER ~testClass(){
        MonteRayHostFree( elements, true );
    }

#ifdef __CUDACC__
    CUDAHOST_CALLABLE_MEMBER void copyToGPU(cudaStream_t* stream = nullptr, MonteRayGPUProps device = MonteRayGPUProps() ) {
        ManagedMemoryBase::copyToGPU( stream, device );
        cudaMemAdvise(elements,  N*sizeof(gpuFloatType_t), cudaMemAdviseSetReadMostly, device.deviceID);
        if( device.deviceProps->concurrentManagedAccess ) {
            if( stream ) {
                cudaMemPrefetchAsync(elements, N*sizeof(gpuFloatType_t), device.deviceID, *stream );
            } else {
                cudaMemPrefetchAsync(elements, N*sizeof(gpuFloatType_t), device.deviceID, NULL );
            }
        }
    }
#else
    void copyToGPU(void) {
        throw std::runtime_error( "Non-CUDA code - copyToGPU not valid without CUDA.");
    }
#endif

#ifdef __CUDACC__
    CUDAHOST_CALLABLE_MEMBER void copyToCPU(cudaStream_t* stream = nullptr) {
        ManagedMemoryBase::copyToCPU( stream );
        if( stream ) {
            cudaMemPrefetchAsync(elements, N*sizeof(gpuFloatType_t), cudaCpuDeviceId, *stream );
        } else {
            cudaMemPrefetchAsync(elements, N*sizeof(gpuFloatType_t), cudaCpuDeviceId, NULL );
        }
    }
#else
    void copyToCPU(void) {
        throw std::runtime_error( "copyToGPU not valid without CUDA.");
    }
#endif

    gpuFloatType_t multiple;
    unsigned N;
    gpuFloatType_t* elements;
};

class ManagedMemoryTestHelper
{
public:

    ManagedMemoryTestHelper();

    ~ManagedMemoryTestHelper();

    void setupTimers();

    void stopTimers();

    void launchSumVectors( testClass* A, testClass* B, testClass* C);

private:
#ifdef __CUDACC__
    cudaEvent_t start, stop;
#endif

};

#endif /* GPUDISTANCECALCULATOR_TEST_HELPER_HH_ */


