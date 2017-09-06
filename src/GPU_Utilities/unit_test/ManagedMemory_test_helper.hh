#ifndef MONTERAYMEMORY_TEST_HELPER_HH_
#define MONTERAYMEMORY_TEST_HELPER_HH_

#include "MonteRayConstants.hh"

#include "MonteRayManagedMemory.hh"
#include <stdexcept>

using namespace MonteRay;

class testClass : public ManagedMemoryBase {
public:
	testClass( unsigned num = 1, double mult = 1.0 ) : ManagedMemoryBase() {
		N = num;
		multiple = mult;
		elements = (gpuFloatType_t*) MonteRayHostAlloc( N * sizeof( gpuFloatType_t ));

		for( unsigned i=0; i<N; ++i) {
			elements[i] = 0.0;
		}
	}

	~testClass(){
		MonteRayHostFree( elements );
	}

#ifdef __CUDACC__
	void copyToGPU(cudaStream_t stream = NULL, MonteRayGPUProps device = MonteRayGPUProps() ) {
		ManagedMemoryBase::copyToGPU( stream, device );
		cudaMemAdvise(elements,  N*sizeof(double), cudaMemAdviseSetReadMostly, device.deviceID);
		if( device.deviceProps.concurrentManagedAccess ) {
			cudaMemPrefetchAsync(elements, N*sizeof(double), device.deviceID, stream );
		}
	}
#else
	void copyToGPU(void) {
		throw std::runtime_error( "copyToGPU not valid without CUDA.");
	}
#endif

#ifdef __CUDACC__
	void copyToCPU(cudaStream_t stream = NULL) {
		ManagedMemoryBase::copyToCPU( stream );
		cudaMemPrefetchAsync(elements, N*sizeof(double), cudaCpuDeviceId, stream );
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
	cudaEvent_t start, stop;

};

#endif /* GPUDISTANCECALCULATOR_TEST_HELPER_HH_ */


