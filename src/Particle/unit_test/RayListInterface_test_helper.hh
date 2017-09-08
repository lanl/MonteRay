#ifndef RAYLISTINTERFACE_TEST_HELPER_HH_
#define RAYLISTINTERFACE_TEST_HELPER_HH_

#include "MonteRayConstants.hh"
#include "driver_types.h" // cuda driver types

#ifdef CUDA
#include <cuda.h>
#endif

#include "RayListInterface.hh"

namespace MonteRay {

template< unsigned N = 1>
class RayListInterfaceTester
{
public:

	RayListInterfaceTester();

	~RayListInterfaceTester();

	void setupTimers();

	void stopTimers();

	MonteRay::RayListSize_t launchGetCapacity( unsigned nBlocks, unsigned nThreads, RayListInterface<N>& CPs);
	gpuFloatType_t launchTestSumEnergy( unsigned nBlocks, unsigned nThreads, RayListInterface<N>& CPs);

private:
	cudaEvent_t start, stop;

};

} // end namespace
#endif /* RAYLISTINTERFACE_TEST_HELPER_HH_ */


