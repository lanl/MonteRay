#ifndef COLLISIONPOINTS_TEST_HELPER_HH_
#define COLLISIONPOINTS_TEST_HELPER_HH_

#include "global.h"
#include "/projects/opt/centos7/cuda/7.5/include/driver_types.h"

#ifdef CUDA
#include <cuda.h>
#endif

using namespace MonteRay;

class CollisionPointsTester
{
public:
	typedef global::float_t float_t;

	CollisionPointsTester();

	~CollisionPointsTester();

	void setupTimers();

	void stopTimers();

	CollisionPointsSize_t launchGetCapacity( unsigned nBlocks, unsigned nThreads, CollisionPointsHost& CPs);
	gpuFloatType_t launchTestSumEnergy( unsigned nBlocks, unsigned nThreads, CollisionPointsHost& CPs);

private:
	cudaEvent_t start, stop;

};
#endif /* COLLISIONPOINTS_TEST_HELPER_HH_ */


