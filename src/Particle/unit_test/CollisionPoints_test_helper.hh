#ifndef COLLISIONPOINTS_TEST_HELPER_HH_
#define COLLISIONPOINTS_TEST_HELPER_HH_

#include "MonteRayConstants.hh"
#include "driver_types.h" // cuda driver types

#ifdef CUDA
#include <cuda.h>
#endif

#include "CollisionPoints.h"

namespace MonteRay {

class CollisionPointsTester
{
public:

	CollisionPointsTester();

	~CollisionPointsTester();

	void setupTimers();

	void stopTimers();

	CollisionPointsHost::CollisionPointsSize_t launchGetCapacity( unsigned nBlocks, unsigned nThreads, CollisionPointsHost& CPs);
	gpuFloatType_t launchTestSumEnergy( unsigned nBlocks, unsigned nThreads, CollisionPointsHost& CPs);

private:
	cudaEvent_t start, stop;

};

} // end namespace
#endif /* COLLISIONPOINTS_TEST_HELPER_HH_ */

