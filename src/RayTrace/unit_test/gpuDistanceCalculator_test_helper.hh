#ifndef GPUDISTANCECALCULATOR_TEST_HELPER_HH_
#define GPUDISTANCECALCULATOR_TEST_HELPER_HH_

#include "MonteRayConstants.hh"
#include "GridBins.h"

#ifdef CUDA
#include <cuda.h>
#include "driver_types.h" // cuda driver types
#include "gpuRayTrace.h"
#endif

namespace MonteRay{

class gpuDistanceCalculatorTestHelper
{
public:

	gpuDistanceCalculatorTestHelper();

	~gpuDistanceCalculatorTestHelper();

	void gpuCheck();

	void setupTimers();

	void stopTimers();

	void launchGetDistancesToAllCenters( unsigned nBlocks, unsigned nThreads, const Position_t& pos);
	void launchRayTrace( const Position_t& pos, const Direction_t& dir, float_t distance, bool);

	void copyGridtoGPU( GridBins* );
	void copyDistancesFromGPU( float_t* );
	void copyCellsFromCPU( int* );
	unsigned getNumCrossingsFromGPU(void);

private:
	void* grid_device;
	void* distances_device;
	void* cells_device;
	void* numCrossings_device;

	unsigned nCells;

	cudaEvent_t start, stop;

};

}
#endif /* GPUDISTANCECALCULATOR_TEST_HELPER_HH_ */

