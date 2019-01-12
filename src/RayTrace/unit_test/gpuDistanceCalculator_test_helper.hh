#ifndef GPUDISTANCECALCULATOR_TEST_HELPER_HH_
#define GPUDISTANCECALCULATOR_TEST_HELPER_HH_

#include <memory>

#include "MonteRayDefinitions.hh"
#include "MonteRayConstants.hh"
#include "GridBins.hh"
#include "MonteRay_timer.hh"

namespace MonteRay{

class RayWorkInfo;

class gpuDistanceCalculatorTestHelper
{
public:

    gpuDistanceCalculatorTestHelper();

    ~gpuDistanceCalculatorTestHelper();

    void gpuCheck();

    void setupTimers();

    void stopTimers();

    //	void launchGetDistancesToAllCenters( unsigned nBlocks, unsigned nThreads, const Position_t& pos);
    void launchRayTrace( const Position_t& pos, const Direction_t& dir, gpuRayFloat_t distance, bool);

    void copyGridtoGPU( GridBins* );



    void copyDistancesFromGPU( gpuRayFloat_t* distance );
    void copyCellsFromCPU( int* cells );
    unsigned getNumCrossingsFromGPU(void);

    std::unique_ptr< RayWorkInfo > pRayInfo;
private:
    GridBins* grid_device;


    unsigned nCells;

#ifdef __CUDACC__
    cudaEvent_t start, stop;
#else
    cpuTimer timer;
#endif

};

}
#endif /* GPUDISTANCECALCULATOR_TEST_HELPER_HH_ */

