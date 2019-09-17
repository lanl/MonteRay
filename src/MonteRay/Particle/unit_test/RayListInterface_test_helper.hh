#ifndef RAYLISTINTERFACE_TEST_HELPER_HH_
#define RAYLISTINTERFACE_TEST_HELPER_HH_

#include "RayListInterface.hh"
#include "MonteRayConstants.hh"

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
#ifdef __CUDACC__
    cudaEvent_t start, stop;
#endif

};

} // end namespace
#endif /* RAYLISTINTERFACE_TEST_HELPER_HH_ */


