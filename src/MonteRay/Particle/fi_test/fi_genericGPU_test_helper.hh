#ifndef FI_TEST_GENERICGPU_TEST_HELPER_HH_
#define FI_TEST_GENERICGPU_TEST_HELPER_HH_

#include "MonteRayConstants.hh"
#include "MonteRayDefinitions.hh"

#include "RayListInterface.hh"
#include "CrossSection.hh"
#include "MaterialList.hh"
#include "MaterialProperties.hh"
#include "GridBins.hh"

#ifndef __CUDACC__
#include "MonteRay_timer.hh"
#endif

using namespace MonteRay;

template<unsigned N = 1>
class FIGenericGPUTestHelper
{
public:

    FIGenericGPUTestHelper(unsigned nCells);

    ~FIGenericGPUTestHelper();

    void setupTimers();

    void stopTimers();

    void copyGridtoGPU(GridBins* );

    void launchTallyCrossSection(unsigned nBlocks, unsigned nThreads, const RayListInterface<N>* pCP, const CrossSection* pXS );
    void launchTallyCrossSection(unsigned nBlocks, unsigned nThreads, const RayListInterface<N>* pCP, const MaterialList* pMatList, unsigned matIndex, gpuFloatType_t density );
    void launchTallyCrossSectionAtCollision(unsigned nBlocks, unsigned nThreads, const RayListInterface<N>* pCP, const MaterialList* pMatList, const MaterialProperties* pMatProps );

    gpuFloatType_t getTotalXSByMatProp(const MaterialProperties* matProps, const MaterialList* pMatList, const HashLookup* pHash, unsigned HashBin, unsigned cell, gpuFloatType_t E);
    gpuFloatType_t getTotalXSByMatProp(const MaterialProperties* matProps, const MaterialList* pMatList, unsigned cell, gpuFloatType_t E);

    void launchSumCrossSectionAtCollisionLocation(unsigned nBlocks, unsigned nThreads, const RayListInterface<N>* pCP, const MaterialList* pMatList, const MaterialProperties* pMatProps );
    void launchRayTraceTally(unsigned nBlocks, unsigned nThreads, const RayListInterface<N>* pCP, const MaterialList* pMatList, const MaterialProperties* pMatProps );

    gpuFloatType_t getTally(unsigned i) const { return tally[i]; }

private:

#ifdef __CUDACC__
    cudaEvent_t start, stop;
#else
    cpuTimer timer;
#endif

	unsigned nCells;

	gpuTallyType_t* tally = nullptr;
	GridBins* grid_device = nullptr;

};

#endif /* FI_TEST_GENERICGPU_TEST_HELPER_HH_ */

