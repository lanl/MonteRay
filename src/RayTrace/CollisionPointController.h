#ifndef COLLISIONPOINTCONTROLLER_H_
#define COLLISIONPOINTCONTROLLER_H_

#include "gpuGlobal.h"
#include "cpuTimer.h"

namespace MonteRay {

class GridBinsHost;
class SimpleMaterialListHost;
class SimpleMaterialPropertiesHost;
class gpuTallyHost;
class CollisionPointsHost;

class CollisionPointController {
public:
	CollisionPointController(unsigned nBlocks,
			                 unsigned nThreads,
			                 GridBinsHost*,
			                 SimpleMaterialListHost*,
			                 SimpleMaterialPropertiesHost*,
			                 gpuTallyHost* );

	virtual ~CollisionPointController();

	unsigned capacity(void) const;
	unsigned size(void) const;
	void setCapacity(unsigned n );

	void add( gpuFloatType_t pos[3],
			  gpuFloatType_t dir[3],
			  gpuFloatType_t energy, gpuFloatType_t weight, unsigned index);

    void add( gpuFloatType_t x, gpuFloatType_t y, gpuFloatType_t z,
              gpuFloatType_t u, gpuFloatType_t v, gpuFloatType_t w,
              gpuFloatType_t energy, gpuFloatType_t weight, unsigned index);

    void flush(bool final=false);
    void finalFlush(void);
    void stopTimers(void);
    void startTimers(void);
    void swapBanks(void);
    void printCycleTime(float_t cpu, float_t gpu, float_t wall) const;
    void printTotalTime(void) const;

    double getCPUTime(void) const { return cpuTime; }
    double getGPUTime(void) const { return gpuTime; }

    void sync(void);

    void clearTally(void);
private:
	unsigned nBlocks;
	unsigned nThreads;
	GridBinsHost* pGrid;
	SimpleMaterialListHost* pMatList;
	SimpleMaterialPropertiesHost* pMatProps;
	gpuTallyHost* pTally;

	CollisionPointsHost* currentBank;
	CollisionPointsHost* bank1;
	CollisionPointsHost* bank2;
	unsigned nFlushs;

	cudaStream_t stream1;
	cudaEvent_t startGPU, stopGPU, start, stop;
	cudaEvent_t copySync1, copySync2;
	cudaEvent_t* currentCopySync;
	cpuTimer timer;
	double cpuTime, gpuTime, wallTime;
};

} /* namespace MonteRay */

#endif /* COLLISIONPOINTCONTROLLER_H_ */
