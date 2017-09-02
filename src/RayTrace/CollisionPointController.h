#ifndef COLLISIONPOINTCONTROLLER_H_
#define COLLISIONPOINTCONTROLLER_H_

#include <string>

#include "MonteRayDefinitions.hh"
#include "MonteRay_timer.hh"
#include "CollisionPoints.h"

namespace MonteRay {

class GridBinsHost;
class SimpleMaterialListHost;
class MonteRay_MaterialProperties;
class gpuTallyHost;

class CollisionPointController {
public:

	CollisionPointController(unsigned nBlocks,
			                 unsigned nThreads,
			                 GridBinsHost*,
			                 SimpleMaterialListHost*,
			                 MonteRay_MaterialProperties*,
			                 gpuTallyHost* );

	virtual ~CollisionPointController();

	unsigned capacity(void) const;
	unsigned size(void) const;
	void setCapacity(unsigned n );

	void add( gpuFloatType_t pos[3],
			  gpuFloatType_t dir[3],
			  gpuFloatType_t energy, gpuFloatType_t weight,
			  unsigned index, DetectorIndex_t detectorIndex, ParticleType_t particleType);

    void add( gpuFloatType_t x, gpuFloatType_t y, gpuFloatType_t z,
              gpuFloatType_t u, gpuFloatType_t v, gpuFloatType_t w,
              gpuFloatType_t energy, gpuFloatType_t weight,
			  unsigned index, DetectorIndex_t detectorIndex, ParticleType_t particleType);

    void add( const ParticleRay_t& );
    void add( const ParticleRay_t* particle, unsigned N=1 );
    void add( const void* particle, unsigned N=1 );

    void flush(bool final=false);
    void finalFlush(void);
    void stopTimers(void);
    void startTimers(void);
    void swapBanks(void);
    void printCycleTime(float_t cpu, float_t gpu, float_t wall) const;
    void printTotalTime(void) const;

    double getCPUTime(void) const { return cpuTime; }
    double getGPUTime(void) const { return gpuTime; }
    unsigned getNFlushes(void) const { return nFlushs; }

    void sync(void);

    void clearTally(void);

    bool isSendingToFile(void) { return toFile; }

    void setOutputFileName(std::string name) {
    	outputFileName = name;
    	sendToFile();
    }

    size_t readCollisionsFromFile(std::string name);

    void flushToFile(bool final=false);

    void debugPrint() {
    	currentBank->debugPrint();
    }

private:
	unsigned nBlocks;
	unsigned nThreads;
	GridBinsHost* pGrid;
	SimpleMaterialListHost* pMatList;
	MonteRay_MaterialProperties* pMatProps;
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
	bool toFile;
	bool fileIsOpen;

	std::string outputFileName;

    void sendToFile(void) { toFile = true; }
};

} /* namespace MonteRay */

#endif /* COLLISIONPOINTCONTROLLER_H_ */
