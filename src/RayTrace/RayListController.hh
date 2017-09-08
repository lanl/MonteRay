#ifndef RAYLISTCONTROLLER_H_
#define RAYLISTCONTROLLER_H_

#include <string>

#include "MonteRayDefinitions.hh"
#include "MonteRay_timer.hh"
#include "RayListInterface.hh"

namespace MonteRay {

class GridBinsHost;
class SimpleMaterialListHost;
class MonteRay_MaterialProperties;
class gpuTallyHost;

template<unsigned N = 1>
class RayListController {
public:

	RayListController(unsigned nBlocks,
			                 unsigned nThreads,
			                 GridBinsHost*,
			                 SimpleMaterialListHost*,
			                 MonteRay_MaterialProperties*,
			                 gpuTallyHost* );

	virtual ~RayListController();

	unsigned capacity(void) const;
	unsigned size(void) const;
	void setCapacity(unsigned n );

    void add( const Ray_t<N>& ray);
    void add( const Ray_t<N>* rayArray, unsigned num=1 );
    void add( const void* ray, unsigned num=1 ) { add(  (const Ray_t<N>*) ray, num  ); }

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

	RayListInterface<N>* currentBank;
	RayListInterface<N>* bank1;
	RayListInterface<N>* bank2;
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

typedef RayListController<1> CollisionPointController;

} /* namespace MonteRay */

#endif /* RAYLISTCONTROLLER_H_ */
