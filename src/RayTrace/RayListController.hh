#ifndef RAYLISTCONTROLLER_HH_
#define RAYLISTCONTROLLER_HH_

#include <string>
#include <memory>
#include <functional>

#include "MonteRayTypes.hh"

namespace MonteRay {

class MonteRayMaterialListHost;
class MonteRay_MaterialProperties;
class gpuTallyHost;
class cpuTimer;

template<typename GRID_T>
class MonteRayNextEventEstimator;

template< unsigned N >
class RayListInterface;

template< unsigned N >
class Ray_t;

#ifndef __CUDACC__
class cudaStream_t;
class cudaEvent_t;
#endif

template<typename GRID_T, unsigned N = 1>
class RayListController {
public:

    /// Ctor for the volumetric ray casting solver
    RayListController(unsigned nBlocks,
            unsigned nThreads,
            GRID_T*,
            MonteRayMaterialListHost*,
            MonteRay_MaterialProperties*,
            gpuTallyHost* );

    /// Ctor for the next event estimator solver
    RayListController(unsigned nBlocks,
            unsigned nThreads,
            GRID_T*,
            MonteRayMaterialListHost*,
            MonteRay_MaterialProperties*,
            unsigned numPointDets );

    /// Ctor for the writing next-event estimator collision and source points to file
    /// Can not launch a kernel
    RayListController( unsigned numPointDets, const std::string& filename );

    virtual ~RayListController();

    void initialize();
    unsigned capacity(void) const;
    unsigned size(void) const;
    void setCapacity(unsigned n );

    void add( const Ray_t<N>& ray);
    void add( const Ray_t<N>* rayArray, unsigned num=1 );
    void add( const void* ray, unsigned num=1 ) { add(  (const Ray_t<N>*) ray, num  ); }

    unsigned addPointDet( gpuFloatType_t x, gpuFloatType_t y, gpuFloatType_t z );
    void setPointDetExclusionRadius(gpuFloatType_t r);
    void copyPointDetTallyToCPU(void);
    gpuTallyType_t getPointDetTally(unsigned i ) const;
    void copyPointDetToGPU(void);
    void printPointDets( const std::string& outputFile, unsigned nSamples, unsigned constantDimension=2);

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

    bool isUsingNextEventEstimator(void) const {
        return usingNextEventEstimator;
    }

private:
    unsigned nBlocks;
    unsigned nThreads;
    GRID_T* pGrid;
    MonteRayMaterialListHost* pMatList;
    MonteRay_MaterialProperties* pMatProps;
    gpuTallyHost* pTally;
    std::shared_ptr<MonteRayNextEventEstimator<GRID_T>> pNextEventEstimator;

    RayListInterface<N>* currentBank;
    RayListInterface<N>* bank1;
    RayListInterface<N>* bank2;
    unsigned nFlushs;

    cpuTimer* pTimer;
    double cpuTime, gpuTime, wallTime;
    bool toFile;
    bool fileIsOpen;

    std::string outputFileName;

    void sendToFile(void) { toFile = true; }

    bool usingNextEventEstimator = false;

    typedef std::function<void ( void )> kernel_t;
    kernel_t kernel;

    cudaStream_t* stream1;
    cudaEvent_t* startGPU;
    cudaEvent_t* stopGPU;
    cudaEvent_t* start;
    cudaEvent_t* stop;
    cudaEvent_t* copySync1;
    cudaEvent_t* copySync2;
    cudaEvent_t* currentCopySync;
};

template<class GRID_T>
using CollisionPointController = typename MonteRay::RayListController<GRID_T,1>;

template<class GRID_T>
using NextEventEstimatorController = typename MonteRay::RayListController<GRID_T,3>;

} /* namespace MonteRay */

#endif /* RAYLISTCONTROLLER_HH_ */
