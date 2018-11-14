#ifndef RAYLISTCONTROLLER_HH_
#define RAYLISTCONTROLLER_HH_

#include <string>
#include <vector>
#include <memory>
#include <functional>

#include "MonteRayTypes.hh"
#include "MonteRayParallelAssistant.hh"

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
    gpuTallyType_t getPointDetTally(unsigned spatialIndex, unsigned timeIndex=0 ) const;
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

    template<typename T>
    void setTimeBinEdges( std::vector<T> edges) {
        TallyTimeBinEdges.resize( edges.size() );
        for( unsigned i=0; i<edges.size(); ++i) {
            TallyTimeBinEdges[i] = edges[i];
        }
    }

    void gather();

private:
    unsigned nBlocks = 0;
    unsigned nThreads = 0;
    GRID_T* pGrid = nullptr;
    MonteRayMaterialListHost* pMatList = nullptr;
    MonteRay_MaterialProperties* pMatProps = nullptr;
    gpuTallyHost* pTally = nullptr;
    const MonteRayParallelAssistant& PA;

    std::shared_ptr<MonteRayNextEventEstimator<GRID_T>> pNextEventEstimator;
    std::vector<gpuFloatType_t> TallyTimeBinEdges;

    RayListInterface<N>* currentBank = nullptr;
    std::unique_ptr<RayListInterface<N>> bank1;
    std::unique_ptr<RayListInterface<N>> bank2;
    unsigned nFlushs = 0;

    std::unique_ptr<cpuTimer> pTimer;
    double cpuTime = 0.0;
    double gpuTime = 0.0;
    double wallTime= 0.0;
    bool toFile = false;
    bool fileIsOpen = false;
    bool tallyInitialized = false;

    std::string outputFileName;

    void sendToFile(void) { toFile = true; }

    bool usingNextEventEstimator = false;

    typedef std::function<void ( void )> kernel_t;
    kernel_t kernel;

    std::unique_ptr<cudaStream_t> stream1;
    std::unique_ptr<cudaEvent_t> startGPU;
    std::unique_ptr<cudaEvent_t> stopGPU;
    std::unique_ptr<cudaEvent_t> start;
    std::unique_ptr<cudaEvent_t> stop;
    std::unique_ptr<cudaEvent_t> copySync1;
    std::unique_ptr<cudaEvent_t> copySync2;
    cudaEvent_t* currentCopySync = nullptr;
};

template<class GRID_T>
using CollisionPointController = typename MonteRay::RayListController<GRID_T,1>;

template<class GRID_T>
using NextEventEstimatorController = typename MonteRay::RayListController<GRID_T,3>;

} /* namespace MonteRay */

#endif /* RAYLISTCONTROLLER_HH_ */
