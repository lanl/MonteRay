#ifndef RAYLISTCONTROLLER_HH_
#define RAYLISTCONTROLLER_HH_

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <iostream>
#include <algorithm>

#include "MaterialList.hh"
#include "MaterialProperties.hh"
#include "BasicTally.hh"
#include "RayListInterface.hh"
#include "ExpectedPathLength.hh"
#include "GPUErrorCheck.hh"
#include "GPUUtilityFunctions.hh"
#include "NextEventEstimator.hh"
#include "MonteRay_timer.hh"
#include "MonteRayTypes.hh"
#include "MonteRayParallelAssistant.hh"
#include "RayWorkInfo.hh"
#include "NextEventEstimator.hh"

namespace MonteRay {

class cpuTimer;

template< unsigned N >
class RayListInterface;

template< unsigned N >
class Ray_t;

template<typename Geometry, unsigned N = 1>
class RayListController {
public:
    using Ray = Ray_t<N>;

    /// Ctor for the volumetric ray casting solver
    RayListController(int nBlocks,
            int nThreads,
            Geometry*,
            MaterialList*,
            MaterialProperties*,
            BasicTally* );

    /// Ctor for the next event estimator solver
    RayListController(int nBlocks,
            int nThreads,
            Geometry*,
            MaterialList*,
            MaterialProperties*,
            unsigned numPointDets );

    /// Ctor for the writing next-event estimator collision and source points to file
    /// Can not launch a kernel
    RayListController( unsigned numPointDets, const std::string& filename );

    ~RayListController();

    void initialize();
    unsigned capacity(void) const;
    unsigned size(void) const;
    void setCapacity(unsigned n );

    void addRay(const Ray& ray){
      if( PA.getWorkGroupRank() != 0 ) { return; }
      currentBank->add( ray );
      if( size() == capacity() ) {
        flush();
      }
    }

    void add( const Ray& ray){ addRay(ray); }
    template <typename Rays>
    void addRays(Rays&& rays){
      for (const auto& ray: rays){
        addRay(ray);
      }
    }
    void add(const Ray* const rayArray, unsigned num=1){
      addRays(SimpleView<const Ray>(rayArray, rayArray + num));
    }
    void add(const void* const rayPtr, unsigned num = 1){ add(static_cast<const Ray*>(rayPtr), num); }


    unsigned addPointDet( gpuFloatType_t x, gpuFloatType_t y, gpuFloatType_t z );
    void setPointDetExclusionRadius(gpuFloatType_t r);
    void copyPointDetTallyToCPU(void);
    gpuTallyType_t getPointDetTally(unsigned spatialIndex, unsigned timeIndex=0 ) const;
    void copyPointDetToGPU(void);
    void dumpPointDetForDebug(const std::string& baseFileName = std::string() );
    void printPointDets( const std::string& outputFile, unsigned nSamples, unsigned constantDimension=2);
    void outputTimeBinnedTotal(std::ostream& out,unsigned nSamples=1, unsigned constantDimension=2);
    CUDAHOST_CALLABLE_MEMBER void updateMaterialProperties( MaterialProperties* pMPs);

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

    // reads a single block of rays to a buffer but doesn't flush them
    // usually for testing or debugging
    size_t readCollisionsFromFileToBuffer(std::string name);

    void flushToFile(bool final=false);

    void debugPrint() {
        currentBank->debugPrint();
    }

    bool isUsingNextEventEstimator(void) const {
        return usingNextEventEstimator;
    }

    template<typename T>
    void setTimeBinEdges( const std::vector<T>& edges) {
        TallyTimeBinEdges.assign( edges.begin(), edges.end() );
    }

    unsigned getWorldRank();

    template<typename T>
    void setEnergyBinEdges( const std::vector<T>& edges) {
        if( getWorldRank() == 0 ) {
            std::cout << "WARNING:  MonteRay does not currently support energy binned tallies.  Ignoring.\n";
        }
    }

    void gather();

private:
    unsigned nBlocks = 0;
    unsigned nThreads = 0;
    Geometry* pGeometry = nullptr;
    MaterialList* pMatList = nullptr;
    MaterialProperties* pMatProps = nullptr;
    BasicTally* pTally = nullptr;
    const MonteRayParallelAssistant& PA;

    std::unique_ptr<NextEventEstimator::Builder> pNextEventEstimatorBuilder; // TPB TODO: why is this a shared ptr
    std::unique_ptr<NextEventEstimator> pNextEventEstimator; // TPB TODO: why is this a shared ptr
    std::vector<gpuFloatType_t> TallyTimeBinEdges;

    RayListInterface<N>* currentBank = nullptr;
    std::unique_ptr<RayListInterface<N>> bank1;
    std::unique_ptr<RayListInterface<N>> bank2;

    std::unique_ptr<RayWorkInfo> rayInfo;

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

template<class Geometry>
using CollisionPointController = typename MonteRay::RayListController<Geometry,1>;

template<class Geometry>
using NextEventEstimatorController = typename MonteRay::RayListController<Geometry,3>;

} /* namespace MonteRay */

#endif /* RAYLISTCONTROLLER_HH_ */
