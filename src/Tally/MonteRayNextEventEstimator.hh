#ifndef MONTERAYNEXTEVENTESTIMATOR_HH_
#define MONTERAYNEXTEVENTESTIMATOR_HH_

#include <iostream>
#include <sys/types.h>
#include <unistd.h>
#include <limits>
#include <tuple>

#include "MonteRayTypes.hh"
#include "MonteRayAssert.hh"
#include "MonteRayCopyMemory.hh"
#include "MonteRayVector3D.hh"

namespace MonteRay {

class HashLookup;
class HashLookupHost;
class GridBins;
class MonteRayMaterialListHost;
class MonteRayMaterialList;
class MonteRayTally;

template< unsigned N>
class RayList_t;

template< unsigned N>
class Ray_t;

class RayWorkInfo;

template<typename Geometry>
class MonteRayNextEventEstimator : public CopyMemoryBase<MonteRayNextEventEstimator<Geometry>> {
public:
    using tally_t = gpuTallyType_t;
    using position_t = gpuRayFloat_t;
    using DetectorIndex_t = unsigned;
    using Base = MonteRay::CopyMemoryBase<MonteRayNextEventEstimator<Geometry>> ;

    CUDAHOST_CALLABLE_MEMBER std::string className(){ return std::string("MonteRayNextEventEstimator");}

    CUDAHOST_CALLABLE_MEMBER MonteRayNextEventEstimator(unsigned num);

    CUDAHOST_CALLABLE_MEMBER ~MonteRayNextEventEstimator();

    void reallocate(unsigned num);

    CUDAHOST_CALLABLE_MEMBER void init();

    void initialize();

    CUDAHOST_CALLABLE_MEMBER void copy(const MonteRayNextEventEstimator* rhs);

    void copyToGPU();
    void copyToCPU();

    CUDAHOST_CALLABLE_MEMBER unsigned add( position_t xarg, position_t yarg, position_t zarg);

    CUDA_CALLABLE_MEMBER unsigned size(void) const { return nUsed; }
    CUDA_CALLABLE_MEMBER unsigned capacity(void) const { return nAllocated; }

    CUDAHOST_CALLABLE_MEMBER void setExclusionRadius(position_t r) { radius = r; }
    CUDA_CALLABLE_MEMBER position_t getExclusionRadius(void) const { return radius; }

    CUDA_CALLABLE_MEMBER tally_t getTally(unsigned spatialIndex, unsigned timeIndex=0) const;


    template<unsigned N>
    CUDA_CALLABLE_MEMBER tally_t calcScore( unsigned threadID, Ray_t<N>& ray, RayWorkInfo& rayInfo );

    template<unsigned N>
    CUDA_CALLABLE_MEMBER void score( const RayList_t<N>* pRayList, RayWorkInfo* pRayInfo, unsigned tid, unsigned pid );

    template<unsigned N>
    void cpuScoreRayList( const RayList_t<N>* pRayList, RayWorkInfo* pRayInfo );

    template<unsigned N>
    void launch_ScoreRayList( int nBlocks, int nThreads, const RayList_t<N>* pRayList, RayWorkInfo* pRayInfo, cudaStream_t* stream = nullptr, bool dumpOnFailure = true );

    template<unsigned N>
    void dumpState( const RayList_t<N>* pRayList, const std::string& optBaseName  = std::string("") );

    CUDAHOST_CALLABLE_MEMBER void setGeometry(const Geometry* pGeometry, const MaterialProperties* pMPs);
    CUDAHOST_CALLABLE_MEMBER void updateMaterialProperties( MaterialProperties* pMPs);

    CUDAHOST_CALLABLE_MEMBER void setMaterialList(const MonteRayMaterialListHost* ptr);

    CUDAHOST_CALLABLE_MEMBER
    const auto& getPoint(unsigned i) const { MONTERAY_ASSERT(i<nUsed);  return tallyPoints[i]; }

    CUDAHOST_CALLABLE_MEMBER
    void
    printPointDets( const std::string& outputFile, unsigned nSamples, unsigned constantDimension=2);

    CUDAHOST_CALLABLE_MEMBER
    void
    outputTimeBinnedTotal(std::ostream& out, unsigned nSamples=1, unsigned constantDimension=2);

    template<typename T>
    void setTimeBinEdges( std::vector<T> edges) {
        pTallyTimeBinEdges = new std::vector<gpuFloatType_t>;
        pTallyTimeBinEdges->assign( edges.begin(), edges.end() );
    }

    std::vector<gpuFloatType_t> getTimeBinEdges() {
        if( ! pTallyTimeBinEdges ) {
            return std::vector<gpuFloatType_t>();
        } else {
            return *pTallyTimeBinEdges;
        }
    }

    void gather();

    // gather work group is rarely used, mainly for testing
    void gatherWorkGroup();

    template<typename IOTYPE>
    void write(IOTYPE& out);

    template<typename IOTYPE>
    void read(IOTYPE& in);

    // write out state of MonteRayNextEventEstimator class
    void writeToFile( const std::string& fileName);
    void readFromFile( const std::string& fileName);

private:
    unsigned nUsed;
    unsigned nAllocated;
    position_t radius;

    MonteRay::Vector3D<position_t>* tallyPoints = NULL;

    MonteRayTally* pTally = NULL;
    std::vector<gpuFloatType_t>* pTallyTimeBinEdges = NULL;

    const Geometry* pGeometry = NULL;
    const MaterialProperties* pMatProps = NULL;
    const MonteRayMaterialListHost* pMatListHost = NULL;
    const MonteRayMaterialList* pMatList = NULL;
    const HashLookupHost* pHashHost = NULL;
    const HashLookup* pHash = NULL;

    bool initialized = false;
    bool copiedToGPU = false;
};

template<typename Geometry, unsigned N>
CUDA_CALLABLE_KERNEL  kernel_ScoreRayList(MonteRayNextEventEstimator<Geometry>* ptr, const RayList_t<N>* pRayList );

} /* namespace MonteRay */

#endif /* MONTERAYNEXTEVENTESTIMATOR_HH_ */
