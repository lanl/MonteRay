#ifndef MONTERAYNEXTEVENTESTIMATOR_HH_
#define MONTERAYNEXTEVENTESTIMATOR_HH_

#include <iostream>

#include "MonteRayTypes.hh"
#include "MonteRayAssert.hh"
#include "MonteRayCopyMemory.hh"
#include "MonteRayVector3D.hh"

namespace MonteRay {

class HashLookup;
class HashLookupHost;
class GridBins;
class MonteRay_MaterialProperties;
class MonteRay_MaterialProperties_Data;
class MonteRayMaterialListHost;
class MonteRayMaterialList;

template< unsigned N>
class RayList_t;

#ifndef __CUDACC__
class cudaStream_t;
#endif

template<typename GRID_T>
class MonteRayNextEventEstimator : public CopyMemoryBase<MonteRayNextEventEstimator<GRID_T>> {
public:
    typedef gpuTallyType_t tally_t;
    typedef gpuRayFloat_t position_t;
    typedef unsigned DetectorIndex_t;
    using Base = MonteRay::CopyMemoryBase<MonteRayNextEventEstimator<GRID_T>> ;

    CUDAHOST_CALLABLE_MEMBER std::string className(){ return std::string("MonteRayNextEventEstimator");}

    CUDAHOST_CALLABLE_MEMBER MonteRayNextEventEstimator(unsigned num);

    CUDAHOST_CALLABLE_MEMBER ~MonteRayNextEventEstimator();

    CUDAHOST_CALLABLE_MEMBER void init();

    CUDAHOST_CALLABLE_MEMBER void copy(const MonteRayNextEventEstimator* rhs);

    CUDAHOST_CALLABLE_MEMBER unsigned add( position_t xarg, position_t yarg, position_t zarg);

    CUDA_CALLABLE_MEMBER unsigned size(void) const { return nUsed; }
    CUDA_CALLABLE_MEMBER unsigned capacity(void) const { return nAllocated; }

    CUDAHOST_CALLABLE_MEMBER void setExclusionRadius(position_t r) { radius = r; }
    CUDA_CALLABLE_MEMBER position_t getExclusionRadius(void) const { return radius; }

    CUDA_CALLABLE_MEMBER position_t getX(unsigned i) const { MONTERAY_ASSERT(i<nUsed); return x[i]; }
    CUDA_CALLABLE_MEMBER position_t getY(unsigned i) const { MONTERAY_ASSERT(i<nUsed); return y[i]; }
    CUDA_CALLABLE_MEMBER position_t getZ(unsigned i) const { MONTERAY_ASSERT(i<nUsed); return z[i]; }

    CUDA_CALLABLE_MEMBER tally_t getTally(unsigned i) const { MONTERAY_ASSERT(i<nUsed); return tally[i]; }

    CUDA_CALLABLE_MEMBER position_t distance(unsigned i, MonteRay::Vector3D<gpuRayFloat_t>& pos ) const;

    CUDA_CALLABLE_MEMBER position_t getDistanceDirection(
            unsigned i, MonteRay::Vector3D<gpuRayFloat_t>& pos, MonteRay::Vector3D<gpuRayFloat_t>& dir ) const;

    template<unsigned N>
    CUDA_CALLABLE_MEMBER tally_t calcScore(
            DetectorIndex_t detectorIndex,
            position_t x, position_t y, position_t z,
            position_t u, position_t v, position_t w,
            gpuFloatType_t energies[N], gpuFloatType_t weights[N],
            unsigned locationIndex,
            ParticleType_t particleType
    );

    template<unsigned N>
    CUDA_CALLABLE_MEMBER void score( const RayList_t<N>* pRayList, unsigned tid );

    template<unsigned N>
    void cpuScoreRayList( const RayList_t<N>* pRayList ) {
        for( auto i=0; i<pRayList->size(); ++i ) {
            score(pRayList,i);
        }
    }

    template<unsigned N>
    void launch_ScoreRayList( unsigned nBlocks, unsigned nThreads, cudaStream_t* stream, const RayList_t<N>* pRayList );

    CUDAHOST_CALLABLE_MEMBER void setGeometry(const GRID_T* pGrid, const MonteRay_MaterialProperties* pMPs);

    CUDAHOST_CALLABLE_MEMBER void setMaterialList(const MonteRayMaterialListHost* ptr);

    CUDAHOST_CALLABLE_MEMBER
    MonteRay::Vector3D<gpuFloatType_t> getPoint(unsigned i) const {
        return MonteRay::Vector3D<gpuFloatType_t>( x[i], y[i], z[i] );
    }

    CUDAHOST_CALLABLE_MEMBER
    void
    printPointDets( const std::string& outputFile, unsigned nSamples, unsigned constantDimension=2);

private:
    unsigned nUsed;
    unsigned nAllocated;
    position_t radius;

    position_t* x;
    position_t* y;
    position_t* z;

    tally_t* tally;

    const GRID_T* pGridBins;
    const MonteRay_MaterialProperties* pMatPropsHost;
    const MonteRay_MaterialProperties_Data* pMatProps;
    const MonteRayMaterialListHost* pMatListHost;
    const MonteRayMaterialList* pMatList;
    const HashLookupHost* pHashHost;
    const HashLookup* pHash;
};

template<typename GRID_T, unsigned N>
CUDA_CALLABLE_KERNEL void kernel_ScoreRayList(MonteRayNextEventEstimator<GRID_T>* ptr, const RayList_t<N>* pRayList );

} /* namespace MonteRay */

#endif /* MONTERAYNEXTEVENTESTIMATOR_HH_ */
