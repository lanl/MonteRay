#ifndef MONTERAYGRIDSYSTEMINTERFACE_HH_
#define MONTERAYGRIDSYSTEMINTERFACE_HH_

#include <utility>
#include <vector>
#include <climits>

#include "MonteRay_GridBins.hh"
#include "RayWorkInfo.hh"

namespace MonteRay {


class singleDimRayTraceMap_t {
private:
    unsigned N = 0;
    int CellId[MAXNUMVERTICES]; // negative indicates outside mesh
    gpuRayFloat_t distance[MAXNUMVERTICES];

public:
    CUDA_CALLABLE_MEMBER singleDimRayTraceMap_t() {}
    CUDA_CALLABLE_MEMBER singleDimRayTraceMap_t(unsigned n) : N(n) {}
    CUDA_CALLABLE_MEMBER ~singleDimRayTraceMap_t(){}

    // for conversion of old tests
    CUDA_CALLABLE_MEMBER singleDimRayTraceMap_t(RayWorkInfo&, const unsigned threadID, int dim = -1);

    CUDA_CALLABLE_MEMBER
    void add( const int cell, const gpuRayFloat_t dist);

    CUDA_CALLABLE_MEMBER void clear() { reset(); }
    CUDA_CALLABLE_MEMBER void reset() { N = 0; }
    CUDA_CALLABLE_MEMBER unsigned size() const { return N; }

    CUDA_CALLABLE_MEMBER int id( const size_t i) const { return CellId[i]; }
    CUDA_CALLABLE_MEMBER gpuRayFloat_t dist(const size_t i) const { return distance[i]; }
};

class rayTraceList_t {
private:
    unsigned N = 0;
    unsigned CellId[MAXNUMVERTICES*2];
    gpuRayFloat_t distance[MAXNUMVERTICES*2];

public:
    CUDA_CALLABLE_MEMBER rayTraceList_t() {}
    CUDA_CALLABLE_MEMBER rayTraceList_t(unsigned n) : N(n) {}
    CUDA_CALLABLE_MEMBER ~rayTraceList_t(){}

    // for conversion of old tests
    CUDA_CALLABLE_MEMBER rayTraceList_t(RayWorkInfo&, const unsigned threadID, int dim = -1);

    CUDA_CALLABLE_MEMBER
    void add( const unsigned cell, const gpuRayFloat_t dist);

    CUDA_CALLABLE_MEMBER void clear() { reset(); }
    CUDA_CALLABLE_MEMBER void reset() { N = 0; }
    CUDA_CALLABLE_MEMBER unsigned size() const { return N; }

    CUDA_CALLABLE_MEMBER unsigned id(const size_t i) const { return CellId[i]; }
    CUDA_CALLABLE_MEMBER gpuRayFloat_t dist(const size_t i) const { return distance[i]; }
};

template<unsigned DIM = 3>
class multiDimRayTraceMap_t {
public:
    CUDA_CALLABLE_MEMBER multiDimRayTraceMap_t(){}
    CUDA_CALLABLE_MEMBER ~multiDimRayTraceMap_t(){}

    const unsigned N = DIM;
    singleDimRayTraceMap_t traceMapList[DIM];

    CUDA_CALLABLE_MEMBER singleDimRayTraceMap_t& operator[] (const size_t i ) { return traceMapList[i];}
    CUDA_CALLABLE_MEMBER const singleDimRayTraceMap_t& operator[] ( const size_t i ) const { return traceMapList[i];}

};

class MonteRay_GridSystemInterface {

#define OUTSIDE_INDEX UINT_MAX;
public:
    using GridBins_t = MonteRay_GridBins;
    using pGridBins_t = GridBins_t*;
    //static const unsigned OUTSIDE = UINT_MAX;

    CUDA_CALLABLE_MEMBER MonteRay_GridSystemInterface(unsigned dim) : DIM(dim) {}
    CUDA_CALLABLE_MEMBER virtual ~MonteRay_GridSystemInterface(){};

    CUDA_CALLABLE_MEMBER virtual unsigned getIndex( const GridBins_t::Position_t& particle_pos ) const = 0;

    CUDA_CALLABLE_MEMBER
    virtual void
    rayTrace( const unsigned threadID,
              RayWorkInfo& rayInfo,
              const GridBins_t::Position_t& particle_pos,
              const GridBins_t::Position_t& particle_dir,
              const gpuRayFloat_t distance,
              const bool outsideDistances=false ) const = 0;

    CUDA_CALLABLE_MEMBER
    virtual void
    rayTraceWithMovingMaterials( const unsigned threadID,
              RayWorkInfo& rayInfo,
              const GridBins_t::Position_t& particle_pos,
              const GridBins_t::Position_t& particle_dir,
              const gpuRayFloat_t distance,
              const bool outsideDistances=false ) const = 0;

    CUDA_CALLABLE_MEMBER
    virtual void
    crossingDistance( const unsigned dim,
                      const unsigned threadID,
                      RayWorkInfo& rayInfo,
                      const GridBins_t::Position_t& pos,
                      const GridBins_t::Direction_t& dir,
                      const gpuRayFloat_t distance ) const = 0;

    CUDA_CALLABLE_MEMBER
    virtual gpuRayFloat_t getVolume( unsigned index ) const = 0;

    CUDA_CALLABLE_MEMBER
    virtual bool isOutside( const int i[]) const = 0;

    CUDA_CALLABLE_MEMBER
    virtual bool isIndexOutside( unsigned d,  int i) const = 0;

    CUDA_CALLABLE_MEMBER
    virtual unsigned calcIndex( const int i[] ) const = 0;

    CUDA_CALLABLE_MEMBER
    unsigned getDimension(void) const { return DIM; }

    CUDAHOST_CALLABLE_MEMBER
    virtual void copyToGPU(void) = 0;

    CUDAHOST_CALLABLE_MEMBER
    virtual MonteRay_GridSystemInterface* getDeviceInstancePtr(void) = 0;

protected:

    template<unsigned NUMDIM>
    CUDA_CALLABLE_MEMBER
    void orderCrossings(
            const unsigned threadID,
            RayWorkInfo& rayInfo,
            int indices[],
            const gpuRayFloat_t distance,
            const bool outsideDistances=false ) const;

    CUDA_CALLABLE_MEMBER
    void planarCrossingDistance(
            const unsigned dim,
            const unsigned threadID,
            RayWorkInfo& rayInfo,
            const GridBins_t& Bins,
            const gpuRayFloat_t pos,
            const gpuRayFloat_t dir,
            const gpuRayFloat_t distance,
            const int index) const;

    template<bool OUTWARD>
    CUDA_CALLABLE_MEMBER
    bool radialCrossingDistanceSingleDirection(
            const unsigned dim,
            const unsigned threadID,
            RayWorkInfo& rayInfo,
            const GridBins_t& Bins,
            const gpuRayFloat_t particle_R2,
            const gpuRayFloat_t A,
            const gpuRayFloat_t B,
            const gpuRayFloat_t distance,
            int index) const;

public:
    static constexpr unsigned OUTSIDE_GRID = UINT_MAX;

    unsigned DIM = 0;
    static const unsigned MAXDIM = 3;

private:
    static constexpr gpuRayFloat_t inf = std::numeric_limits<gpuRayFloat_t>::infinity();
};

} /* namespace MonteRay */
#endif /* MONTERAYGRIDSYSTEMINTERFACE_HH_ */

