/*
 * MonteRay_GridSystemInterface.hh
 *
 *  Created on: Feb 2, 2018
 *      Author: jsweezy
 */

#ifndef MONTERAYGRIDSYSTEMINTERFACE_HH_
#define MONTERAYGRIDSYSTEMINTERFACE_HH_

#include "MonteRayDefinitions.hh"

#include <utility>
#include <vector>

#include "MonteRay_GridBins.hh"

namespace MonteRay {

class singleDimRayTraceMap_t {
private:
	unsigned N = 0;
	int CellId[MAXNUMVERTICES]; // negative indicates outside mesh
	gpuRayFloat_t distance[MAXNUMVERTICES];

public:
	CUDA_CALLABLE_MEMBER singleDimRayTraceMap_t() : N(0) {}
	CUDA_CALLABLE_MEMBER singleDimRayTraceMap_t(unsigned n) : N(n) {}
	CUDA_CALLABLE_MEMBER ~singleDimRayTraceMap_t(){}

	CUDA_CALLABLE_MEMBER
	void add( int cell, gpuRayFloat_t dist) {
		MONTERAY_ASSERT( N < MAXNUMVERTICES-1);
		CellId[N] = cell;
		distance[N] = dist;
		++N;
	}

	CUDA_CALLABLE_MEMBER void clear() { reset(); }
	CUDA_CALLABLE_MEMBER void reset() { N = 0; }
	CUDA_CALLABLE_MEMBER unsigned size() const { return N; }

	CUDA_CALLABLE_MEMBER int id(size_t i) const { return CellId[i]; }
	CUDA_CALLABLE_MEMBER gpuRayFloat_t dist(size_t i) const { return distance[i]; }
};

struct rayTraceList_t {
private:
	unsigned N;
	unsigned CellId[MAXNUMVERTICES*2];
	gpuRayFloat_t distance[MAXNUMVERTICES*2];

public:
	CUDA_CALLABLE_MEMBER rayTraceList_t() : N(0) {}
	CUDA_CALLABLE_MEMBER rayTraceList_t(unsigned n) : N(n) {}
	CUDA_CALLABLE_MEMBER ~rayTraceList_t(){}

	CUDA_CALLABLE_MEMBER
	void add( unsigned cell, gpuRayFloat_t dist) {
		MONTERAY_ASSERT( N < MAXNUMVERTICES-1);
		CellId[N] = cell;
		distance[N] = dist;
		++N;
	}

	CUDA_CALLABLE_MEMBER void clear() { reset(); }
	CUDA_CALLABLE_MEMBER void reset() { N = 0; }
	CUDA_CALLABLE_MEMBER unsigned size() const { return N; }

	CUDA_CALLABLE_MEMBER unsigned id(size_t i) const { return CellId[i]; }
	CUDA_CALLABLE_MEMBER gpuRayFloat_t dist(size_t i) const { return distance[i]; }
};

class multiDimRayTraceMap_t {
public:
	CUDA_CALLABLE_MEMBER multiDimRayTraceMap_t(){}
	CUDA_CALLABLE_MEMBER ~multiDimRayTraceMap_t(){}

	const unsigned N = 3 ;  // hardcoded to 3D for now.
	singleDimRayTraceMap_t traceMapList[3];

	CUDA_CALLABLE_MEMBER singleDimRayTraceMap_t& operator[] (size_t i ) { return traceMapList[i];}
	CUDA_CALLABLE_MEMBER const singleDimRayTraceMap_t& operator[] (size_t i ) const { return traceMapList[i];}

};

class MonteRay_GridSystemInterface {

#define OUTSIDE_INDEX UINT_MAX;
public:
//    typedef std::vector<std::pair<int,gpuRayFloat_t>> singleDimRayTraceMap_t;
//    typedef std::vector<singleDimRayTraceMap_t> multiDimRayTraceMap_t;
//    typedef std::vector<std::pair<unsigned, gpuRayFloat_t>> rayTraceList_t;
	using GridBins_t = MonteRay_GridBins;
    typedef GridBins_t* pGridBins_t;
    //static const unsigned OUTSIDE = UINT_MAX;

    CUDA_CALLABLE_MEMBER MonteRay_GridSystemInterface(unsigned dim) : DIM(dim) {}
    CUDA_CALLABLE_MEMBER virtual ~MonteRay_GridSystemInterface(){};

    CUDA_CALLABLE_MEMBER virtual unsigned getIndex( const GridBins_t::Position_t& particle_pos ) const = 0;

    CUDA_CALLABLE_MEMBER
    virtual void
    rayTrace( rayTraceList_t&, const GridBins_t::Position_t& particle_pos, const GridBins_t::Position_t& particle_dir, gpuRayFloat_t distance, bool outsideDistances=false ) const = 0;

    CUDA_CALLABLE_MEMBER
    virtual void
    crossingDistance( singleDimRayTraceMap_t&, unsigned dim, gpuRayFloat_t pos, gpuRayFloat_t dir, gpuRayFloat_t distance ) const {
        ABORT("Single dimension crossingDistance function not implemented for this grid type.");
    }

    CUDA_CALLABLE_MEMBER
    virtual void
    crossingDistance( singleDimRayTraceMap_t&, const GridBins_t::Position_t& pos, const GridBins_t::Direction_t& dir, gpuRayFloat_t distance ) const {
        ABORT("Multi-dimension crossingDistance function not implemented for this grid type.");
    }

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
    CUDA_CALLABLE_MEMBER
    void orderCrossings( rayTraceList_t&, const multiDimRayTraceMap_t& distances, int indices[], gpuRayFloat_t distance, bool outsideDistances=false ) const;

    CUDA_CALLABLE_MEMBER
    void planarCrossingDistance( singleDimRayTraceMap_t&, const GridBins_t& Bins, gpuRayFloat_t pos, gpuRayFloat_t dir, gpuRayFloat_t distance, int index) const;

    CUDA_CALLABLE_MEMBER
    bool radialCrossingDistanceSingleDirection( singleDimRayTraceMap_t& distances, const GridBins_t& Bins, gpuRayFloat_t particle_R2, gpuRayFloat_t A, gpuRayFloat_t B, gpuRayFloat_t distance, int index, bool outward ) const;

public:
	static constexpr unsigned OUTSIDE_GRID = UINT_MAX;

    unsigned DIM = 0;
    static const unsigned MAXDIM = 3;

private:
    static constexpr gpuRayFloat_t inf = std::numeric_limits<gpuRayFloat_t>::infinity();
};

} /* namespace MonteRay */
#endif /* MONTERAYGRIDSYSTEMINTERFACE_HH_ */

