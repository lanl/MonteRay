#ifndef MONTERAYGRIDSYSTEMINTERFACE_HH_
#define MONTERAYGRIDSYSTEMINTERFACE_HH_

#include <utility>
#include <vector>
#include <climits>

#include "MonteRay_GridBins.hh"
#include "RayWorkInfo.hh"
#include "MaterialProperties.hh"
#include "ThirdParty/Math.hh"
#include "ThirdParty/Array.hh"
#include "MonteRay_QuadraticRootFinder.hh"

#include <sstream>
#include <fstream>

namespace MonteRay {

class MonteRay_GridSystemInterface: public Managed {
public:
  using GridBins_t = MonteRay_GridBins;
  using GridBinsArray_t = Array<GridBins_t, 3>;
  using Position_t = GridBins_t::Position_t;
  using Direction_t = GridBins_t::Direction_t;
  int DIM = 0;
protected:
  static constexpr gpuRayFloat_t inf = std::numeric_limits<gpuRayFloat_t>::infinity();
  GridBinsArray_t gridBins;

public:
  MonteRay_GridSystemInterface() = default;

  MonteRay_GridSystemInterface(GridBinsArray_t otherGridBins, int otherDim) :
    gridBins(std::move(otherGridBins)), DIM(otherDim) {}

  template <typename GridBinsArray>
  MonteRay_GridSystemInterface(GridBinsArray otherGridBins, int otherDim) :
    gridBins( {std::move(otherGridBins[0]), std::move(otherGridBins[1]), std::move(otherGridBins[2])} ), DIM(otherDim) {}


  CUDA_CALLABLE_MEMBER
  unsigned getDimension() const { return DIM; }
  CUDA_CALLABLE_MEMBER
  unsigned getNumBins(unsigned d) const { return gridBins[d].getNumBins(); }
  CUDA_CALLABLE_MEMBER
  unsigned getNumGridBins(unsigned d) const { return gridBins[d].getNumBins(); }
  CUDA_CALLABLE_MEMBER
  gpuRayFloat_t getMinVertex(unsigned d) const { return gridBins[d].getMinVertex(); }
  CUDA_CALLABLE_MEMBER
  gpuRayFloat_t getMaxVertex(unsigned d) const { return gridBins[d].getMaxVertex(); }
  CUDA_CALLABLE_MEMBER
  gpuRayFloat_t getVertex(unsigned d, unsigned i) const { return gridBins[d].vertices[i]; }
  CUDA_CALLABLE_MEMBER
  unsigned getNumCells(void) const {
    unsigned nCells = 1;
    for( int d=0; d < DIM; ++d  ){
      nCells *= gridBins[d].getNumBins(); 
    }
    return nCells;
  }
  CUDA_CALLABLE_MEMBER
  size_t numCells() const {return getNumCells();}
  CUDA_CALLABLE_MEMBER
  size_t size() const {return getNumCells();}
  CUDA_CALLABLE_MEMBER
  size_t getNumVertices(unsigned i) const { return gridBins[i].getNumVertices(); }
  CUDA_CALLABLE_MEMBER
  size_t getNumVerticesSq(unsigned i) const { return gridBins[i].getNumVerticesSq(); }

  CUDA_CALLABLE_MEMBER
  bool isIndexOutside(unsigned d, int i) const {
    return gridBins[d].isIndexOutside(i);
  }

  CUDA_CALLABLE_MEMBER
  bool isOutside(const int i[]) const {
    for( int d=0; d<DIM; ++d){
      if( isIndexOutside(d, i[d]) ){ 
        return true;
      }
    }
    return false;
  }

  template <typename Indices>
  CUDA_CALLABLE_MEMBER
  unsigned calcIndex(const Indices indices) const {
    unsigned index = indices[0];
    if( DIM > 1 ) {
      index += indices[1]*gridBins[0].getNumBins();
    }
    if( DIM > 2 ) {
      index += indices[2] * gridBins[0].getNumBins()*gridBins[1].getNumBins();
    }
    return index;
  }

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
  void write(std::ostream& outfile) const;
  void read(std::istream& infile);
};

using DimType = int;
// This is really DistAndSurf
struct DistAndDir : public std::tuple<gpuRayFloat_t, DimType, bool>{
  using std::tuple<gpuRayFloat_t, DimType, bool>::tuple;
  constexpr auto distance() const { return std::get<0>(*this); }
  constexpr auto dimension() const { return std::get<1>(*this); }
  constexpr auto isPositiveDir() const { return std::get<2>(*this); }

  constexpr void setDistance(gpuRayFloat_t val) { std::get<0>(*this) = val; }
  constexpr void setDimension(DimType val) { std::get<1>(*this) = val; }
  constexpr void setDir(bool val) { std::get<2>(*this) = val; }
  // implement operator = because tuple's operator = is not constexpr until 20
  constexpr auto& operator=(const DistAndDir& other){
    this->setDistance(other.distance());
    this->setDimension(other.dimension());
    this->setDir(other.isPositiveDir());
    return *this;
  }

};

struct DirectionAndSpeed : public std::tuple<MonteRay_GridBins::Direction_t, gpuRayFloat_t>{
  using std::tuple<MonteRay_GridBins::Direction_t, gpuRayFloat_t>::tuple;
  constexpr auto& direction() const { return std::get<0>(*this); }
  constexpr auto speed() const { return std::get<1>(*this); }

  constexpr void setDirection(MonteRay_GridBins::Direction_t& direction) { std::get<0>(*this) = direction; }
  constexpr void setSpeed(gpuRayFloat_t val) { std::get<1>(*this) = val; }
};


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
  CUDA_CALLABLE_MEMBER rayTraceList_t(RayWorkInfo& rayInfo, const unsigned threadID, int dim = -1);

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




} /* namespace MonteRay */
#endif /* MONTERAYGRIDSYSTEMINTERFACE_HH_ */

