#ifndef RAYWORKINFO_HH_
#define RAYWORKINFO_HH_

#include <string>
#include <iostream>
#include "SimpleVector.hh"
#include "MonteRayAssert.hh"
#include "ManagedAllocator.hh"

namespace MonteRay{

template <template <typename... T> class Container = std::vector>
class RayWorkInfo_t: public Managed {
  public:
// Data
// // TODO: Make private
  unsigned size = 0;

  Container<int> indices;
  Container<int> rayCastSize;
  Container<int> rayCastCell;
  Container<gpuRayFloat_t> rayCastDistance;

  Container<int> crossingSize;
  Container<int> crossingCell;
  Container<gpuRayFloat_t> crossingDistance;

  /// Primary RayWorkInfo constructor.
  /// Takes the size of the list as an argument.
  CUDAHOST_CALLABLE_MEMBER RayWorkInfo_t(unsigned num){
    num = num ? num : 1;
    if( size > MONTERAY_MAX_THREADS ) {
        std::cout << "WARNING: Limiting MonteRay RayWorkInfo size, requested size="
                  << num << ", limit=" << MONTERAY_MAX_THREADS << "\n";
        num = MONTERAY_MAX_THREADS;
    }


    size = num;
    indices.resize(size*3);
    rayCastSize.resize(size);
    rayCastCell.resize(size*MAXNUMRAYCELLS);
    rayCastDistance.resize(size*MAXNUMRAYCELLS);

    crossingSize.resize(size*3);
    crossingCell.resize(size*MAXNUMVERTICES*3);
    crossingDistance.resize(size*MAXNUMVERTICES*3);
    clear();
  }
  CUDAHOST_CALLABLE_MEMBER RayWorkInfo_t(unsigned num, bool): RayWorkInfo_t(num) { }

  CUDA_CALLABLE_MEMBER const std::string className() { return {"RayWorkInfo"}; }

  CUDA_CALLABLE_MEMBER void clear(void) {
    for( auto& val : rayCastSize){ 
      val = 0;
    }
    for( auto& val : crossingSize ){
      val = 0;
    }
  }

  constexpr unsigned capacity(void) const {
      return size;
  }

  constexpr unsigned getIndex(unsigned dim, unsigned i) const {
      return indices[i+size*dim];
  }

  constexpr void setIndex(unsigned dim, unsigned i, int index) {
      indices[i+size*dim] = index;
  }

  constexpr void clear( unsigned i) {
      getRayCastSize(i) = 0;
      getCrossingSize(0, i) = 0;
      getCrossingSize(1, i) = 0;
      getCrossingSize(2, i) = 0;
  }

  constexpr void addRayCastCell(unsigned i, int cellID, gpuRayFloat_t dist){
    MONTERAY_ASSERT_MSG( dist >= 0, "distance must be > 0.0!" );
    getRayCastCell(i, getRayCastSize(i) ) = cellID;
    getRayCastDist(i, getRayCastSize(i) ) = dist;
    ++(getRayCastSize(i));
  }

  constexpr void addCrossingCell(unsigned dim, unsigned i, int cellID, gpuRayFloat_t dist){
    MONTERAY_ASSERT_MSG( dist >= 0, "distance must be > 0.0!" );
    /* printf("dim %d i %d getCrossingSize(dim, i) %d \n", dim, i, getCrossingSize(dim, i)); */
    getCrossingCell(dim, i, getCrossingSize(dim,i) ) = cellID;
    getCrossingDist(dim, i, getCrossingSize(dim,i) ) = dist;
    ++(getCrossingSize(dim,i));
  }

  constexpr int& getRayCastSize( const unsigned i) {
      return rayCastSize[i];
  }

  constexpr unsigned getRayCastIndex(unsigned i, unsigned cell) const {
      return cell*size + i;
  }

  constexpr int& getRayCastCell(unsigned i, unsigned cell)  {
      return rayCastCell[ getRayCastIndex(i,cell) ];
  }

  constexpr gpuRayFloat_t& getRayCastDist(unsigned i, unsigned cell)  {
      return rayCastDistance[ getRayCastIndex(i,cell) ];
  }

  constexpr  int& getCrossingSize(unsigned dim, unsigned i)  {
    /* printf("dim %d, size %d, i %d \n", dim, size, i); */
      return crossingSize[dim*size + i];
  }

  constexpr unsigned getCrossingIndex(unsigned dim, unsigned i, unsigned cell) const {
      /* printf("dim %d size %d MAXNUMVERTICES %d cell %d i %d \n", dim, int(size), MAXNUMVERTICES, cell, i); */
      return dim*size*MAXNUMVERTICES + cell*size + i;
  }

  constexpr int& getCrossingCell(unsigned dim, unsigned i, unsigned cell)  {
      /* printf("CROSSING INDEX %d CELL %d crossingCell.size %d \n", getCrossingIndex(dim, i, cell), cell, crossingCell.size()); */
      return crossingCell[ getCrossingIndex(dim,i,cell) ];
  }

  constexpr gpuRayFloat_t& getCrossingDist(unsigned dim, unsigned i, unsigned cell)  {
      return crossingDistance[ getCrossingIndex(dim,i,cell) ];
  }

};

class RayWorkInfo : public RayWorkInfo_t<SimpleVector>{
  public:
  using RayWorkInfo_t<SimpleVector>::RayWorkInfo_t;
};

} // end namespace

#endif /* RAYWORKINFO_HH_ */
