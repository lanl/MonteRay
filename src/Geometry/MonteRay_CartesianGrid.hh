/*
 * MonteRayCartesianGrid.hh
 *
 *  Created on: Feb 2, 2018
 *      Author: jsweezy
 */

#ifndef MONTERAYCARTESIANGRID_HH_
#define MONTERAYCARTESIANGRID_HH_

#include "MonteRayTypes.hh"
#include "MonteRay_GridSystemInterface.hh"

namespace MonteRay {

class MonteRay_CartesianGrid;

class MonteRay_CartesianGrid : public MonteRay_GridSystemInterface {
public:
    enum coord {X,Y,Z,DimMax};

    using GridBins_t = MonteRay_GridBins;
    //typedef singleDimRayTraceMap_t;
    using pGridInfo_t = GridBins_t*;
    using pArrayOfpGridInfo_t = pGridInfo_t[3];

    CUDA_CALLABLE_MEMBER MonteRay_CartesianGrid(unsigned d, pArrayOfpGridInfo_t pBins);
    CUDA_CALLABLE_MEMBER MonteRay_CartesianGrid(unsigned d, GridBins_t*, GridBins_t*, GridBins_t* );

    CUDA_CALLABLE_MEMBER virtual ~MonteRay_CartesianGrid(void);

    CUDAHOST_CALLABLE_MEMBER void copyToGPU(void);

    CUDAHOST_CALLABLE_MEMBER MonteRay_CartesianGrid* getDeviceInstancePtr();

    CUDA_CALLABLE_MEMBER unsigned getIndex( const GridBins_t::Position_t& particle_pos) const;

    CUDA_CALLABLE_MEMBER int getDimIndex( unsigned d, gpuRayFloat_t pos) const {  return pGridBins[d]->getLinearIndex( pos ); }

    CUDA_CALLABLE_MEMBER gpuRayFloat_t getVolume( unsigned index ) const;

    CUDA_CALLABLE_MEMBER unsigned calcIndex( const int[] ) const;

    CUDA_CALLABLE_MEMBER uint3 calcIJK( unsigned index ) const;

    CUDA_CALLABLE_MEMBER bool isOutside(  const int i[]) const;

    CUDA_CALLABLE_MEMBER bool isIndexOutside( unsigned d,  int i) const { return pGridBins[d]->isIndexOutside(i);}

    CUDA_CALLABLE_MEMBER unsigned getNumBins( unsigned d) const;

    CUDA_CALLABLE_MEMBER
    void
    rayTrace( rayTraceList_t&, const GridBins_t::Position_t&, const GridBins_t::Position_t&, gpuRayFloat_t distance,  bool outsideDistances=false) const;

    CUDA_CALLABLE_MEMBER
    void
    crossingDistance(singleDimRayTraceMap_t&, unsigned d, gpuRayFloat_t pos, gpuRayFloat_t dir, gpuRayFloat_t distance ) const;

    CUDA_CALLABLE_MEMBER
    void
    crossingDistance(singleDimRayTraceMap_t&, const GridBins_t& Bins, gpuRayFloat_t pos, gpuRayFloat_t dir, gpuRayFloat_t distance, bool equal_spacing=false) const;

public:
    MonteRay_CartesianGrid** ptrDevicePtr = nullptr;
    MonteRay_CartesianGrid* devicePtr = nullptr;

private:
    pArrayOfpGridInfo_t pGridBins;
    //bool regular = false;

    const bool debug = false;
};

} /* namespace MonteRay */

#endif /* MONTERAYCARTESIANGRID_HH_ */
