/*
 * MonteRayCartesianGrid.hh
 *
 *  Created on: Feb 2, 2018
 *      Author: jsweezy
 */

#ifndef MONTERAYCARTESIANGRID_HH_
#define MONTERAYCARTESIANGRID_HH_

#include "MonteRayDefinitions.hh"
#include "MonteRay_GridSystemInterface.hh"

namespace MonteRay {

#ifdef __CUDACC__
class MonteRay_CartesianGrid;

CUDA_CALLABLE_KERNEL
void createDeviceInstance(MonteRay_CartesianGrid** pPtrInstance, MonteRay_GridBins* pGridX, MonteRay_GridBins* pGridY, MonteRay_GridBins* pGridZ );

CUDA_CALLABLE_KERNEL
void deleteDeviceInstance(MonteRay_CartesianGrid* pInstance);
#endif

class MonteRay_CartesianGrid : public MonteRay_GridSystemInterface {
public:
    enum coord {X,Y,Z,DimMax};

    using GridBins_t = MonteRay_GridBins;
    //typedef singleDimRayTraceMap_t;
    using pGridInfo_t = GridBins_t*;
    using pArrayOfpGridInfo_t = pGridInfo_t[3];

    CUDA_CALLABLE_MEMBER MonteRay_CartesianGrid(unsigned d, pArrayOfpGridInfo_t pBins);
    CUDA_CALLABLE_MEMBER MonteRay_CartesianGrid(unsigned d, GridBins_t*, GridBins_t*, GridBins_t* );

    CUDA_CALLABLE_MEMBER virtual ~MonteRay_CartesianGrid(void){
#ifndef __CUDA_ARCH__
//    	if( devicePtr ) {
//    		deleteDeviceInstance<<<1,1>>>( devicePtr );
//    	}
#endif
    }

    CUDAHOST_CALLABLE_MEMBER void copyToGPU(void) {
    	if( debug ) std::cout << "Debug: MonteRay_CartesianGrid::copyToGPU \n";
#ifdef __CUDACC__
    	cudaMalloc(&devicePtr, sizeof(MonteRay_CartesianGrid*) );

    	pGridBins[0]->copyToGPU();
    	pGridBins[1]->copyToGPU();
    	pGridBins[2]->copyToGPU();
    	createDeviceInstance<<<1,1>>>( devicePtr, pGridBins[0]->devicePtr, pGridBins[1]->devicePtr, pGridBins[2]->devicePtr );
#endif
	}

    CUDA_CALLABLE_MEMBER unsigned getIndex( const GridBins_t::Position_t& particle_pos) const;

    CUDA_CALLABLE_MEMBER int getDimIndex( unsigned d, gpuFloatType_t pos) const {  return pGridBins[d]->getLinearIndex( pos ); }

    CUDA_CALLABLE_MEMBER gpuFloatType_t getVolume( unsigned index ) const;

    CUDA_CALLABLE_MEMBER unsigned calcIndex( const int[] ) const;

    CUDA_CALLABLE_MEMBER uint3 calcIJK( unsigned index ) const;

    CUDA_CALLABLE_MEMBER bool isOutside(  const int i[]) const;

    CUDA_CALLABLE_MEMBER bool isIndexOutside( unsigned d,  int i) const { return pGridBins[d]->isIndexOutside(i);}

    CUDA_CALLABLE_MEMBER unsigned getNumBins( unsigned d) const;


    CUDA_CALLABLE_MEMBER
    void
    rayTrace( rayTraceList_t&, const GridBins_t::Position_t&, const GridBins_t::Position_t&, gpuFloatType_t distance,  bool outsideDistances=false) const;

    CUDA_CALLABLE_MEMBER
    void
    crossingDistance(singleDimRayTraceMap_t&, unsigned d, gpuFloatType_t pos, gpuFloatType_t dir, gpuFloatType_t distance ) const;

    CUDA_CALLABLE_MEMBER
    void
    crossingDistance(singleDimRayTraceMap_t&, const GridBins_t& Bins, gpuFloatType_t pos, gpuFloatType_t dir, gpuFloatType_t distance, bool equal_spacing=false) const;

public:
    MonteRay_CartesianGrid** devicePtr;

private:
    pArrayOfpGridInfo_t pGridBins;
    //bool regular = false;

    const bool debug = false;
};

} /* namespace MonteRay */

#endif /* MONTERAYCARTESIANGRID_HH_ */
