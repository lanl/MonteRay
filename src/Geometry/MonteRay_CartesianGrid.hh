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
#include "MonteRay_SingleValueCopyMemory.hh"

namespace MonteRay {

class MonteRay_CartesianGrid;

using ptrCartesianGrid_result_t = MonteRay_SingleValueCopyMemory<MonteRay_CartesianGrid*>;

CUDA_CALLABLE_KERNEL
void createDeviceInstance(MonteRay_CartesianGrid** pPtrInstance, ptrCartesianGrid_result_t* pResult, MonteRay_GridBins* pGridX, MonteRay_GridBins* pGridY, MonteRay_GridBins* pGridZ );


CUDA_CALLABLE_KERNEL
void deleteDeviceInstance(MonteRay_CartesianGrid** pInstance);

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
#ifdef __CUDACC__
#ifndef __CUDA_ARCH__
    	if( ptrDevicePtr ) {
    		deleteDeviceInstance<<<1,1>>>( ptrDevicePtr );
    		cudaDeviceSynchronize();
    	}
    	MonteRayDeviceFree( ptrDevicePtr );
#endif
#endif
    }

    CUDAHOST_CALLABLE_MEMBER void copyToGPU(void) {
    	if( debug ) std::cout << "Debug: MonteRay_CartesianGrid::copyToGPU \n";
#ifdef __CUDACC__
    	ptrDevicePtr = (MonteRay_CartesianGrid**) MONTERAYDEVICEALLOC(sizeof(MonteRay_CartesianGrid*), std::string("device - MonteRay_CartesianGrid::ptrDevicePtr") );

    	pGridBins[0]->copyToGPU();
    	pGridBins[1]->copyToGPU();
    	pGridBins[2]->copyToGPU();

    	std::unique_ptr<ptrCartesianGrid_result_t> ptrResult = std::unique_ptr<ptrCartesianGrid_result_t>( new ptrCartesianGrid_result_t() );
    	ptrResult->copyToGPU();

    	createDeviceInstance<<<1,1>>>( ptrDevicePtr, ptrResult->devicePtr, pGridBins[0]->devicePtr, pGridBins[1]->devicePtr, pGridBins[2]->devicePtr );
    	cudaDeviceSynchronize();
    	ptrResult->copyToCPU();
    	devicePtr = ptrResult->v;

#endif
	}

    CUDAHOST_CALLABLE_MEMBER
    MonteRay_CartesianGrid* getDeviceInstancePtr();

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
