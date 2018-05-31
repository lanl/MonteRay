/*
 * MonteRaySphericalGrid.hh
 *
 *  Created on: Feb 2, 2018
 *      Author: jsweezy
 */

#ifndef MONTERAYSPHERICALGRID_HH_
#define MONTERAYSPHERICALGRID_HH_

#include "MonteRayDefinitions.hh"
#include "MonteRay_GridSystemInterface.hh"
#include "MonteRay_SingleValueCopyMemory.hh"

namespace MonteRay {

#ifdef __CUDACC__
class MonteRay_SphericalGrid;

using ptrSphericalGrid_result_t = MonteRay_SingleValueCopyMemory<MonteRay_SphericalGrid*>;

CUDA_CALLABLE_KERNEL
void createDeviceInstance(MonteRay_SphericalGrid** pPtrInstance, ptrSphericalGrid_result_t* pResult, MonteRay_GridBins* pGridR );


CUDA_CALLABLE_KERNEL
void deleteDeviceInstance(MonteRay_SphericalGrid** pInstance);
#endif

class MonteRay_SphericalGrid : public MonteRay_GridSystemInterface {
public:
    typedef MonteRay_GridBins::Position_t Position_t;
    typedef MonteRay_GridBins::Direction_t Direction_t;

    enum coord {R=0,DimMax=1};

    using GridBins_t = MonteRay_GridBins;
    //typedef singleDimRayTraceMap_t;
    using pGridInfo_t = GridBins_t*;
    using pArrayOfpGridInfo_t = pGridInfo_t[3];

    CUDA_CALLABLE_MEMBER MonteRay_SphericalGrid(unsigned d, pArrayOfpGridInfo_t pBins);
    CUDA_CALLABLE_MEMBER MonteRay_SphericalGrid(unsigned d, GridBins_t* );

    CUDA_CALLABLE_MEMBER virtual ~MonteRay_SphericalGrid(void){
#ifndef __CUDA_ARCH__
    	if( ptrDevicePtr ) {
    		deleteDeviceInstance<<<1,1>>>( ptrDevicePtr );
    		cudaDeviceSynchronize();
    	}
    	MonteRayDeviceFree( ptrDevicePtr );
#endif
    }

    CUDAHOST_CALLABLE_MEMBER void copyToGPU(void) {
    	if( debug ) std::cout << "Debug: MonteRay_SphericalGrid::copyToGPU \n";
#ifdef __CUDACC__
    	ptrDevicePtr = (MonteRay_SphericalGrid**) MONTERAYDEVICEALLOC(sizeof(MonteRay_SphericalGrid*), std::string("device - MonteRay_SphericalGrid::ptrDevicePtr") );

    	pRVertices->copyToGPU();

    	std::unique_ptr<ptrSphericalGrid_result_t> ptrResult = std::unique_ptr<ptrSphericalGrid_result_t>( new ptrSphericalGrid_result_t() );
    	ptrResult->copyToGPU();

    	createDeviceInstance<<<1,1>>>( ptrDevicePtr, ptrResult->devicePtr, pRVertices->devicePtr );
    	cudaDeviceSynchronize();
    	ptrResult->copyToCPU();
    	devicePtr = ptrResult->v;

#endif
	}

    CUDAHOST_CALLABLE_MEMBER
    MonteRay_SphericalGrid* getDeviceInstancePtr();

    CUDA_CALLABLE_MEMBER void validate(void);
    CUDA_CALLABLE_MEMBER void validateR(void);

    CUDA_CALLABLE_MEMBER unsigned getNumRBins() const { return numRBins; }
    CUDA_CALLABLE_MEMBER unsigned getNumBins( unsigned d) const { return d == 0 ? numRBins : 0; }

    CUDA_CALLABLE_MEMBER gpuRayFloat_t getRVertex(unsigned i) const { return pRVertices->vertices[i]; }
    CUDA_CALLABLE_MEMBER gpuRayFloat_t getRSqVertex(unsigned i) const { return pRVertices->verticesSq[i]; }

    CUDA_CALLABLE_MEMBER Position_t convertFromCartesian( const Position_t& pos) const;

    CUDA_CALLABLE_MEMBER int getRadialIndexFromR( gpuRayFloat_t R ) const { return pRVertices->getRadialIndexFromR(R); }
    CUDA_CALLABLE_MEMBER int getRadialIndexFromRSq( gpuRayFloat_t RSq ) const { return pRVertices->getRadialIndexFromRSq(RSq); }

    CUDA_CALLABLE_MEMBER unsigned getIndex( const GridBins_t::Position_t& particle_pos) const;
    CUDA_CALLABLE_MEMBER bool isIndexOutside( unsigned d,  int i) const {
    	MONTERAY_VERIFY( d == 0, "MonteRay_SphericalGrid::isIndexOutside -- Index i must not be negative." );
    	MONTERAY_VERIFY( d == 0, "MonteRay_SphericalGrid::isIndexOutside -- Dimension d must be 0 because spherical geometry is 1-D." );
    	return pRVertices->isIndexOutside(i);
    }

    CUDA_CALLABLE_MEMBER bool isOutside(  const int i[]) const;

    CUDA_CALLABLE_MEMBER unsigned calcIndex( const int[] ) const;

    CUDA_CALLABLE_MEMBER uint3 calcIJK( unsigned index ) const { return {index, 0, 0}; }

    CUDA_CALLABLE_MEMBER gpuRayFloat_t getVolume( unsigned index ) const;

    CUDA_CALLABLE_MEMBER
    void
    rayTrace( rayTraceList_t& rayTraceList, const GridBins_t::Position_t&, const GridBins_t::Position_t&, gpuRayFloat_t distance,  bool outsideDistances=false) const;

    CUDA_CALLABLE_MEMBER
    void
    crossingDistance(singleDimRayTraceMap_t& rayTraceMap, const GridBins_t::Position_t& pos, const GridBins_t::Direction_t& dir, gpuRayFloat_t distance ) const;

    CUDA_CALLABLE_MEMBER
    void
    radialCrossingDistances(singleDimRayTraceMap_t& rayTraceMap, const Position_t& pos, const Direction_t& dir, unsigned rIndex, gpuRayFloat_t distance ) const;

    CUDA_CALLABLE_MEMBER
    void
    radialCrossingDistances( singleDimRayTraceMap_t& rayTraceMap, const Position_t& pos, const Direction_t& dir, gpuRayFloat_t distance ) const;

    CUDA_CALLABLE_MEMBER
    void
    radialCrossingDistancesSingleDirection( singleDimRayTraceMap_t& rayTraceMap, const Position_t& pos, const Direction_t& dir, gpuRayFloat_t distance, bool outward ) const;

protected:
    CUDA_CALLABLE_MEMBER gpuRayFloat_t calcParticleRSq( const gpuRayFloat_t&  pos) const { return pos*pos; }
    CUDA_CALLABLE_MEMBER gpuRayFloat_t calcParticleRSq( const Position_t&  pos) const { return pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]; }
    CUDA_CALLABLE_MEMBER gpuRayFloat_t calcQuadraticA(  const Direction_t& dir) const { return 1.0; } //=dot(pDirection,pDirection)
    CUDA_CALLABLE_MEMBER gpuRayFloat_t calcQuadraticB(  const Position_t&  pos, const Direction_t& dir) const { return 2.0*(pos[0]*dir[0] + pos[1]*dir[1] + pos[2]*dir[2]); }

public:
    MonteRay_SphericalGrid** ptrDevicePtr = nullptr;
    MonteRay_SphericalGrid* devicePtr = nullptr;

private:
    pGridInfo_t pRVertices = nullptr;
    unsigned numRBins = 0;
    //bool regular = false;

    const bool debug = false;
};

} /* namespace MonteRay */

#endif /* MONTERAYSPHERICALGRID_HH_ */
