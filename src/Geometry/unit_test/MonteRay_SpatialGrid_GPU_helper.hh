/*
 * MonteRay_SpatialGrid_helper.hh
 *
 *  Created on: Feb 13, 2018
 *      Author: jsweezy
 */

#ifndef MONTERAY_SPATIALGRID_HELPER_HH_
#define MONTERAY_SPATIALGRID_HELPER_HH_

#include "MonteRay_SpatialGrid.hh"
#include "GPUSync.hh"
#include "GPUUtilityFunctions.hh"
#include "MonteRay_GridSystemInterface.hh"
#include "MonteRay_SingleValueCopyMemory.hh"
#include "MonteRayDefinitions.hh"

namespace MonteRay_SpatialGrid_helper {

using namespace MonteRay;

	typedef MonteRay_SpatialGrid Grid_t;
	using Position_t = MonteRay_SpatialGrid::Position_t;

	template<typename T>
	using resultClass = MonteRay_SingleValueCopyMemory<T>;

   	CUDA_CALLABLE_KERNEL void kernelGetNumCells(Grid_t* pSpatialGrid, resultClass<unsigned>* pResult) ;

   	CUDA_CALLABLE_KERNEL void kernelGetCoordinateSystem(Grid_t* pSpatialGrid, resultClass<TransportMeshTypeEnum::TransportMeshTypeEnum_t>* pResult);

   	CUDA_CALLABLE_KERNEL void kernelGetDimension(Grid_t* pSpatialGrid, resultClass<unsigned>* pResult);

   	CUDA_CALLABLE_KERNEL void kernelIsInitialized(Grid_t* pSpatialGrid, resultClass<bool>* pResult);

   	CUDA_CALLABLE_KERNEL void kernelGetNumGridBins(Grid_t* pSpatialGrid, resultClass<unsigned>* pResult, unsigned index);

   	CUDA_CALLABLE_KERNEL void kernelGetMinVertex(Grid_t* pSpatialGrid, resultClass<gpuRayFloat_t>* pResult, unsigned index);

   	CUDA_CALLABLE_KERNEL void kernelGetMaxVertex(Grid_t* pSpatialGrid, resultClass<gpuRayFloat_t>* pResult, unsigned index);

   	CUDA_CALLABLE_KERNEL void kernelGetDelta(Grid_t* pSpatialGrid, resultClass<gpuRayFloat_t>* pResult, unsigned index);

   	CUDA_CALLABLE_KERNEL void kernelGetVertex(Grid_t* pSpatialGrid, resultClass<gpuRayFloat_t>* pResult, unsigned d, unsigned index);

   	CUDA_CALLABLE_KERNEL void kernelGetVolume(Grid_t* pSpatialGrid, resultClass<gpuRayFloat_t>* pResult, unsigned index);

   	CUDA_CALLABLE_KERNEL void kernelGetIndex(Grid_t* pSpatialGrid, resultClass<unsigned>* pResult, Position_t pos);

   	template<typename particle>
   	CUDA_CALLABLE_KERNEL void kernelGetIndexByParticle(Grid_t* pSpatialGrid, resultClass<unsigned>* pResult, particle p) {
   		pResult->v = pSpatialGrid->getIndex(p);
   	}

  	//CUDA_CALLABLE_KERNEL void kernelRayTrace(Grid_t* pSpatialGrid, resultClass<unsigned>* pResult, Position_t pos, Position_t dir, gpuRayFloat_t distance);
  	CUDA_CALLABLE_KERNEL void kernelRayTrace(Grid_t* pSpatialGrid, resultClass<rayTraceList_t>* pResult,
  	   			gpuRayFloat_t x, gpuRayFloat_t y, gpuRayFloat_t z, gpuRayFloat_t u, gpuRayFloat_t v, gpuRayFloat_t w,
  	   			gpuRayFloat_t distance, bool outside = false);

  	CUDA_CALLABLE_KERNEL void kernelCrossingDistance(Grid_t* pSpatialGrid, resultClass<singleDimRayTraceMap_t>* pResult,
  			unsigned d, gpuRayFloat_t pos, gpuRayFloat_t dir, gpuRayFloat_t distance );

   	template<class Particle>
   	CUDA_CALLABLE_KERNEL void kernelRayTraceParticle(Grid_t* pSpatialGrid, resultClass<rayTraceList_t>* pResult,
   			Particle p,
   			gpuRayFloat_t distance, bool outside = false) {
   		pSpatialGrid->rayTrace( pResult->v, p, distance, outside);
   	}

   	class SpatialGridGPUTester {
   	public:
   		SpatialGridGPUTester(){
   			pGridInfo = std::unique_ptr<Grid_t>( new Grid_t() );
   	    	cudaDeviceSetLimit( cudaLimitStackSize, 40960 );
   		}

   		~SpatialGridGPUTester(){}

   		void cartesianGrid1_setup(void) {
   			pGridInfo = std::unique_ptr<Grid_t>( new Grid_t() );
   			pGridInfo->setCoordinateSystem( TransportMeshTypeEnum::Cartesian );
   			pGridInfo->setDimension( 3 );
   			pGridInfo->setGrid( MonteRay_SpatialGrid::CART_X, -10.0, 10.0, 100);
   			pGridInfo->setGrid( MonteRay_SpatialGrid::CART_Y, -20.0, 20.0, 100);
   			pGridInfo->setGrid( MonteRay_SpatialGrid::CART_Z, -30.0, 30.0, 100);
   			pGridInfo->initialize();

   			pGridInfo->copyToGPU();
   		}

   		void sphericalGrid1_setup(void) {
   			pGridInfo = std::unique_ptr<Grid_t>( new Grid_t() );
   			pGridInfo->setCoordinateSystem( TransportMeshTypeEnum::Spherical );
   			pGridInfo->setDimension( 1 );
   			pGridInfo->setGrid( MonteRay_SpatialGrid::SPH_R, 0.0, 10.0, 100);
   			pGridInfo->initialize();

   			pGridInfo->copyToGPU();
   		}

   		void initialize() {
   			pGridInfo->initialize();
   		}

   		void copyToGPU() {
   			pGridInfo->copyToGPU();
   		}

   		void setGrid(unsigned index, const std::vector<gpuRayFloat_t>& vertices ) {
   			pGridInfo->setGrid(index, vertices);
   		}

   		void setGrid(unsigned index, gpuRayFloat_t min, gpuRayFloat_t max, unsigned numBins ) {
   			pGridInfo->setGrid(index, min, max, numBins);
   		}

   		void setCoordinateSystem(TransportMeshTypeEnum::TransportMeshTypeEnum_t system) {
   			pGridInfo->setCoordinateSystem(system);
   		}

   		void setDimension( unsigned dim) {
   			pGridInfo->setDimension(dim);
   		}

   	   	int getNumCells( void ) {
   	   		using result_t = resultClass<unsigned>;
   	   	    std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   	    pResult->copyToGPU();

   	   	    kernelGetNumCells<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr);
   	   		gpuErrchk( cudaPeekAtLastError() );
   	   		pResult->copyToCPU();
   	   		return pResult->v;
   	   	}

   	   	unsigned getDimension( void ) {
   	   		using result_t = resultClass<unsigned>;
   	   		std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   		pResult->copyToGPU();

   	   		kernelGetDimension<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr);
   	   		gpuErrchk( cudaPeekAtLastError() );
   	   		pResult->copyToCPU();
   	   		return pResult->v;
   	   	}

   	   	TransportMeshTypeEnum::TransportMeshTypeEnum_t getCoordinateSystem( void ) const {
   	   		using result_t = resultClass<TransportMeshTypeEnum::TransportMeshTypeEnum_t>;
   	   	    std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   	    pResult->copyToGPU();

   	   	    kernelGetCoordinateSystem<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr);
   	   		gpuErrchk( cudaPeekAtLastError() );
   	   		pResult->copyToCPU();
   	   		return pResult->v;
   	   	}

   	   	bool isInitialized( void ) const {
   	   		using result_t = resultClass<bool>;
   	   		std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   		pResult->copyToGPU();

   	   		kernelIsInitialized<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr);
   	   		gpuErrchk( cudaPeekAtLastError() );
   	   		pResult->copyToCPU();
   	   		return pResult->v;
   	   	}

   	   	unsigned getNumGridBins( unsigned index ) const {
   	   		using result_t = resultClass<unsigned>;
   	   		std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   		pResult->copyToGPU();

   	   		kernelGetNumGridBins<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr, index);
   	   		gpuErrchk( cudaPeekAtLastError() );
   	   		pResult->copyToCPU();
   	   		return pResult->v;
   	   	}

   	   	gpuRayFloat_t getMinVertex( unsigned index ) const {
   	   		using result_t = resultClass<gpuRayFloat_t>;
   	   		std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   		pResult->copyToGPU();

   	   		kernelGetMinVertex<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr, index);
   	   		gpuErrchk( cudaPeekAtLastError() );
   	   		pResult->copyToCPU();
   	   		return pResult->v;
   	   	}

   	   	gpuRayFloat_t getMaxVertex( unsigned index ) const {
   	   		using result_t = resultClass<gpuRayFloat_t>;
   	   		std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   		pResult->copyToGPU();

   	   		kernelGetMaxVertex<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr, index);
   	   		gpuErrchk( cudaPeekAtLastError() );
   	   		pResult->copyToCPU();
   	   		return pResult->v;
   	   	}

   	   	gpuRayFloat_t getDelta( unsigned index ) const {
   	   		using result_t = resultClass<gpuRayFloat_t>;
   	   		std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   		pResult->copyToGPU();

   	   		kernelGetDelta<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr, index);
   	   		gpuErrchk( cudaPeekAtLastError() );
   	   		pResult->copyToCPU();
   	   		return pResult->v;
   	   	}

   	   	gpuRayFloat_t getVertex(unsigned d, unsigned index ) const {
   	   		using result_t = resultClass<gpuRayFloat_t>;
   	   		std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   		pResult->copyToGPU();

   	   		kernelGetVertex<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr, d, index);
   	   		gpuErrchk( cudaPeekAtLastError() );
   	   		pResult->copyToCPU();
   	   		return pResult->v;
   	   	}

   	   	gpuRayFloat_t getVolume( unsigned index ) const {
   	   		using result_t = resultClass<gpuRayFloat_t>;
   	   		std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   		pResult->copyToGPU();

   	   		kernelGetVolume<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr, index);
   	   		gpuErrchk( cudaPeekAtLastError() );
   	   		pResult->copyToCPU();
   	   		return pResult->v;
   	   	}

   	   	unsigned getIndex(Position_t pos ) const {
   	   		using result_t = resultClass<unsigned>;
   	   		std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   		pResult->copyToGPU();

   	   		kernelGetIndex<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr, pos );
   	   		gpuErrchk( cudaPeekAtLastError() );
   	   		pResult->copyToCPU();
   	   		return pResult->v;
   	   	}

   	   	template<typename particle>
  	   	unsigned getIndex(particle& p) const {
   	   		using result_t = resultClass<unsigned>;
   	   		std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   		pResult->copyToGPU();

   	   		kernelGetIndexByParticle<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr, p );
   	   		gpuErrchk( cudaPeekAtLastError() );
   	   		pResult->copyToCPU();
   	   		return pResult->v;
   	   	}

   	   	rayTraceList_t rayTrace( Position_t pos, Position_t dir, gpuRayFloat_t distance, bool outside=false ) {

   	   		using result_t = resultClass<rayTraceList_t>;
   	   		std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   		pResult->copyToGPU();

   	   		cudaDeviceSynchronize();
   	   		kernelRayTrace<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr,
   	   				                 pos[0], pos[1], pos[2], dir[0], dir[1], dir[2], distance, outside );
   	   		cudaDeviceSynchronize();

   	   		gpuErrchk( cudaPeekAtLastError() );

   	   		pResult->copyToCPU();

   	   		return pResult->v;
   	   	}

   	   	singleDimRayTraceMap_t crossingDistance( unsigned d, gpuRayFloat_t pos, gpuRayFloat_t dir, gpuRayFloat_t distance  ) {

   	   		using result_t = resultClass<singleDimRayTraceMap_t>;
   	   		std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   		pResult->copyToGPU();

   	   		cudaDeviceSynchronize();
   	   		kernelCrossingDistance<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr,
   	   				                 d, pos, dir, distance );
   	   		cudaDeviceSynchronize();

   	   		gpuErrchk( cudaPeekAtLastError() );

   	   		pResult->copyToCPU();

   	   		return pResult->v;
   	   	}

   	   	template<typename particle>
  	   	rayTraceList_t rayTrace( particle& p, gpuRayFloat_t distance, bool outside = false) {

   	   		using result_t = resultClass<rayTraceList_t>;
   	   		std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   		pResult->copyToGPU();

   	   		cudaDeviceSynchronize();
   	   		kernelRayTraceParticle<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr,
   	   				                 p, distance, outside );
   	   		cudaDeviceSynchronize();

   	   		gpuErrchk( cudaPeekAtLastError() );

   	   		pResult->copyToCPU();

   	   		return pResult->v;
   	   	}

   	   	void read( const std::string& fileName ) {
   	   		pGridInfo->read(fileName);
   	   	}

   	   	std::unique_ptr<Grid_t> pGridInfo;
   	};

   	class particle {
    public:
    	CUDA_CALLABLE_MEMBER particle(void){};

        Position_t pos;
        Position_t dir;

        CUDA_CALLABLE_MEMBER
        MonteRay_SpatialGrid::Position_t getPosition(void) const { return pos; }

        CUDA_CALLABLE_MEMBER
        MonteRay_SpatialGrid::Position_t getDirection(void) const { return dir; }
    };

}

#endif /* MONTERAY_SPATIALGRID_HELPER_HH_ */
