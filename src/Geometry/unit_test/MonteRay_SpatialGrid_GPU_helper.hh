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

   	CUDA_CALLABLE_KERNEL void kernelGetMinVertex(Grid_t* pSpatialGrid, resultClass<gpuFloatType_t>* pResult, unsigned index);

   	CUDA_CALLABLE_KERNEL void kernelGetMaxVertex(Grid_t* pSpatialGrid, resultClass<gpuFloatType_t>* pResult, unsigned index);

   	CUDA_CALLABLE_KERNEL void kernelGetDelta(Grid_t* pSpatialGrid, resultClass<gpuFloatType_t>* pResult, unsigned index);

   	CUDA_CALLABLE_KERNEL void kernelGetVertex(Grid_t* pSpatialGrid, resultClass<gpuFloatType_t>* pResult, unsigned d, unsigned index);

   	CUDA_CALLABLE_KERNEL void kernelGetVolume(Grid_t* pSpatialGrid, resultClass<gpuFloatType_t>* pResult, unsigned index);

   	CUDA_CALLABLE_KERNEL void kernelGetIndex(Grid_t* pSpatialGrid, resultClass<unsigned>* pResult, Position_t pos);

   	template<typename particle>
   	CUDA_CALLABLE_KERNEL void kernelGetIndexByParticle(Grid_t* pSpatialGrid, resultClass<unsigned>* pResult, particle p) {
   		pResult->v = pSpatialGrid->getIndex(p);
   	}

//   	CUDA_CALLABLE_KERNEL void kernelRayTrace(Grid_t* pSpatialGrid, resultClass<rayTraceList_t>* pResult, Position_t pos, Position_t dir, gpuFloatType_t distance);
  	CUDA_CALLABLE_KERNEL void kernelRayTrace(Grid_t* pSpatialGrid, Position_t pos, Position_t dir, gpuFloatType_t distance);

   	class SpatialGridGPUTester {
   	public:
   		SpatialGridGPUTester(){
   			pGridInfo = std::unique_ptr<Grid_t>( new Grid_t() );
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

   		void initialize() {
   			pGridInfo->initialize();
   		}

   		void copyToGPU() {
   			pGridInfo->copyToGPU();
   		}

   		void setGrid(unsigned index, const std::vector<gpuFloatType_t>& vertices ) {
   			pGridInfo->setGrid(index, vertices);
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

   	   	gpuFloatType_t getMinVertex( unsigned index ) const {
   	   		using result_t = resultClass<gpuFloatType_t>;
   	   		std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   		pResult->copyToGPU();

   	   		kernelGetMinVertex<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr, index);
   	   		gpuErrchk( cudaPeekAtLastError() );
   	   		pResult->copyToCPU();
   	   		return pResult->v;
   	   	}

   	   	gpuFloatType_t getMaxVertex( unsigned index ) const {
   	   		using result_t = resultClass<gpuFloatType_t>;
   	   		std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   		pResult->copyToGPU();

   	   		kernelGetMaxVertex<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr, index);
   	   		gpuErrchk( cudaPeekAtLastError() );
   	   		pResult->copyToCPU();
   	   		return pResult->v;
   	   	}

   	   	gpuFloatType_t getDelta( unsigned index ) const {
   	   		using result_t = resultClass<gpuFloatType_t>;
   	   		std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   		pResult->copyToGPU();

   	   		kernelGetDelta<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr, index);
   	   		gpuErrchk( cudaPeekAtLastError() );
   	   		pResult->copyToCPU();
   	   		return pResult->v;
   	   	}

   	   	gpuFloatType_t getVertex(unsigned d, unsigned index ) const {
   	   		using result_t = resultClass<gpuFloatType_t>;
   	   		std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   		pResult->copyToGPU();

   	   		kernelGetVertex<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr, d, index);
   	   		gpuErrchk( cudaPeekAtLastError() );
   	   		pResult->copyToCPU();
   	   		return pResult->v;
   	   	}

   	   	gpuFloatType_t getVolume( unsigned index ) const {
   	   		using result_t = resultClass<gpuFloatType_t>;
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

   	   	rayTraceList_t rayTrace( Position_t pos, Position_t dir, gpuFloatType_t distance ) {
   	   		std::cout << "Debug: MonteRay_SpatialGrid_GPU_helper -- rayTrace - 1\n";

   	   		using result_t = resultClass<rayTraceList_t>;
   	   		std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
   	   		pResult->copyToGPU();
   	   		pGridInfo->copyToGPU();

   	   		std::cout << "Debug: MonteRay_SpatialGrid_GPU_helper -- rayTrace - 2\n";
   	   		cudaDeviceSynchronize();
   	   		//kernelRayTrace<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr, pos, dir, distance );
   	   	    kernelRayTrace<<<1,1>>>( pGridInfo->devicePtr, pos, dir, distance );
   	   		cudaDeviceSynchronize();

   	   		std::cout << "Debug: MonteRay_SpatialGrid_GPU_helper -- rayTrace - 3\n";

   	   		gpuErrchk( cudaPeekAtLastError() );

   	   		std::cout << "Debug: MonteRay_SpatialGrid_GPU_helper -- rayTrace - 4\n";

   	   		pResult->copyToCPU();

   	   		std::cout << "Debug: MonteRay_SpatialGrid_GPU_helper -- rayTrace - 5\n";
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
