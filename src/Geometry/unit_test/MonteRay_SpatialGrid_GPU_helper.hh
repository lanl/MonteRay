/*
 * MonteRay_SpatialGrid_helper.hh
 *
 *  Created on: Feb 13, 2018
 *      Author: jsweezy
 */

#ifndef MONTERAY_SPATIALGRID_HELPER_HH_
#define MONTERAY_SPATIALGRID_HELPER_HH_

#include "MonteRay_SpatialGrid.hh"
#include "GPUUtilityFunctions.hh"
#include "MonteRay_GridSystemInterface.hh"
#include "MonteRay_SingleValueCopyMemory.t.hh"
#include "MonteRayDefinitions.hh"
#include "MonteRay_ReadLnk3dnt.hh"
#include "MonteRayCopyMemory.t.hh"
#include "RayWorkInfo.hh"

namespace MonteRay_SpatialGrid_helper {

using namespace MonteRay;

typedef MonteRay_SpatialGrid Grid_t;
using Position_t = MonteRay_SpatialGrid::Position_t;

template<typename T>
using resultClass = MonteRay_SingleValueCopyMemory<T>;

CUDA_CALLABLE_KERNEL  kernelGetNumCells(Grid_t* pSpatialGrid, resultClass<unsigned>* pResult) ;

CUDA_CALLABLE_KERNEL  kernelGetCoordinateSystem(Grid_t* pSpatialGrid, resultClass<TransportMeshTypeEnum::TransportMeshTypeEnum_t>* pResult);

CUDA_CALLABLE_KERNEL  kernelGetDimension(Grid_t* pSpatialGrid, resultClass<unsigned>* pResult);

CUDA_CALLABLE_KERNEL  kernelIsInitialized(Grid_t* pSpatialGrid, resultClass<bool>* pResult);

CUDA_CALLABLE_KERNEL  kernelGetNumGridBins(Grid_t* pSpatialGrid, resultClass<unsigned>* pResult, unsigned index);

CUDA_CALLABLE_KERNEL  kernelGetMinVertex(Grid_t* pSpatialGrid, resultClass<gpuRayFloat_t>* pResult, unsigned index);

CUDA_CALLABLE_KERNEL  kernelGetMaxVertex(Grid_t* pSpatialGrid, resultClass<gpuRayFloat_t>* pResult, unsigned index);

CUDA_CALLABLE_KERNEL  kernelGetDelta(Grid_t* pSpatialGrid, resultClass<gpuRayFloat_t>* pResult, unsigned index);

CUDA_CALLABLE_KERNEL  kernelGetVertex(Grid_t* pSpatialGrid, resultClass<gpuRayFloat_t>* pResult, unsigned d, unsigned index);

CUDA_CALLABLE_KERNEL  kernelGetVolume(Grid_t* pSpatialGrid, resultClass<gpuRayFloat_t>* pResult, unsigned index);

CUDA_CALLABLE_KERNEL  kernelGetIndex(Grid_t* pSpatialGrid, resultClass<unsigned>* pResult, Position_t pos);

template<typename particle>
CUDA_CALLABLE_KERNEL  kernelGetIndexByParticle(Grid_t* pSpatialGrid, resultClass<unsigned>* pResult, particle p) {
    pResult->v = pSpatialGrid->getIndex(p);
}

//CUDA_CALLABLE_KERNEL  kernelRayTrace(Grid_t* pSpatialGrid, resultClass<unsigned>* pResult, Position_t pos, Position_t dir, gpuRayFloat_t distance);
CUDA_CALLABLE_KERNEL  kernelRayTrace(Grid_t* pSpatialGrid, RayWorkInfo* pRayInfo,
        gpuRayFloat_t x, gpuRayFloat_t y, gpuRayFloat_t z, gpuRayFloat_t u, gpuRayFloat_t v, gpuRayFloat_t w,
        gpuRayFloat_t distance, bool outside = false);

CUDA_CALLABLE_KERNEL  kernelCrossingDistance(Grid_t* pSpatialGrid, RayWorkInfo* pRayInfo,
        unsigned d, gpuRayFloat_t pos, gpuRayFloat_t dir, gpuRayFloat_t distance );

CUDA_CALLABLE_KERNEL  kernelCrossingDistance(Grid_t* pSpatialGrid, RayWorkInfo* pRayInfo,
        unsigned d, Position_t pos, Position_t dir, gpuRayFloat_t distance );

template<class Particle>
CUDA_CALLABLE_KERNEL  kernelRayTraceParticle(Grid_t* pSpatialGrid, RayWorkInfo* pRayInfo,
        Particle p,
        gpuRayFloat_t distance, bool outside = false) {
    pSpatialGrid->rayTrace( 0, *pRayInfo, p, distance, outside);
}

class SpatialGridGPUTester {
public:
    SpatialGridGPUTester(){
#ifdef __CUDACC__
        //setCudaStackSize( 100000 );
#endif
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

    void sphericalGrid1_setup(void) {
        pGridInfo = std::unique_ptr<Grid_t>( new Grid_t() );
        pGridInfo->setCoordinateSystem( TransportMeshTypeEnum::Spherical );
        pGridInfo->setDimension( 1 );
        pGridInfo->setGrid( MonteRay_SpatialGrid::SPH_R, 0.0, 10.0, 100);
        pGridInfo->initialize();

        pGridInfo->copyToGPU();
    }

    void cylindricalGrid_setup(const std::vector<gpuRayFloat_t>& Rverts, const std::vector<gpuRayFloat_t>& Zverts) {
        pGridInfo = std::unique_ptr<Grid_t>( new Grid_t() );
        pGridInfo->setCoordinateSystem( TransportMeshTypeEnum::Cylindrical );
        pGridInfo->setDimension( 2 );
        pGridInfo->setGrid(  MonteRay_SpatialGrid::CYLR_R, Rverts);
        pGridInfo->setGrid(  MonteRay_SpatialGrid::CYLR_Z, Zverts);
        pGridInfo->initialize();

        pGridInfo->copyToGPU();
    }

    void reader_setup(const std::string& filename) {
        MonteRay_ReadLnk3dnt readerObject( filename );
        pGridInfo = std::unique_ptr<Grid_t>( new Grid_t(readerObject) );
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

    void write( const std::string& fileName) const {
        pGridInfo->write(fileName);
    }

    int getNumCells( void ) {
        using result_t = resultClass<unsigned>;
        std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
        pResult->copyToGPU();
#ifdef __CUDACC__
        kernelGetNumCells<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr);
        gpuErrchk( cudaPeekAtLastError() );
#else
        kernelGetNumCells( pGridInfo->devicePtr, pResult->devicePtr);
#endif

        pResult->copyToCPU();
        return pResult->v;
    }

    unsigned getDimension( void ) {
        using result_t = resultClass<unsigned>;
        std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );

#ifdef __CUDACC__
        pResult->copyToGPU();
        kernelGetDimension<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr);
        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelGetDimension( pGridInfo.get(), pResult.get());
#endif

        return pResult->v;
    }

    TransportMeshTypeEnum::TransportMeshTypeEnum_t getCoordinateSystem( void ) const {
        using result_t = resultClass<TransportMeshTypeEnum::TransportMeshTypeEnum_t>;
        std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );

#ifdef __CUDACC__
        pResult->copyToGPU();
        kernelGetCoordinateSystem<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr);
        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelGetCoordinateSystem( pGridInfo.get(), pResult.get() );
#endif
        return pResult->v;
    }

    bool isInitialized( void ) const {
        using result_t = resultClass<bool>;
        std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
        pResult->copyToGPU();

#ifdef __CUDACC__
        kernelIsInitialized<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr);
        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelIsInitialized( pGridInfo.get(), pResult.get() );
#endif
        return pResult->v;
    }

    unsigned getNumGridBins( unsigned index ) const {
        using result_t = resultClass<unsigned>;
        std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );

#ifdef __CUDACC__
        pResult->copyToGPU();

        kernelGetNumGridBins<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr, index);
        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelGetNumGridBins( pGridInfo.get(), pResult.get(), index);
#endif
        return pResult->v;
    }

    gpuRayFloat_t getMinVertex( unsigned index ) const {
        using result_t = resultClass<gpuRayFloat_t>;
        std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );

#ifdef __CUDACC__
        pResult->copyToGPU();

        kernelGetMinVertex<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr, index);
        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelGetMinVertex( pGridInfo.get(), pResult.get(), index);
#endif
        return pResult->v;
    }

    gpuRayFloat_t getMaxVertex( unsigned index ) const {
        using result_t = resultClass<gpuRayFloat_t>;
        std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );

#ifdef __CUDACC__
        pResult->copyToGPU();

        kernelGetMaxVertex<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr, index);
        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelGetMaxVertex( pGridInfo.get(), pResult.get(), index);
#endif
        return pResult->v;
    }

    gpuRayFloat_t getDelta( unsigned index ) const {
        using result_t = resultClass<gpuRayFloat_t>;
        std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );

#ifdef __CUDACC__
        pResult->copyToGPU();

        kernelGetDelta<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr, index);
        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelGetDelta( pGridInfo.get(), pResult.get(), index);
#endif
        return pResult->v;
    }

    gpuRayFloat_t getVertex(unsigned d, unsigned index ) const {
        using result_t = resultClass<gpuRayFloat_t>;
        std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );

#ifdef __CUDACC__
        pResult->copyToGPU();

        kernelGetVertex<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr, d, index);
        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelGetVertex( pGridInfo.get(), pResult.get(), d, index);
#endif
        return pResult->v;
    }

    gpuRayFloat_t getVolume( unsigned index ) const {
        using result_t = resultClass<gpuRayFloat_t>;
        std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );

#ifdef __CUDACC__
        pResult->copyToGPU();

        kernelGetVolume<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr, index);
        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelGetVolume( pGridInfo.get(), pResult.get(), index);
#endif
        return pResult->v;
    }

    unsigned getIndex(Position_t pos ) const {
        using result_t = resultClass<unsigned>;
        std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );

#ifdef __CUDACC__
        pResult->copyToGPU();

        kernelGetIndex<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr, pos );
        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelGetIndex( pGridInfo.get(), pResult.get(), pos);
#endif
        return pResult->v;
    }

    template<typename particle>
    unsigned getIndex(particle& p) const {
        using result_t = resultClass<unsigned>;
        std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );

#ifdef __CUDACC__
        pResult->copyToGPU();

        kernelGetIndexByParticle<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr, p );
        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelGetIndexByParticle( pGridInfo, pResult, p);
#endif
        return pResult->v;
    }

    rayTraceList_t rayTrace( Position_t pos, Position_t dir, gpuRayFloat_t distance, bool outside=false ) {
        auto pRayInfo = std::make_unique<RayWorkInfo>(1);

#ifdef __CUDACC__

        cudaDeviceSynchronize();
        kernelRayTrace<<<1,1>>>( pGridInfo->devicePtr, pRayInfo.get(),
                pos[0], pos[1], pos[2], dir[0], dir[1], dir[2], distance, outside );
        cudaDeviceSynchronize();

        gpuErrchk( cudaPeekAtLastError() );

#else
        kernelRayTrace( pGridInfo.get(), pRayInfo.get(),
                pos[0], pos[1], pos[2], dir[0], dir[1], dir[2], distance, outside );
#endif

        rayTraceList_t rayTraceList;
        for( unsigned i = 0; i < pRayInfo->getRayCastSize(0); ++i ) {
            rayTraceList.add( pRayInfo->getRayCastCell(0,i), pRayInfo->getRayCastDist(0,i) );
        }
        return rayTraceList;
    }

    singleDimRayTraceMap_t crossingDistance( unsigned d, gpuRayFloat_t pos, gpuRayFloat_t dir, gpuRayFloat_t distance  ) {

        auto pRayInfo = std::make_unique<RayWorkInfo>(1);

#ifdef __CUDACC__

        cudaDeviceSynchronize();
        kernelCrossingDistance<<<1,1>>>(
                pGridInfo->devicePtr,
                pRayInfo.get(),
                d, pos, dir, distance );
        cudaDeviceSynchronize();

        gpuErrchk( cudaPeekAtLastError() );

#else
        kernelCrossingDistance(
                pGridInfo.get(),
                pRayInfo.get(),
                d, pos, dir, distance );
#endif

        return singleDimRayTraceMap_t( *pRayInfo, 0, d );
    }

    singleDimRayTraceMap_t crossingDistance( unsigned d, Position_t& pos, Position_t& dir, gpuRayFloat_t distance  ) {

        RayWorkInfo rayInfo(1,true);

#ifdef __CUDACC__
        auto pRayInfo = std::make_unique<RayWorkInfo>(1);

        cudaDeviceSynchronize();
        kernelCrossingDistance<<<1,1>>>(
                pGridInfo->devicePtr,
                pRayInfo.get(),
                d, pos, dir, distance );
        cudaDeviceSynchronize();

        gpuErrchk( cudaPeekAtLastError() );

#else
        kernelCrossingDistance(
                pGridInfo.get(),
                pRayInfo.get(),
                d, pos, dir, distance );
#endif
        return singleDimRayTraceMap_t( *pRayInfo, 0, d );
    }

    template<typename particle>
    rayTraceList_t rayTrace( particle& p, gpuRayFloat_t distance, bool outside = false) {

        auto pRayInfo = std::make_unique<RayWorkInfo>(1);

#ifdef __CUDACC__

        cudaDeviceSynchronize();
        kernelRayTraceParticle<<<1,1>>>( pGridInfo->devicePtr, pRayInfo.get(),
                p, distance, outside );
        cudaDeviceSynchronize();

        gpuErrchk( cudaPeekAtLastError() );

#else
        kernelRayTraceParticle( pGridInfo.get(), pRayInfo.get()
                p, distance, outside );
#endif
        rayTraceList_t rayTraceList;
        auto rayInfo = *pRayInfo;
        for( unsigned i = 0; i < rayInfo.getRayCastSize(0); ++i ) {
            rayTraceList.add( rayInfo.getRayCastCell(0,i), rayInfo.getRayCastDist(0,i) );
        }
        return rayTraceList;
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
