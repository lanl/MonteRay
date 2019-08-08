#include <UnitTest++.h>

#include <iostream>
#include <functional>
#include <cmath>
#include <memory>

#include "GPUUtilityFunctions.hh"
#include "gpuTally.hh"
#include "ExpectedPathLength.hh"
#include "MonteRay_timer.hh"
#include "MonteRay_SpatialGrid.hh"
#include "MonteRayMaterial.hh"
#include "MonteRayMaterialList.hh"
#include "MonteRay_MaterialProperties.hh"
#include "gpuTally.hh"
#include "RayListInterface.hh"
#include "RayListController.hh"
#include "MonteRay_SingleValueCopyMemory.t.hh"
#include "MonteRayCrossSection.hh"
#include "HashLookup.hh"
#include "MonteRay_GridSystemInterface.hh"
#include "RayWorkInfo.hh"

namespace RayListController_w_SpatialGrid_unit_tester{

using namespace MonteRay;

SUITE( RayListController_w_SpatialGrid_unit_tests ) {
    typedef MonteRay_SpatialGrid Grid_t;
    typedef Grid_t::Position_t Position_t;

    class UnitControllerSetup {
    public:
        UnitControllerSetup(){

            pGrid = new Grid_t;
            pGrid->setCoordinateSystem( TransportMeshTypeEnum::Cartesian );
            pGrid->setDimension( 3 );
            pGrid->setGrid( MonteRay_SpatialGrid::CART_X, -5.0, 5.0, 10);
            pGrid->setGrid( MonteRay_SpatialGrid::CART_Y, -5.0, 5.0, 10);
            pGrid->setGrid( MonteRay_SpatialGrid::CART_Z, -5.0, 5.0, 10);
            pGrid->initialize();

            pTally = new gpuTallyHost( pGrid->getNumCells() );

            pMatProps = new MonteRay_MaterialProperties();
            pMatProps->disableMemoryReduction();

            // xs from 0.0 to 100.0 mev with total cross-section of 1.0
            xs = new MonteRayCrossSectionHost(2);

            metal = new MonteRayMaterialHost(1);

            pMatList = new MonteRayMaterialListHost(1);

        }

        void setup(){

            pGrid->copyToGPU();

            pTally->copyToGPU();
            pTally->clear();

            // Density of 1.0 for mat number 0
            pMatProps->initializeMaterialDescription( std::vector<int>( pGrid->getNumCells(), 0), std::vector<float>( pGrid->getNumCells(), 1.0), pGrid->getNumCells());

            pMatProps->copyToGPU();

            xs->setTotalXS(0, 0.00001, 1.0 );
            xs->setTotalXS(1, 100.0, 1.0 );
            xs->setAWR( 1.0 );


            metal->add(0, *xs, 1.0);
            metal->copyToGPU();

            // add metal as mat number 0
            pMatList->add( 0, *metal, 0 );
            pMatList->copyToGPU();
            xs->copyToGPU();
#ifdef __CUDACC__
            gpuErrchk( cudaPeekAtLastError() );
#endif
        }

        ~UnitControllerSetup(){
            delete pGrid;
            delete pMatList;
            delete pMatProps;
            delete pTally;
            delete xs;
            delete metal;
        }

        Grid_t* pGrid;
        MonteRayMaterialListHost* pMatList;
        MonteRay_MaterialProperties* pMatProps;
        gpuTallyHost* pTally;

        MonteRayCrossSectionHost* xs;

        MonteRayMaterialHost* metal;
    };

    TEST( Reset ) {
#ifdef __CUDACC__
        cudaReset();
        gpuCheck();
        cudaDeviceSetLimit( cudaLimitStackSize, 48000 );
#endif
    }

    template<typename T>
    using resultClass = MonteRay_SingleValueCopyMemory<T>;

    CUDA_CALLABLE_KERNEL  kernelGetVertex(const Grid_t* pSpatialGrid, resultClass<gpuRayFloat_t>* pResult, unsigned d, unsigned index) {
        pResult->v = pSpatialGrid->getVertex(d,index);
    }

    template<typename GRID_T>
    gpuRayFloat_t getVertex(GRID_T* pGrid, unsigned d, unsigned index )  {
        using result_t = resultClass<gpuRayFloat_t>;
        std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );

#ifdef __CUDACC__
        pResult->copyToGPU();
        kernelGetVertex<<<1,1>>>( pGrid->getDevicePtr(), pResult->devicePtr, d, index);
        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelGetVertex( pGrid, pResult.get(), d, index);
#endif
        return pResult->v;
    }


    TEST_FIXTURE(UnitControllerSetup, getVertex ){
        setup();
#ifdef __CUDACC__
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );
#endif
        CHECK_CLOSE(-5.0, getVertex(pGrid, MonteRay_SpatialGrid::CART_X,0), 1e-11 );
    }

    class particle {
    public:
        CUDA_CALLABLE_MEMBER particle(void){};

        MonteRay_SpatialGrid::Position_t pos;
        MonteRay_SpatialGrid::Position_t dir;

        CUDA_CALLABLE_MEMBER
        MonteRay_SpatialGrid::Position_t getPosition(void) const { return pos; }

        CUDA_CALLABLE_MEMBER
        MonteRay_SpatialGrid::Position_t getDirection(void) const { return dir; }
    };

    template<typename particle>
    CUDA_CALLABLE_KERNEL  kernelGetIndexByParticle(Grid_t* pSpatialGrid, resultClass<unsigned>* pResult, particle p) {
        pResult->v = pSpatialGrid->getIndex(p);
    }

    template<typename GRID_T, typename particle>
    unsigned getIndex(GRID_T* pGridInfo, particle& p) {
        using result_t = resultClass<unsigned>;
        std::unique_ptr<result_t> pResult = std::unique_ptr<result_t> ( new result_t() );
#ifdef __CUDACC__
        pResult->copyToGPU();

        kernelGetIndexByParticle<<<1,1>>>( pGridInfo->devicePtr, pResult->devicePtr, p );
        gpuErrchk( cudaPeekAtLastError() );
        pResult->copyToCPU();
#else
        kernelGetIndexByParticle( pGridInfo, pResult.get(), p );
#endif
        return pResult->v;
    }

    TEST_FIXTURE(UnitControllerSetup, getIndex ){
        setup();
#ifdef __CUDACC__
        cudaDeviceSynchronize();
#endif

        particle p;

        MonteRay_SpatialGrid::Position_t pos1( -4.5, -4.5, -4.5 );
        MonteRay_SpatialGrid::Position_t pos2( -3.5, -4.5, -4.5 );
        MonteRay_SpatialGrid::Position_t pos3( -4.5, -3.5, -4.5 );
        MonteRay_SpatialGrid::Position_t pos4( -4.5, -4.5, -3.5 );
        MonteRay_SpatialGrid::Position_t pos5( -3.5, -3.5, -3.5 );

        p.pos = pos1;
        CHECK_EQUAL(  0, getIndex( pGrid, p ) );
        p.pos = pos2;
        CHECK_EQUAL(   1, getIndex( pGrid, p ) );
        p.pos = pos3;
        CHECK_EQUAL(  10, getIndex( pGrid, p ) );
        p.pos = pos4;
        CHECK_EQUAL( 100, getIndex( pGrid, p ) );
        p.pos = pos5;
        CHECK_EQUAL( 111, getIndex( pGrid, p ) );
    }

    CUDA_CALLABLE_KERNEL  kernelRayTrace(Grid_t* pSpatialGrid, RayWorkInfo* pRayInfo,
            gpuRayFloat_t x, gpuRayFloat_t y, gpuRayFloat_t z, gpuRayFloat_t u, gpuRayFloat_t v, gpuRayFloat_t w,
            gpuRayFloat_t distance, bool outside) {
        Position_t pos = Position_t( x,y,z);
        Position_t dir = Position_t( u,v,w);
        pSpatialGrid->rayTrace( 0, *pRayInfo, pos, dir, distance, outside);
    }

    template<typename GRID_T>
    rayTraceList_t rayTrace( GRID_T* pGridInfo, Position_t pos, Position_t dir, gpuRayFloat_t distance, bool outside=false ) {

        auto pRayInfo = std::make_unique<RayWorkInfo>(1);

#ifdef __CUDACC__
        cudaDeviceSynchronize();
        //std::cout << "Calling kernelRayTrace\n";
        kernelRayTrace<<<1,1>>>( pGridInfo->devicePtr, pRayInfo.get(),
                pos[0], pos[1], pos[2], dir[0], dir[1], dir[2], distance, outside );
        cudaDeviceSynchronize();

        gpuErrchk( cudaPeekAtLastError() );

#else
        kernelRayTrace( pGridInfo, pRayInfo.get(),
                pos[0], pos[1], pos[2], dir[0], dir[1], dir[2], distance, outside );
#endif
        rayTraceList_t rayTraceList;
        for( unsigned i = 0; i < pRayInfo->getRayCastSize(0); ++i ) {
            rayTraceList.add( pRayInfo->getRayCastCell(0,i), pRayInfo->getRayCastDist(0,i) );
        }
        return rayTraceList;
    }

    TEST_FIXTURE(UnitControllerSetup, rayTrace_outside_to_inside_posX ){
        setup();
#ifdef __CUDACC__
        cudaDeviceSynchronize();
#endif

        Grid_t::Position_t position (  -5.5, -4.5, -4.5 );
        Grid_t::Position_t direction(    1,   0,    0 );
        direction.normalize();
        gpuRayFloat_t distance = 2.0;

        rayTraceList_t distances = rayTrace( pGrid, position, direction, distance);

        CHECK_EQUAL( 2, distances.size() );
        CHECK_EQUAL( 0, distances.id(0) );
        CHECK_CLOSE( 1.0, distances.dist(0), 1e-5 );
        CHECK_EQUAL( 1, distances.id(1) );
        CHECK_CLOSE( 0.5, distances.dist(1), 1e-5 );
    }

    TEST_FIXTURE(UnitControllerSetup, single_ray ){
        //std::cout << "Debug: CollisionPointController_unit_tester -- single_ray\n";

        setup();
#ifdef __CUDACC__
        cudaDeviceSynchronize();
#endif

        CollisionPointController<Grid_t> controller( 1,
                1,
                pGrid,
                pMatList,
                pMatProps,
                pTally );


        unsigned int matID=0;
        gpuFloatType_t energy = 1.0;
        gpuFloatType_t density = 1.0;
        unsigned HashBin = getHashBin( pMatList->getHashPtr()->getPtr(), energy);
        double testXS = MonteRay::getTotalXS( pMatList->getPtr(), matID, pMatList->getHashPtr()->getPtr(), HashBin, energy, density);
        CHECK_CLOSE(.602214179f/1.00866491597f, testXS, 1e-6);

        gpuFloatType_t x = 0.5;
        gpuFloatType_t y = 0.5;
        gpuFloatType_t z = 0.5;

        unsigned i = pGrid->getIndex( Position_t(x, y, z) );
        CHECK_EQUAL( 555, i);

        ParticleRay_t particle;

        particle.pos[0] = x;
        particle.pos[1] = y;
        particle.pos[2] = z;

        particle.dir[0] = 1.0;
        particle.dir[1] = 0.0;
        particle.dir[2] = 0.0;

        particle.energy[0] = 1.0;
        particle.weight[0] = 1.0;
        particle.index = i;
        particle.detectorIndex = 1;
        particle.particleType = 0;

#ifdef __CUDACC__
        cudaDeviceSynchronize();
#endif
        controller.add(  particle );

        controller.flush(true);

        pTally->copyToCPU();

        float distance = 0.5f;
        CHECK_CLOSE( (1.0f-std::exp(-testXS*distance))/testXS, pTally->getTally(i), 1e-5 );
    }
}

}
