#include <UnitTest++.h>

#include <unistd.h>
#include <memory>
#include <cmath>

#include "MonteRayDefinitions.hh"
#include "GPUUtilityFunctions.hh"
#include "GridBins.hh"
#include "MonteRay_SpatialGrid.hh"
#include "Ray.hh"
#include "MonteRayNextEventEstimator.t.hh"
#include "MonteRayCrossSection.hh"

#include "MonteRayMaterial.hh"
#include "MonteRay_MaterialProperties.hh"
#include "MonteRayParallelAssistant.hh"
#include "GPUSync.hh"

namespace NextEventEsimator_pTester_namespace{

using namespace MonteRay;

SUITE( NextEventEstimator_pTester ) {

    class CalcScore_test {
    public:
        CalcScore_test(){

            // Two 1-cm think slabs in x direction
            grid.setVertices( 0, 0.0, 2.0, 2);
            grid.setVertices( 1, -10.0, 10.0, 1);
            grid.setVertices( 2, -10.0, 10.0, 1);
            grid.finalize();
            grid.copyToGPU();

            cell1.add( 0, 0.0); // vacuum
            matProps.add( cell1 );

            cell2.add( 0, 1.0); // density = 1.0
            matProps.add( cell2 );

            matProps.setupPtrData();

            // setup a material list
            pXS = std::unique_ptr<MonteRayCrossSectionHost> ( new MonteRayCrossSectionHost(4) );
            pXS->setParticleType( photon );
            pXS->setTotalXS(0, 1e-11, 1.0 );
            pXS->setTotalXS(1, 0.75, 1.0 );
            pXS->setTotalXS(2, 1.00, 2.0 );
            pXS->setTotalXS(3, 3.00, 4.0 );
            pXS->setAWR( gpu_AvogadroBarn / gpu_neutron_molar_mass );

            pMat = std::unique_ptr<MonteRayMaterialHost>( new MonteRayMaterialHost(1) );
            pMat->add( 0, *pXS, 1.0);
            pMat->normalizeFractions();
            pMat->copyToGPU();

            pMatList = std::unique_ptr<MonteRayMaterialListHost>( new MonteRayMaterialListHost(1,1,3) );
            pMatList->add(0, *pMat, 0);
            pMatList->copyToGPU();

            matProps.renumberMaterialIDs(*pMatList);
            matProps.copyToGPU();

            pXS->copyToGPU();

            pEstimator = std::unique_ptr<MonteRayNextEventEstimator<GridBins>>( new MonteRayNextEventEstimator<GridBins>(10) );
            pEstimator->setGeometry( &grid, &matProps );
            pEstimator->setMaterialList( pMatList.get() );
        }
        ~CalcScore_test(){}

    public:
        GridBins grid;
        MonteRay_CellProperties cell1, cell2;
        std::unique_ptr<MonteRayMaterialListHost> pMatList;
        std::unique_ptr<MonteRayMaterialHost> pMat;
        std::unique_ptr<MonteRayCrossSectionHost> pXS;
        MonteRay_MaterialProperties matProps;

        std::unique_ptr<MonteRayNextEventEstimator<GridBins>> pEstimator;
    };

    TEST_FIXTURE(CalcScore_test, getTally ) {
        const MonteRayParallelAssistant& PA( MonteRayParallelAssistant::getInstance() );
        //CHECK(false);
        gpuFloatType_t distance1 = 2.0; // score at t=0.006
        unsigned id = pEstimator->add( distance1, 0.0, 0.0);
        pEstimator->initialize();

        gpuFloatType_t x = 0.0;
        gpuFloatType_t y = 0.0;
        gpuFloatType_t z = 0.0;
        gpuFloatType_t u = 1.0;
        gpuFloatType_t v = 0.0;
        gpuFloatType_t w = 0.0;

        gpuFloatType_t energy[1];
        energy[0]= 0.5;

        gpuFloatType_t weight[1];
        weight[0] = 0.5;  // isotropic

        Ray_t<> ray;
        ray.pos[0] = x;
        ray.pos[1] = y;
        ray.pos[2] = z;
        ray.dir[0] = u;
        ray.dir[1] = v;
        ray.dir[2] = w;
        ray.energy[0] = energy[0];
        ray.weight[0] = weight[0];
        ray.detectorIndex = 0;
        ray.particleType = photon;

        unsigned particleID = 0;
        RayWorkInfo rayInfo(1,true);
        pEstimator->calcScore<1>(particleID, ray, rayInfo );

        pEstimator->gatherWorkGroup(); // used for testing only
        pEstimator->gather();

        if( PA.getWorldRank() == 0 ) {
//            std::cout << "Debug: value= " << pEstimator->getTally(0,0) << "\n";
            gpuFloatType_t expected = ( 1/ (4.0f * MonteRay::pi * distance1*distance1 ) ) * exp(-1.0);
            CHECK_CLOSE( expected*PA.getWorldSize(),  pEstimator->getTally(0,0), 1e-6);
        }

    }

    TEST_FIXTURE(CalcScore_test, addTimeBins ) {
        const MonteRayParallelAssistant& PA( MonteRayParallelAssistant::getInstance() );

        gpuFloatType_t distance1 = 2.0; // score at t=0.006
        gpuFloatType_t distance2 = 400.0; // score at t=1.33

        pEstimator->add( distance1, 0.0, 0.0);
        pEstimator->add( distance2, 0.0, 0.0);
        std::vector<MonteRay::gpuFloatType_t> timeEdges= { 1.0, 2.0, 10.0, 20.0, 100.0 };
        pEstimator->setTimeBinEdges( timeEdges );
        pEstimator->initialize();

        gpuFloatType_t x = 0.0;
        gpuFloatType_t y = 0.0;
        gpuFloatType_t z = 0.0;
        gpuFloatType_t u = 1.0;
        gpuFloatType_t v = 0.0;
        gpuFloatType_t w = 0.0;

        gpuFloatType_t energy[1];
        energy[0]= 0.5;

        gpuFloatType_t weight[1];
        weight[0] = 0.5;  // isotropic

        Ray_t<> ray;
        ray.pos[0] = x;
        ray.pos[1] = y;
        ray.pos[2] = z;
        ray.dir[0] = u;
        ray.dir[1] = v;
        ray.dir[2] = w;
        ray.energy[0] = energy[0];
        ray.weight[0] = weight[0];
        ray.time = 0.0;
        ray.detectorIndex = 0;
        ray.particleType = photon;

        unsigned particleID = 0;
        RayWorkInfo rayInfo(1,true);
        pEstimator->calcScore<1>(particleID, ray, rayInfo );

        pEstimator->gatherWorkGroup(); // used for testing only
        pEstimator->gather();

        gpuFloatType_t expected1 = PA.getWorldSize() * ( 1/ (4.0f * MonteRay::pi * distance1*distance1 ) ) * exp(-1.0);
        if( PA.getWorldRank() == 0 ) {
            CHECK_CLOSE( expected1, pEstimator->getTally(0,0), 1e-6);
            CHECK_CLOSE( 0.0,      pEstimator->getTally(0,1), 1e-6);
            CHECK_CLOSE( 0.0,      pEstimator->getTally(0,2), 1e-6);
            CHECK_CLOSE( 0.0,      pEstimator->getTally(0,3), 1e-6);
            CHECK_CLOSE( 0.0,      pEstimator->getTally(0,4), 1e-6);
            CHECK_CLOSE( 0.0,      pEstimator->getTally(0,5), 1e-6);
            CHECK_CLOSE( 0.0,      pEstimator->getTally(1,0), 1e-6);
            CHECK_CLOSE( 0.0,      pEstimator->getTally(1,1), 1e-6);
            CHECK_CLOSE( 0.0,      pEstimator->getTally(1,2), 1e-6);
            CHECK_CLOSE( 0.0,      pEstimator->getTally(1,3), 1e-6);
            CHECK_CLOSE( 0.0,      pEstimator->getTally(1,4), 1e-6);
            CHECK_CLOSE( 0.0,      pEstimator->getTally(1,5), 1e-6);
        }

        ray.detectorIndex = 1;

        rayInfo.clear();
        pEstimator->calcScore<1>(particleID, ray, rayInfo );

        pEstimator->gatherWorkGroup(); // used for testing only
        pEstimator->gather();

        gpuFloatType_t expected2 = PA.getWorldSize() *( 1/ (4.0f * MonteRay::pi * distance2*distance2 ) ) * exp(-1.0);
        if( PA.getWorldRank() == 0 ) {
            CHECK_CLOSE( expected1, pEstimator->getTally(0,0), 1e-6);
            CHECK_CLOSE( 0.0,       pEstimator->getTally(0,1), 1e-6);
            CHECK_CLOSE( 0.0,       pEstimator->getTally(0,2), 1e-6);
            CHECK_CLOSE( 0.0,       pEstimator->getTally(0,3), 1e-6);
            CHECK_CLOSE( 0.0,       pEstimator->getTally(0,4), 1e-6);
            CHECK_CLOSE( 0.0,       pEstimator->getTally(0,5), 1e-6);
            CHECK_CLOSE( 0.0,       pEstimator->getTally(1,0), 1e-6);
            CHECK_CLOSE( expected2, pEstimator->getTally(1,1), 1e-6);
            CHECK_CLOSE( 0.0,       pEstimator->getTally(1,2), 1e-6);
            CHECK_CLOSE( 0.0,       pEstimator->getTally(1,3), 1e-6);
            CHECK_CLOSE( 0.0,       pEstimator->getTally(1,4), 1e-6);
            CHECK_CLOSE( 0.0,       pEstimator->getTally(1,5), 1e-6);
        } else {
            CHECK_CLOSE( 0.0,       pEstimator->getTally(0,0), 1e-6);
            CHECK_CLOSE( 0.0,       pEstimator->getTally(0,1), 1e-6);
            CHECK_CLOSE( 0.0,       pEstimator->getTally(0,2), 1e-6);
            CHECK_CLOSE( 0.0,       pEstimator->getTally(0,3), 1e-6);
            CHECK_CLOSE( 0.0,       pEstimator->getTally(0,4), 1e-6);
            CHECK_CLOSE( 0.0,       pEstimator->getTally(0,5), 1e-6);
            CHECK_CLOSE( 0.0,       pEstimator->getTally(1,0), 1e-6);
            CHECK_CLOSE( 0.0,       pEstimator->getTally(1,1), 1e-6);
            CHECK_CLOSE( 0.0,       pEstimator->getTally(1,2), 1e-6);
            CHECK_CLOSE( 0.0,       pEstimator->getTally(1,3), 1e-6);
            CHECK_CLOSE( 0.0,       pEstimator->getTally(1,4), 1e-6);
            CHECK_CLOSE( 0.0,       pEstimator->getTally(1,5), 1e-6);
        }
    }

    TEST_FIXTURE(CalcScore_test, calc_with_rayList_on_GPU ) {
        const bool debug = false;
        const MonteRayParallelAssistant& PA( MonteRayParallelAssistant::getInstance() );

        if( debug ) {
            char hostname[1024];
            gethostname(hostname, 1024);
            std::cout << "MonteRayNextEventEstimator::calc_with_rayList_on_GPU -- hostname = " << hostname <<
                     ", world_rank=" << PA.getWorldRank() <<
                     ", world_size=" << PA.getWorldSize() <<
                     ", shared_memory_rank=" << PA.getSharedMemoryRank() <<
                     ", shared_memory_size=" << PA.getSharedMemorySize() <<
                     "\n";
        }

        const unsigned N = 3;
        gpuFloatType_t distance1 = 2.0; // score at t=0.006
        unsigned id = pEstimator->add( distance1, 0.0, 0.0);
        pEstimator->initialize();

        gpuFloatType_t x = 0.0;
        gpuFloatType_t y = 0.0;
        gpuFloatType_t z = 0.0;
        gpuFloatType_t u = 1.0;
        gpuFloatType_t v = 0.0;
        gpuFloatType_t w = 0.0;

        gpuFloatType_t energy[N];
        energy[0]= 0.5;
        energy[1]= 1.0;
        energy[2]= 3.0;

        gpuFloatType_t weight[N];
        weight[0] = 0.3;  // isotropic
        weight[1] = 1.0;
        weight[2] = 2.0;

        Ray_t<N> ray;
        ray.pos[0] = x;
        ray.pos[1] = y;
        ray.pos[2] = z;
        ray.dir[0] = u;
        ray.dir[1] = v;
        ray.dir[2] = w;

        for( unsigned i=0;i<N;++i) {
            ray.energy[i] = energy[i];
            ray.weight[i] = weight[i];
        }
        ray.index = 0;
        ray.detectorIndex = 0;
        ray.particleType = photon;

        std::unique_ptr<RayList_t<N>> pBank =  std::unique_ptr<RayList_t<N>>( new RayList_t<N>(2) );
        if( PA.getWorkGroupRank() == 0 ) {
            pBank->add( ray );
            pBank->add( ray );
        }

        if( PA.getWorkGroupRank() == 0 ) {

#ifdef __CUDACC__
            RayWorkInfo rayInfo(pBank->size());

            cudaEvent_t start, stop;
            cudaEventCreate(&start);

            pBank->copyToGPU();
            pEstimator->copyToGPU();

            rayInfo.copyToGPU();

            GPUSync();

            cudaStream_t* stream = NULL;
            stream = new cudaStream_t;
            stream[0] = 0;  // use the default stream

            cudaEventRecord(start, 0);
            cudaEventSynchronize(start);

            cudaEventCreate(&stop);

            cudaStreamSynchronize(*stream);
            pEstimator->launch_ScoreRayList(1, 1, pBank.get(), &rayInfo, stream );

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaStreamSynchronize(*stream);
            pEstimator->copyToCPU();

            delete stream;
#else
            RayWorkInfo rayInfo(1,true);
            pEstimator->launch_ScoreRayList(1,1,pBank.get(), &rayInfo );
#endif
            pEstimator->copyToCPU();
        }

        pEstimator->gather();

        gpuFloatType_t expected1 = ( 0.3f / (2.0f * MonteRay::pi * distance1*distance1 ) ) * exp( -1.0*1.0 ) +
                                   ( 1.0f / (2.0f * MonteRay::pi * distance1*distance1 ) ) * exp( -1.0*2.0 ) +
                                   ( 2.0f / (2.0f * MonteRay::pi * distance1*distance1 ) ) * exp( -1.0*4.0 );

        if( PA.getWorldRank() == 0 ) {
            // non-zero shared memory ranks do not launch a kernel call
            CHECK_CLOSE( 2 * expected1 * PA.getInterWorkGroupSize(), pEstimator->getTally(0), 1e-7);
        } else {
            CHECK_CLOSE( 0.0, pEstimator->getTally(0), 1e-7);
        }
    }

    TEST_FIXTURE(CalcScore_test, rayListOnGPU_withTimeBins ) {
        const MonteRayParallelAssistant& PA( MonteRayParallelAssistant::getInstance() );

 //        std::cout << "Debug: **********************\n";
 //        std::cout << "Debug: MonteRayNextEventEstimator_unittest.cc -- TEST calc_with_rayList_on_GPU\n";
         const unsigned N = 3;
         gpuFloatType_t distance1 = 2.0; // score at t=0.006
         gpuFloatType_t distance2 = 400.0; // score at t=1.33

         pEstimator->add( distance1, 0.0, 0.0);
         pEstimator->add( distance2, 0.0, 0.0);
         std::vector<MonteRay::gpuFloatType_t> timeEdges= { 1.0, 2.0, 10.0, 20.0, 100.0 };
         pEstimator->setTimeBinEdges( timeEdges );
         pEstimator->initialize();

         gpuFloatType_t x = 0.0;
         gpuFloatType_t y = 0.0;
         gpuFloatType_t z = 0.0;
         gpuFloatType_t u = 1.0;
         gpuFloatType_t v = 0.0;
         gpuFloatType_t w = 0.0;

         gpuFloatType_t energy[N];
         energy[0]= 0.5;
         energy[1]= 1.0;
         energy[2]= 3.0;

         gpuFloatType_t weight[N];
         weight[0] = 0.3;  // isotropic
         weight[1] = 1.0;
         weight[2] = 2.0;

         Ray_t<N> ray;
         ray.pos[0] = x;
         ray.pos[1] = y;
         ray.pos[2] = z;
         ray.dir[0] = u;
         ray.dir[1] = v;
         ray.dir[2] = w;

         for( unsigned i=0;i<N;++i) {
             ray.energy[i] = energy[i];
             ray.weight[i] = weight[i];
         }
         ray.index = 0;
         ray.detectorIndex = 0;
         ray.particleType = photon;

         std::unique_ptr<RayList_t<N>> pBank =  std::unique_ptr<RayList_t<N>>( new RayList_t<N>(4) );
         if( PA.getWorkGroupRank() == 0 ) {
             pBank->add( ray );
             pBank->add( ray );
         }

         ray.detectorIndex = 1;
         if( PA.getWorkGroupRank() == 0 ) {
             pBank->add( ray );
             pBank->add( ray );
         }

         if( PA.getWorkGroupRank() == 0 ) {

#ifdef __CUDACC__
             RayWorkInfo rayInfo(pBank->size());

             cudaEvent_t start, stop;
             cudaEventCreate(&start);

             pBank->copyToGPU();
             pEstimator->copyToGPU();

             rayInfo.copyToGPU();

             cudaStream_t* stream = NULL;
             stream = new cudaStream_t;
             stream[0] = 0;  // use the default stream

             cudaEventRecord(start, 0);
             cudaEventSynchronize(start);

             cudaEventCreate(&stop);

             cudaStreamSynchronize(*stream);
             pEstimator->launch_ScoreRayList(1, 1, pBank.get(), &rayInfo, stream );

             cudaEventRecord(stop, 0);
             cudaEventSynchronize(stop);
             cudaStreamSynchronize(*stream);
             pEstimator->copyToCPU();

             delete stream;
#else
             RayWorkInfo rayInfo(1,true);
             pEstimator->launch_ScoreRayList(1, 1, pBank.get(), &rayInfo);
#endif
         }
         pEstimator->gather();

         gpuFloatType_t expected1  = ( 0.3f / (2.0f * MonteRay::pi * distance1*distance1 ) ) * exp( -1.0*1.0 ) +
                                     ( 1.0f / (2.0f * MonteRay::pi * distance1*distance1 ) ) * exp( -1.0*2.0 ) +
                                     ( 2.0f / (2.0f * MonteRay::pi * distance1*distance1 ) ) * exp( -1.0*4.0 );

         gpuFloatType_t expected2  = ( 0.3f / (2.0f * MonteRay::pi * distance2*distance2 ) ) * exp( -1.0*1.0 ) +
                                     ( 1.0f / (2.0f * MonteRay::pi * distance2*distance2 ) ) * exp( -1.0*2.0 ) +
                                     ( 2.0f / (2.0f * MonteRay::pi * distance2*distance2 ) ) * exp( -1.0*4.0 );

         if( PA.getWorldRank() == 0 ) {
             // non-zero shared memory ranks do not launch a kernel call
             CHECK_CLOSE( 2*expected1*PA.getInterWorkGroupSize(), pEstimator->getTally(0,0), 1e-7);
             CHECK_CLOSE(                                    0.0, pEstimator->getTally(0,1), 1e-7);
             CHECK_CLOSE(                                    0.0, pEstimator->getTally(0,2), 1e-7);
             CHECK_CLOSE(                                    0.0, pEstimator->getTally(0,3), 1e-7);
             CHECK_CLOSE(                                    0.0, pEstimator->getTally(0,4), 1e-7);
             CHECK_CLOSE(                                    0.0, pEstimator->getTally(0,5), 1e-7);

             CHECK_CLOSE(                                    0.0, pEstimator->getTally(1,0), 1e-7);
             CHECK_CLOSE( 2*expected2*PA.getInterWorkGroupSize(), pEstimator->getTally(1,1), 1e-7);
             CHECK_CLOSE(                                    0.0, pEstimator->getTally(1,2), 1e-7);
             CHECK_CLOSE(                                    0.0, pEstimator->getTally(1,3), 1e-7);
             CHECK_CLOSE(                                    0.0, pEstimator->getTally(1,4), 1e-7);
             CHECK_CLOSE(                                    0.0, pEstimator->getTally(1,5), 1e-7);
         } else {
             CHECK_CLOSE(                                    0.0, pEstimator->getTally(0,0), 1e-7);
             CHECK_CLOSE(                                    0.0, pEstimator->getTally(0,1), 1e-7);
             CHECK_CLOSE(                                    0.0, pEstimator->getTally(0,2), 1e-7);
             CHECK_CLOSE(                                    0.0, pEstimator->getTally(0,3), 1e-7);
             CHECK_CLOSE(                                    0.0, pEstimator->getTally(0,4), 1e-7);
             CHECK_CLOSE(                                    0.0, pEstimator->getTally(0,5), 1e-7);

             CHECK_CLOSE(                                    0.0, pEstimator->getTally(1,0), 1e-7);
             CHECK_CLOSE(                                    0.0, pEstimator->getTally(1,1), 1e-7);
             CHECK_CLOSE(                                    0.0, pEstimator->getTally(1,2), 1e-7);
             CHECK_CLOSE(                                    0.0, pEstimator->getTally(1,3), 1e-7);
             CHECK_CLOSE(                                    0.0, pEstimator->getTally(1,4), 1e-7);
             CHECK_CLOSE(                                    0.0, pEstimator->getTally(1,5), 1e-7);
         }


 //        std::cout << "Debug: **********************\n";
     }

}

} // end namespace

