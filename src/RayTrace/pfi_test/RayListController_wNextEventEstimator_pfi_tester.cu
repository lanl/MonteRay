#include <UnitTest++.h>

#include <iostream>
#include <functional>
#include <memory>
#include <unistd.h>

#include "GPUUtilityFunctions.hh"

#include "gpuTally.hh"

#include "MonteRay_timer.hh"
#include "RayListController.hh"
#include "GridBins.hh"
#include "MonteRayMaterial.hh"
#include "MonteRayMaterialList.hh"
#include "MaterialProperties.hh"
#include "MonteRay_ReadLnk3dnt.hh"
#include "RayListInterface.hh"
#include "MonteRayConstants.hh"
#include "MonteRayNextEventEstimator.t.hh"
#include "MonteRayCrossSection.hh"
#include "MonteRayParallelAssistant.hh"

namespace RayListController_wNextEventEstimator_pfi_tester {

using namespace MonteRay;

SUITE( RayListController_wNextEventEstimator_pfi_tester_suite ) {

    class ControllerSetup {
    public:
        ControllerSetup(){
            const MonteRayParallelAssistant& PA( MonteRayParallelAssistant::getInstance() );

            //cudaReset();
            //gpuCheck();
            pGrid = new GridBins();
            pGrid->setVertices( 0, 0.0, 2.0, 2);
            pGrid->setVertices( 1, -10.0, 10.0, 1);
            pGrid->setVertices( 2, -10.0, 10.0, 1);
            pGrid->finalize();

            pMatList = new MonteRayMaterialListHost(1,1,3);

            auto matPropBuilder = MaterialProperties::Builder{};

            MaterialProperties::Builder::Cell cell1, cell2;
            cell1.add( 0, 0.0); // vacuum
            matPropBuilder.addCell(cell1);

            cell2.add( 0, 1.0); // density = 1.0
            matPropBuilder.addCell(cell2);
            
            matPropBuilder.renumberMaterialIDs(*pMatList);
            pMatProps = std::make_unique<MaterialProperties>(matPropBuilder.build());

            pXS = new MonteRayCrossSectionHost(4);

            pMat = new MonteRayMaterialHost(1);


        }

        void setup(){
            const MonteRayParallelAssistant& PA( MonteRayParallelAssistant::getInstance() );

            if( PA.getWorkGroupRank() !=0  ) return;
            pGrid->copyToGPU();

            pXS->setParticleType( photon );
            pXS->setTotalXS(0, 1e-11, 1.0 );
            pXS->setTotalXS(1, 0.75, 1.0 );
            pXS->setTotalXS(2, 1.00, 2.0 );
            pXS->setTotalXS(3, 3.00, 4.0 );
            pXS->setAWR( gpu_AvogadroBarn / gpu_neutron_molar_mass );

            pMat->add(0, *pXS, 1.0);
            pMat->copyToGPU();

            pMatList->add( 0, *pMat, 0 );
            pMatList->copyToGPU();

            pXS->copyToGPU();

        }

        ~ControllerSetup(){
            delete pGrid;
            delete pMatList;
            delete pXS;
            delete pMat;
        }

        GridBins* pGrid;
        MonteRayMaterialListHost* pMatList;
        std::unique_ptr<MaterialProperties> pMatProps;
        MonteRayCrossSectionHost* pXS;
        MonteRayMaterialHost* pMat;

    };

    TEST_FIXTURE(ControllerSetup, testOnGPU ){
        const MonteRayParallelAssistant& PA( MonteRayParallelAssistant::getInstance() );
        unsigned numPointDets = 1;
        const bool debug = true;

        if( debug ) {
            char hostname[1024];
            gethostname(hostname, 1024);
            std::cout << "RayListController_wNextEventEstimator_pfi_tester::testOnGPU -- hostname = " << hostname <<
                     ", world_rank=" << PA.getWorldRank() <<
                     ", world_size=" << PA.getWorldSize() <<
                     ", shared_memory_rank=" << PA.getSharedMemoryRank() <<
                     ", shared_memory_size=" << PA.getSharedMemorySize() <<
                     ", work_group_rank=" << PA.getWorkGroupRank() <<
                     ", work_group_size=" << PA.getWorkGroupSize() <<
                     "\n";
        }

        setup();
        NextEventEstimatorController<GridBins> controller( 1,
                1,
                pGrid,
                pMatList,
                pMatProps.get(),
                numPointDets );

        controller.setCapacity(10);

        CHECK_EQUAL( true, controller.isUsingNextEventEstimator() );

        gpuFloatType_t distance1 = 2.0;
        unsigned id = controller.addPointDet( distance1, 0.0, 0.0 );
        CHECK_EQUAL( 0, id);

        controller.copyPointDetToGPU();

        gpuFloatType_t x = 0.0;
        gpuFloatType_t y = 0.0;
        gpuFloatType_t z = 0.0;
        gpuFloatType_t u = 1.0;
        gpuFloatType_t v = 0.0;
        gpuFloatType_t w = 0.0;

        gpuFloatType_t energy[3];
        energy[0]= 0.5;
        energy[1]= 1.0;
        energy[2]= 3.0;

        gpuFloatType_t weight[3];
        weight[0] = 0.3;  // isotropic
        weight[1] = 1.0;
        weight[2] = 2.0;

        Ray_t<3> ray;
        ray.pos[0] = x;
        ray.pos[1] = y;
        ray.pos[2] = z;
        ray.dir[0] = u;
        ray.dir[1] = v;
        ray.dir[2] = w;

        for( unsigned i=0;i<3;++i) {
            ray.energy[i] = energy[i];
            ray.weight[i] = weight[i];
        }
        ray.index = 0;
        ray.detectorIndex = 0;
        ray.particleType = photon;

        controller.add( ray );
        controller.add( ray );

        if( PA.getWorkGroupRank() == 0 ) {
            CHECK_EQUAL( 2, controller.size());
            CHECK_EQUAL(10, controller.capacity());
        } else {
            CHECK_EQUAL( 0, controller.size());
            CHECK_EQUAL( 0, controller.capacity());
        }

        //        std::cout << "Debug: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n";
        //        std::cout << "Debug: RayListController_wNexteVentEstimator_fi_tester.cc:: testOnGPU - flushing \n";
        controller.sync();
        controller.flush(true);
        controller.sync();
        controller.copyPointDetTallyToCPU();
        controller.gather();

        gpuFloatType_t expected1  = ( 0.3f / (2.0f * MonteRay::pi * distance1*distance1 ) ) * exp( -1.0*1.0 ) +
                                    ( 1.0f / (2.0f * MonteRay::pi * distance1*distance1 ) ) * exp( -1.0*2.0 ) +
                                    ( 2.0f / (2.0f * MonteRay::pi * distance1*distance1 ) ) * exp( -1.0*4.0 );

        if( PA.getWorldRank() == 0 ) {
            CHECK_CLOSE( 2*(expected1)*PA.getInterWorkGroupSize(), controller.getPointDetTally(0), 1e-7);
        } else {
            CHECK_CLOSE(           0.0, controller.getPointDetTally(0), 1e-7);
        }
    }
}

//SUITE( RayListController_wNextEventEstimator_UraniumSlab ) {
//
//    class ControllerSetup {
//    public:
//        ControllerSetup(){
//
//            std::vector<double> xverts, yverts, zverts;
//
//            xverts.push_back(-10.0); xverts.push_back(0.0); xverts.push_back(1.0); xverts.push_back(11.0);
//            yverts.push_back(-10.0); yverts.push_back(10.0);
//            zverts.push_back(-10.0); zverts.push_back(10.0);
//
//            //cudaReset();
//            //gpuCheck();
//            pGrid = new GridBins();
//            pGrid->setVertices( 0, xverts );
//            pGrid->setVertices( 1, yverts );
//            pGrid->setVertices( 2, zverts );
//            pGrid->finalize();
//
//            pMatProps = new MaterialProperties;
//
//            MonteRay_CellProperties cell1, cell2;
//            cell1.add( 0, 0.0); // vacuum
//            pMatProps->add( cell1 );
//
//            cell2.add( 0, 10.0); // density = 10.0
//            pMatProps->add( cell2 );
//            pMatProps->add( cell1 );
//
//            pXS = new MonteRayCrossSectionHost(1);
//
//            pMat = new MonteRayMaterialHost(1);
//
//            pMatList = new MonteRayMaterialListHost(1,1,3);
//
//        }
//
//        void setup(){
//
//            pGrid->copyToGPU();
//            pXS->read( "MonteRayTestFiles/92000-04p_MonteRayCrossSection.bin");
//            CHECK_EQUAL( photon, pXS->getParticleType());
//
//            pMat->add(0, *pXS, 1.0);
//            pMat->copyToGPU();
//
//            pMatList->add( 0, *pMat, 0 );
//            pMatList->copyToGPU();
//
//            pMatProps->renumberMaterialIDs(*pMatList);
//            pMatProps->copyToGPU();
//
//            pXS->copyToGPU();
//
//        }
//
//        ~ControllerSetup(){
//            delete pGrid;
//            delete pMatList;
//            delete pMatProps;
//            delete pXS;
//            delete pMat;
//        }
//
//        GridBins* pGrid;
//        MonteRayMaterialListHost* pMatList;
//        MaterialProperties* pMatProps;
//        MonteRayCrossSectionHost* pXS;
//        MonteRayMaterialHost* pMat;
//    };
//
//#if true
//    TEST_FIXTURE(ControllerSetup, testOnGPU ){
//        unsigned numPointDets = 1;
//
//        setup();
//        NextEventEstimatorController<GridBins> controller( 1,
//                1,
//                pGrid,
//                pMatList,
//                pMatProps,
//                numPointDets );
//        controller.setCapacity(10);
//
//        CHECK_EQUAL( true, controller.isUsingNextEventEstimator() );
//        unsigned id = controller.addPointDet( 5.0, 0.0, 0.0 );
//        CHECK_EQUAL( 0, id);
//
//        gpuFloatType_t microXS = pXS->getTotalXS( 1.0 );
//        CHECK_CLOSE( 30.9887, microXS, 1e-4);
//
//        gpuFloatType_t AWR = pXS->getAWR();
//        CHECK_CLOSE( 235.984, AWR, 1e-3);
//        gpuFloatType_t AW = AWR * gpu_neutron_molar_mass;
//        //printf("Debug: AW=%20.10f \n", AW);
//        CHECK_CLOSE( 238.0287933350, AW , 1e-3 );
//
//        //getMicroTotalXS(ptr, E ) * density * gpu_AvogadroBarn / ptr->AtomicWeight;
//        //ptr->AtomicWeight = total * gpu_neutron_molar_mass;
//        gpuFloatType_t expectedXS = microXS * 10.0f * gpu_AvogadroBarn /  AW;
//        //printf("Debug: expectedXS=%20.12f \n", expectedXS);
//        gpuFloatType_t xs = pMatList->getTotalXS(0, 1.0, 10.0, photon);
//        gpuFloatType_t Mat_AWR = pMat->getAtomicWeight();
//        //printf("Debug: Mat_AWR=%20.10f \n", Mat_AWR);
//
//        double diffFromMCATK = std::abs(xs - 0.78365591543) * 100.0 / 0.78365591543;
//        // printf("Debug: Percent Diff=%20.10f \n", diffFromMCATK);
//        CHECK( diffFromMCATK < 0.1 ); // LinLin conversion threshold error
//
//        CHECK_CLOSE( 0.784014642239, xs, 1e-6 );
//        CHECK_CLOSE( 0.784014642239, expectedXS, 1e-6 );
//        CHECK_CLOSE( expectedXS, xs, 1e-6 );
//        CHECK_CLOSE( 238.0287933350, Mat_AWR , 1e-3 );
//        CHECK_CLOSE( AW, Mat_AWR , 1e-3 );
//
//        controller.copyPointDetToGPU();
//        controller.sync();
//        unsigned numParticles = controller.readCollisionsFromFile("MonteRayTestFiles/U-04p_slab_single_source_ray_collisionFile.bin");
//        CHECK_EQUAL( 1, numParticles );
//        controller.sync();
//        controller.copyPointDetTallyToCPU();
//
//        double expectedScore = (0.5/(2.0*MonteRay::pi*(5.001*5.001))) *  exp( -1.0*xs);
//
//        double monteRayValue = controller.getPointDetTally(0);
//        double mcatkValue = 1.4532455123e-03;
//        diffFromMCATK = std::abs(monteRayValue - mcatkValue) * 100.0 / mcatkValue;
//        //     	printf("Debug: MonteRay Score = %20.10f \n", monteRayValue);
//        //     	printf("Debug: Score Percent Diff from MCATK=%20.10f \n", diffFromMCATK);
//        CHECK( diffFromMCATK < 0.036 ); // percent difference = 3 one hundredths of a percent
//
//        CHECK_CLOSE( expectedScore, monteRayValue, 1e-7);
//        CHECK_CLOSE( 0.0014527244, monteRayValue, 1e-8 );
//        //		CHECK(false);
//    }
//#endif
//}

} // end namespace
