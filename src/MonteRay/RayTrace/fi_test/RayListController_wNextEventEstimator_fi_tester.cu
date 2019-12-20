#include <UnitTest++.h>

#include <iostream>
#include <functional>
#include <memory>

#include "GPUUtilityFunctions.hh"
#include "ReadAndWriteFiles.hh"

#include "BasicTally.hh"
#include "MonteRay_timer.hh"
#include "RayListController.hh"
#include "Material.hh"
#include "MaterialList.hh"
#include "MaterialProperties.hh"
#include "MonteRay_ReadLnk3dnt.hh"
#include "RayListInterface.hh"
#include "MonteRayConstants.hh"
#include "NextEventEstimator.t.hh"
#include "MonteRayCrossSection.hh"
#include "MonteRay_SpatialGrid.hh"

namespace RayListController_wNextEventEstimator_fi_tester {

using namespace MonteRay;

SUITE( RayListController_wNextEventEstimator_fi_tester_suite ) {

    class ControllerSetup {
    public:
        ControllerSetup(){

            
            pGrid = std::make_unique<MonteRay_SpatialGrid>(TransportMeshType::Cartesian, 
              std::array<MonteRay_GridBins, 3>{
              MonteRay_GridBins{0.0, 2.0, 2},
              MonteRay_GridBins{-10.0, 10.0, 1},
              MonteRay_GridBins{-10.0, 10.0, 1} }
            );

            auto matPropsBuilder = MaterialProperties::Builder{};

            MaterialProperties::Builder::Cell cell1, cell2;
            cell1.add( 0, 0.0); // vacuum
            matPropsBuilder.addCell(cell1);

            cell2.add( 0, 1.0); // density = 1.0
            matPropsBuilder.addCell(cell2);
            pMatProps = std::make_unique<MaterialProperties>(matPropsBuilder.build());
        }

        void setup(){

            std::array<gpuFloatType_t, 4> energies = {1e-11, 0.75, 1.0, 3.0};
            std::array<gpuFloatType_t, 4> xs_values = {1.0, 1.0, 2.0, 4.0};
            constexpr gpuFloatType_t AWR = gpu_AvogadroBarn / gpu_neutron_molar_mass;
            CrossSectionBuilder xsBuilder(12345, energies, xs_values, photon, AWR);

            CrossSectionList::Builder xsListBuilder;
            xsListBuilder.add(xsBuilder.construct());
            pXsList = std::make_unique<CrossSectionList>(xsListBuilder.build());

            auto matBuilder = Material::make_builder(*pXsList);
            matBuilder.addIsotope(1.0, 12345);
            MaterialList::Builder matListBuilder(0, matBuilder.build());
            pMatList = std::make_unique<MaterialList>(matListBuilder.build());

            // reconstruct matPropsBuilder since the initial constructor didn't renumber material IDs.
            auto matPropsBuilder = MaterialProperties::Builder{};
            MaterialProperties::Builder::Cell cell1, cell2;
            cell1.add( 0, 0.0); // vacuum
            matPropsBuilder.addCell(cell1);

            cell2.add( 0, 1.0); // density = 1.0
            matPropsBuilder.addCell(cell2);
            matPropsBuilder.renumberMaterialIDs(*pMatList);
        }

        std::unique_ptr<MonteRay_SpatialGrid> pGrid;
        std::unique_ptr<MaterialProperties> pMatProps;
        std::unique_ptr<MaterialList> pMatList;
        std::unique_ptr<CrossSectionList> pXsList;
    };

#if true
    TEST( setup ) {
        //gpuCheck();
    }

    TEST_FIXTURE(ControllerSetup, ctorForNEE ){
        unsigned numPointDets = 1;
        NextEventEstimatorController<MonteRay_SpatialGrid> controller( 1,
                1,
                pGrid.get(),
                pMatList.get(),
                pMatProps.get(),
                numPointDets );

        CHECK_EQUAL( true, controller.isUsingNextEventEstimator() );
        CHECK_EQUAL(100000, controller.capacity());
        CHECK_EQUAL(0, controller.size());
    }
#endif

    TEST_FIXTURE(ControllerSetup, testOnGPU ){
        unsigned numPointDets = 1;

        setup();
        NextEventEstimatorController<MonteRay_SpatialGrid> controller( 1,
                1,
                pGrid.get(),
                pMatList.get(),
                pMatProps.get(),
                numPointDets );

        controller.setCapacity(10);

        CHECK_EQUAL( true, controller.isUsingNextEventEstimator() );
        unsigned id = controller.addPointDet( 2.0, 0.0, 0.0 );
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
        CHECK_EQUAL(2, controller.size());
        CHECK_EQUAL(10, controller.capacity());

        //        std::cout << "Debug: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n";
        //        std::cout << "Debug: RayListController_wNexteVentEstimator_fi_tester.cc:: testOnGPU - flushing \n";
        controller.sync();
        controller.flush(true);
        controller.sync();
        controller.copyPointDetTallyToCPU();
        //        std::cout << "Debug: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n";

        gpuFloatType_t expected1 = ( 0.3f / (2.0f * MonteRay::pi * 4.0f ) ) * exp( -1.0*1.0 );
        gpuFloatType_t expected2 = ( 1.0f / (2.0f * MonteRay::pi * 4.0f ) ) * exp( -1.0*2.0 );
        gpuFloatType_t expected3 = ( 2.0f / (2.0f * MonteRay::pi * 4.0f ) ) * exp( -1.0*4.0 );

        CHECK_CLOSE( 2*(expected1+expected2+expected3), controller.getPointDetTally(0), 1e-7);
    }
}

SUITE( RayListController_wNextEventEstimator_UraniumSlab ) {

    class ControllerSetup {
    public:
        ControllerSetup(){

            std::vector<double> xverts, yverts, zverts;

            xverts.push_back(-10.0); xverts.push_back(0.0); xverts.push_back(1.0); xverts.push_back(11.0);
            yverts.push_back(-10.0); yverts.push_back(10.0);
            zverts.push_back(-10.0); zverts.push_back(10.0);

            pGrid = std::make_unique<MonteRay_SpatialGrid>(TransportMeshType::Cartesian, 
              std::array<MonteRay_GridBins, 3>{
              MonteRay_GridBins{xverts},
              MonteRay_GridBins{yverts},
              MonteRay_GridBins{zverts} }
            );

            auto matPropsBuilder = MaterialProperties::Builder{};
            MaterialProperties::Builder::Cell cell1, cell2;
            cell1.add( 0, 0.0); // vacuum
            matPropsBuilder.addCell(cell1);

            cell2.add( 0, 10.0); // density = 10.0
            matPropsBuilder.addCell(cell2);
            matPropsBuilder.addCell(cell1);
            pMatProps = std::make_unique<MaterialProperties>(matPropsBuilder.build());

        }

        void setup(){

            CrossSectionList::Builder xsListBuilder;
            auto pXS = std::make_unique<CrossSection>(
              readFromFile<CrossSection>( std::string("MonteRayTestFiles/92000-04p_MonteRayCrossSection.bin") )
            );

            CHECK_EQUAL( photon, pXS->getParticleType());

            MaterialList::Builder matListBuilder{};

            auto mb = Material::make_builder(*pXsList);
            mb.addIsotope(1.0, 0);
            matListBuilder.addMaterial( 0, mb.build() );

            pMatList = std::make_unique<MaterialList>(matListBuilder.build());
            // reconstruct matPropsBuilder since the initial constructor didn't renumber material IDs.
            auto matPropsBuilder = MaterialProperties::Builder{};
            MaterialProperties::Builder::Cell cell1, cell2;
            cell1.add( 0, 0.0); // vacuum
            matPropsBuilder.addCell(cell1);

            cell2.add( 0, 10.0); // density = 10.0
            matPropsBuilder.addCell(cell2);
            matPropsBuilder.addCell(cell1);
            matPropsBuilder.renumberMaterialIDs(*pMatList);
            pMatProps = std::make_unique<MaterialProperties>(matPropsBuilder.build());
        }

        std::unique_ptr<MonteRay_SpatialGrid> pGrid;
        std::unique_ptr<MaterialProperties> pMatProps;
        std::unique_ptr<MaterialList> pMatList;
        std::unique_ptr<CrossSectionList> pXsList;
    };

#if false // TODO: update matPropsBuilder.readFromFile
    TEST_FIXTURE(ControllerSetup, testOnGPU ){
        unsigned numPointDets = 1;

        setup();
        NextEventEstimatorController<MonteRay_SpatialGrid> controller( 1,
                1,
                pGrid.get(),
                pMatList,
                pMatProps.get(),
                numPointDets );
        controller.setCapacity(10);

        CHECK_EQUAL( true, controller.isUsingNextEventEstimator() );
        unsigned id = controller.addPointDet( 5.0, 0.0, 0.0 );
        CHECK_EQUAL( 0, id);

        gpuFloatType_t microXS = pXS->getTotalXS( 1.0 );
        CHECK_CLOSE( 30.9887, microXS, 1e-4);

        gpuFloatType_t AWR = pXS->getAWR();
        CHECK_CLOSE( 235.984, AWR, 1e-3);
        gpuFloatType_t AW = AWR * gpu_neutron_molar_mass;
        //printf("Debug: AW=%20.10f \n", AW);
        CHECK_CLOSE( 238.0287933350, AW , 1e-3 );

        //getMicroTotalXS(ptr, E ) * density * gpu_AvogadroBarn / ptr->AtomicWeight;
        //ptr->AtomicWeight = total * gpu_neutron_molar_mass;
        gpuFloatType_t expectedXS = microXS * 10.0f * gpu_AvogadroBarn /  AW;
        //printf("Debug: expectedXS=%20.12f \n", expectedXS);
        gpuFloatType_t xs = pMatList->getTotalXS(0, 1.0, 10.0, photon);
        gpuFloatType_t Mat_AWR = pMat->getAtomicWeight();
        //printf("Debug: Mat_AWR=%20.10f \n", Mat_AWR);

        double diffFromMCATK = std::abs(xs - 0.78365591543) * 100.0 / 0.78365591543;
        // printf("Debug: Percent Diff=%20.10f \n", diffFromMCATK);
        CHECK( diffFromMCATK < 0.1 ); // LinLin conversion threshold error

        CHECK_CLOSE( 0.784014642239, xs, 1e-6 );
        CHECK_CLOSE( 0.784014642239, expectedXS, 1e-6 );
        CHECK_CLOSE( expectedXS, xs, 1e-6 );
        CHECK_CLOSE( 238.0287933350, Mat_AWR , 1e-3 );
        CHECK_CLOSE( AW, Mat_AWR , 1e-3 );

        controller.copyPointDetToGPU();

        controller.sync();
        unsigned numParticles = controller.readCollisionsFromFileToBuffer("MonteRayTestFiles/U-04p_slab_single_source_ray_collisionFile.bin");
        /* controller.dumpPointDetForDebug( "nee_debug_dump_test1.bin"); */

        CHECK_EQUAL( 1, numParticles );

        controller.flush(true);
        controller.sync();
        controller.copyPointDetTallyToCPU();

        double expectedScore = (0.5/(2.0*MonteRay::pi*(5.001*5.001))) *  exp( -1.0*xs);

        double monteRayValue = controller.getPointDetTally(0);
        double mcatkValue = 1.4532455123e-03;
        diffFromMCATK = std::abs(monteRayValue - mcatkValue) * 100.0 / mcatkValue;
        //     	printf("Debug: MonteRay Score = %20.10f \n", monteRayValue);
        //     	printf("Debug: Score Percent Diff from MCATK=%20.10f \n", diffFromMCATK);
        CHECK( diffFromMCATK < 0.036 ); // percent difference = 3 one hundredths of a percent

        CHECK_CLOSE( expectedScore, monteRayValue, 1e-7);
        CHECK_CLOSE( 0.0014527244, monteRayValue, 1e-8 );
        //		CHECK(false);
    }

    /* bool exists( const std::string& filename ) { */
    /*     bool good = false; */
    /*     std::ifstream file(filename.c_str()); */
    /*     good = file.good(); */
    /*     file.close(); */
    /*     return good; */
    /* } */

    /* TEST( read_nee_debug_dump_test1 ) { */
    /*     // next-event estimator */
    /*     // test nee save state file exists */
    /*     std::string filename = "nee_debug_dump_test1.bin"; */
    /*     CHECK_EQUAL( true, exists(filename) ); */

    /*     auto estimator = readFromFile( filename, NextEventEstimator::Builder{} ); */

    /*     // raylist */
    /*     filename = std::string("raylist_") + baseName; */
    /*     CHECK_EQUAL( true, exists(filename) ); */
    /*     RayList_t<3> raylist(1); */
    /*     raylist.readFromFile( filename ); */

    /*     CHECK_EQUAL( 1, raylist.size() ); */
    /*     CHECK_CLOSE( 1.0, raylist.getEnergy(0), 1e-6); */

    /*     // geometry */
    /*     filename = std::string("geometry_") + baseName; */
    /*     CHECK_EQUAL( true, exists(filename) ); */
    /*     GridBins grid; */
    /*     grid.readFromFile( filename ); */

    /*     // material properties */
    /*     MaterialProperties::Builder matPropsBuilder; */
    /*     filename = std::string("matProps_") + baseName; */
    /*     CHECK_EQUAL( true, exists(filename) ); */
    /*     matPropsBuilder.readFromFile( filename ); */
    /*     auto matProps = matPropsBuilder.build(); */

    /*     // materials */
    /*     MaterialListHost matlist(1); */
    /*     filename = std::string("materialList_") + baseName; */
    /*     CHECK_EQUAL( true, exists(filename) ); */
    /*     matlist.readFromFile( filename ); */

    /*     grid.copyToGPU(); */
    /*     matlist.copyToGPU(); */

    /*     /1* estimator.dumpState( &raylist, "nee_debug_dump_test2" ); *1/ */

    /*     raylist.copyToGPU(); */

/* #ifdef __CUDACC__ */
    /*     cudaStreamSynchronize(0); */
/* #endif */

    /*     auto pRayInfo = std::make_unique<RayWorkInfo>(raylist.size()); */
    /*     estimator.launch_ScoreRayList(1U,1U, &raylist, pRayInfo.get()); */

/* #ifdef __CUDACC__ */
    /*     cudaStreamSynchronize(0); */
/* #endif */

    /*     estimator.copyToCPU(); */
    /*     double monteRayValue = estimator.getTally(0); */
    /*     CHECK_CLOSE( 0.0014527244, monteRayValue, 1e-8 ); */
    /* } */

#endif
}

} // end namespace
