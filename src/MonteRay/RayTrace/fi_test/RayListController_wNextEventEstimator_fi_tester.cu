#include <UnitTest++.h>

#include <iostream>
#include <functional>
#include <memory>

#include "GPUUtilityFunctions.hh"
#include "ReadAndWriteFiles.hh"

#include "MonteRay_timer.hh"
#include "RayListController.hh"
#include "Material.hh"
#include "MaterialList.hh"
#include "MaterialProperties.hh"
#include "MonteRay_ReadLnk3dnt.hh"
#include "RayListInterface.hh"
#include "MonteRayConstants.hh"
#include "NextEventEstimator.t.hh"
#include "CrossSection.hh"
#include "MonteRay_SpatialGrid.hh"

namespace RayListController_wNextEventEstimator_fi_tester {

using namespace MonteRay;

SUITE(RayListController_wNextEventEstimator_fi_test) {

    TEST(RayListController_WithNEE_IntegralTest_FakeXS){
        auto pGrid = std::make_unique<MonteRay_SpatialGrid>(TransportMeshType::Cartesian, 
          std::array<MonteRay_GridBins, 3>{
          MonteRay_GridBins{0.0, 2.0, 2},
          MonteRay_GridBins{-10.0, 10.0, 1},
          MonteRay_GridBins{-10.0, 10.0, 1} }
       );

        std::array<gpuFloatType_t, 4> energies = {1e-11, 0.75, 1.0, 3.0};
        std::array<gpuFloatType_t, 4> xs_values = {1.0, 1.0, 2.0, 4.0};
        constexpr gpuFloatType_t AWR = gpu_AvogadroBarn / gpu_neutron_molar_mass;
        CrossSectionBuilder xsBuilder(12345, energies, xs_values, photon, AWR);

        CrossSectionList::Builder xsListBuilder;
        xsListBuilder.add(xsBuilder.build());
        auto pXsList = std::make_unique<CrossSectionList>(xsListBuilder.build());

        auto matBuilder = Material::make_builder(*pXsList);
        matBuilder.addIsotope(1.0, 12345);
        MaterialList::Builder matListBuilder(0, matBuilder.build());
        auto pMatList = std::make_unique<MaterialList>(matListBuilder.build());

        // reconstruct matPropsBuilder since the initial constructor didn't renumber material IDs.
        auto matPropsBuilder = MaterialProperties::Builder{};
        MaterialProperties::Builder::Cell cell1, cell2;
        cell1.add(0, 0.0); // vacuum
        matPropsBuilder.addCell(cell1);

        cell2.add(0, 1.0); // density = 1.0
        matPropsBuilder.addCell(cell2);
        matPropsBuilder.renumberMaterialIDs(*pMatList);
        auto pMatProps = std::make_unique<MaterialProperties>(matPropsBuilder.build());

        NextEventEstimator::Builder neeBuilder;
        neeBuilder.addTallyPoint(2.0, 0.0, 0.0);

        NextEventEstimatorController::Builder controllerBuilder;
        controllerBuilder.nBlocks(1)
                         .nThreads(256)
                         .geometry(pGrid.get())
                         .materialList(pMatList.get())
                         .materialProperties(pMatProps.get())
                         .capacity(10)
                         .nextEventEstimator(neeBuilder.build());

        auto controller = controllerBuilder.build();

        CHECK_EQUAL(true, controller.isUsingNextEventEstimator());

        Ray_t<3> ray;
        ray.energy[0]= 0.5;
        ray.energy[1]= 1.0;
        ray.energy[2]= 3.0;

        ray.weight[0] = 0.3;  // isotropic
        ray.weight[1] = 1.0;
        ray.weight[2] = 2.0;

        ray.pos[0] = 0.0;
        ray.pos[1] = 0.0;
        ray.pos[2] = 0.0;
        ray.dir[0] = 1.0;
        ray.dir[1] = 0.0;
        ray.dir[2] = 0.0;

        ray.index = 0;
        ray.detectorIndex = 0;
        ray.particleType = photon;

        controller.add(ray);
        controller.add(ray);
        CHECK_EQUAL(2, controller.size());
        CHECK_EQUAL(10, controller.capacity());

        controller.sync();
        controller.flush(true);
        controller.sync();
        controller.copyPointDetTallyToCPU();

        gpuFloatType_t expected1 = (0.3f / (2.0f * MonteRay::pi * 4.0f)) * exp(-1.0*1.0);
        gpuFloatType_t expected2 = (1.0f / (2.0f * MonteRay::pi * 4.0f)) * exp(-1.0*2.0);
        gpuFloatType_t expected3 = (2.0f / (2.0f * MonteRay::pi * 4.0f)) * exp(-1.0*4.0);

        CHECK_CLOSE(2*(expected1+expected2+expected3), controller.contribution(0), 1e-7);
    }
}

SUITE(RayListController_wNextEventEstimator_UraniumSlab) {

#if false // TODO: update matPropsBuilder.readFromFile
    TEST(RayListController_withNEE_IntegralTest_UraniumSlab){
        std::vector<double> xverts{-10, 0, 1, 11};
        std::vector<double> yverts{-10, 10};
        std::vector<double> zverts{-10, 10};

        auto pGrid = std::make_unique<MonteRay_SpatialGrid>(TransportMeshType::Cartesian, 
          std::array<MonteRay_GridBins, 3>{
          MonteRay_GridBins{xverts},
          MonteRay_GridBins{yverts},
          MonteRay_GridBins{zverts} }
       );

        CrossSectionList::Builder xsListBuilder;
        auto pXS = std::make_unique<CrossSection>(
          readFromFile<CrossSection>(std::string("MonteRayTestFiles/92000-04p_MonteRayCrossSection.bin"))
       );

        CHECK_EQUAL(photon, pXS->getParticleType());

        CrossSectionList::Builder xsListBuilder;
        xsListBuilder.add(xsBuilder.build());
        auto pXsList = std::make_unique<CrossSectionList>(xsListBuilder.build());

        MaterialList::Builder matListBuilder{};

        auto mb = Material::make_builder(*pXsList);
        mb.addIsotope(1.0, 0);
        matListBuilder.addMaterial(0, mb.build());

        auto pMatList = std::make_unique<MaterialList>(matListBuilder.build());

        auto matPropsBuilder = MaterialProperties::Builder{};
        MaterialProperties::Builder::Cell cell1, cell2;
        cell1.add(0, 0.0); // vacuum
        matPropsBuilder.addCell(cell1);

        cell2.add(0, 10.0); // density = 10.0
        matPropsBuilder.addCell(cell2);
        matPropsBuilder.addCell(cell1);
        matPropsBuilder.renumberMaterialIDs(*pMatList);
        auto pMatProps = std::make_unique<MaterialProperties>(matPropsBuilder.build());
        
        NextEventEstimatorController<MonteRay_SpatialGrid> controller(1,
                1,
                pGrid.get(),
                pMatList,
                pMatProps.get(),
                numPointDets);
        controller.setCapacity(10);

        CHECK_EQUAL(true, controller.isUsingNextEventEstimator());
        unsigned id = controller.addPointDet(5.0, 0.0, 0.0);
        CHECK_EQUAL(0, id);

        gpuFloatType_t microXS = pXS->getTotalXS(1.0);
        CHECK_CLOSE(30.9887, microXS, 1e-4);

        gpuFloatType_t AWR = pXS->getAWR();
        CHECK_CLOSE(235.984, AWR, 1e-3);
        gpuFloatType_t AW = AWR * gpu_neutron_molar_mass;
        //printf("Debug: AW=%20.10f \n", AW);
        CHECK_CLOSE(238.0287933350, AW , 1e-3);

        //getMicroTotalXS(ptr, E) * density * gpu_AvogadroBarn / ptr->AtomicWeight;
        //ptr->AtomicWeight = total * gpu_neutron_molar_mass;
        gpuFloatType_t expectedXS = microXS * 10.0f * gpu_AvogadroBarn /  AW;
        //printf("Debug: expectedXS=%20.12f \n", expectedXS);
        gpuFloatType_t xs = pMatList->getTotalXS(0, 1.0, 10.0, photon);
        gpuFloatType_t Mat_AWR = pMat->getAtomicWeight();
        //printf("Debug: Mat_AWR=%20.10f \n", Mat_AWR);

        double diffFromMCATK = std::abs(xs - 0.78365591543) * 100.0 / 0.78365591543;
        // printf("Debug: Percent Diff=%20.10f \n", diffFromMCATK);
        CHECK(diffFromMCATK < 0.1); // LinLin conversion threshold error

        CHECK_CLOSE(0.784014642239, xs, 1e-6);
        CHECK_CLOSE(0.784014642239, expectedXS, 1e-6);
        CHECK_CLOSE(expectedXS, xs, 1e-6);
        CHECK_CLOSE(238.0287933350, Mat_AWR , 1e-3);
        CHECK_CLOSE(AW, Mat_AWR , 1e-3);

        controller.copyPointDetToGPU();

        controller.sync();
        unsigned numParticles = controller.readCollisionsFromFileToBuffer("MonteRayTestFiles/U-04p_slab_single_source_ray_collisionFile.bin");

        CHECK_EQUAL(1, numParticles);

        controller.flush(true);
        controller.sync();
        controller.copyPointDetTallyToCPU();

        double expectedScore = (0.5/(2.0*MonteRay::pi*(5.001*5.001)))*exp(-1.0*xs);

        double monteRayValue = controller.contribution(0);
        double mcatkValue = 1.4532455123e-03;
        diffFromMCATK = std::abs(monteRayValue - mcatkValue) * 100.0 / mcatkValue;
        CHECK(diffFromMCATK < 0.036); // percent difference = 3 one hundredths of a percent

        CHECK_CLOSE(expectedScore, monteRayValue, 1e-7);
        CHECK_CLOSE(0.0014527244, monteRayValue, 1e-8);
    }


#endif
}

} // end namespace
