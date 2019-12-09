#include <UnitTest++.h>

#include <iostream>
#include <functional>
#include <memory>
#include <array>
#include <unistd.h>

#include "GPUUtilityFunctions.hh"

#include "MonteRay_timer.hh"
#include "RayListController.hh" 
#include "MonteRayMaterial.hh"
#include "MonteRayMaterialList.hh"
#include "MaterialProperties.hh"
#include "MonteRay_ReadLnk3dnt.hh"
#include "RayListInterface.hh"
#include "MonteRayConstants.hh"
#include "MonteRayCrossSection.hh"
#include "MonteRayParallelAssistant.hh"
#include "MonteRay_SpatialGrid.hh"

namespace RayListController_wNextEventEstimator_pfi_tester {

using namespace MonteRay;

SUITE( RayListController_wNextEventEstimator_pfi_tester_suite ) {

    class ControllerSetup {
    public:
        ControllerSetup(){
            const MonteRayParallelAssistant& PA( MonteRayParallelAssistant::getInstance() );


            pGrid = std::make_unique<MonteRay_SpatialGrid>(TransportMeshType::Cartesian, 
              std::array<MonteRay_GridBins, 3>{
              MonteRay_GridBins{0.0, 2.0, 2},
              MonteRay_GridBins{-10.0, 10.0, 1},
              MonteRay_GridBins{-10.0, 10.0, 1} }
            );

            auto matPropBuilder = MaterialProperties::Builder{};

            MaterialProperties::Builder::Cell cell1, cell2;
            cell1.add( 0, 0.0); // vacuum
            matPropBuilder.addCell(cell1);

            cell2.add( 0, 1.0); // density = 1.0
            matPropBuilder.addCell(cell2);
            
            CrossSectionList::Builder xsListBuilder;
            // create ficticous xs w/ zaid 12345
            std::array<gpuFloatType_t, 4> energies = {1e-11, 0.75, 1.0, 3.0};
            std::array<gpuFloatType_t, 4> xs_values = {1.0, 1.0, 2.0, 4.0};
            constexpr gpuFloatType_t AWR = gpu_AvogadroBarn / gpu_neutron_molar_mass;
            CrossSectionBuilder xsBuilder(12345, energies, xs_values, neutron, AWR);

            xsListBuilder.add(xsBuilder.construct());
            pXsList = std::make_unique<CrossSectionList>(xsListBuilder.build());

            // using CrossSectionList as a dictionary, create materials and material list
            auto matBuilder = Material::make_builder(*pXsList);

            matBuilder.addIsotope(1.0, 12345);
            // create list builder and add material to list w/ id 0 
            MaterialList::Builder matListBuilder(0, matBuilder.build());
            pMatList = std::make_unique<MaterialList>(matListBuilder.build());

            matPropBuilder.renumberMaterialIDs(*pMatList);

            pMatProps = std::make_unique<MaterialProperties>(matPropBuilder.build());

        }

        void setup(){
            const MonteRayParallelAssistant& PA( MonteRayParallelAssistant::getInstance() );

            if( PA.getWorkGroupRank() !=0  ) return;


        }

        std::unique_ptr<MonteRay_SpatialGrid> pGrid;
        std::unique_ptr<MaterialProperties> pMatProps;
        std::unique_ptr<MaterialList> pMatList;
        std::unique_ptr<CrossSectionList> pXsList;

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
        NextEventEstimatorController<MonteRay_SpatialGrid> controller( 1,
                1,
                pGrid.get(),
                pMatList.get(),
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

} // end namespace
