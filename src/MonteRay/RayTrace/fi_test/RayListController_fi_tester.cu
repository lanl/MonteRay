#include <UnitTest++.h>

#include <iostream>
#include <functional>

#include "GPUUtilityFunctions.hh"
#include "ReadAndWriteFiles.hh"

#include "BasicTally.hh"
#include "MonteRay_SpatialGrid.hh"
#include "ExpectedPathLength.hh"
#include "MonteRay_timer.hh"
#include "RayListInterface.hh"
#include "RayListController.hh"
#include "MonteRayMaterial.hh"
#include "MonteRayMaterialList.hh"
#include "MaterialProperties.hh"
#include "MonteRay_ReadLnk3dnt.hh"
#include "CrossSection.hh"

namespace RayListController_fi_tester {

using namespace MonteRay;

SUITE( Ray_bank_controller_fi_tester ) {
    using Grid_t = MonteRay_SpatialGrid;

    class ControllerSetup {
    public:
        ControllerSetup(){

            MonteRay_ReadLnk3dnt readerObject( "lnk3dnt/godivaR_lnk3dnt_cartesian_100x100x100.lnk3dnt" );
            readerObject.ReadMatData();

            pGrid = std::make_unique<Grid_t>( readerObject );

            MaterialProperties::Builder matPropBuilder{};
            matPropBuilder.disableMemoryReduction();
            matPropBuilder.setMaterialDescription( readerObject );

            CrossSectionList::Builder xsListBuilder;
            auto xsBuilder = CrossSectionBuilder();
            readInPlaceFromFile( "MonteRayTestFiles/92234-69c_MonteRayCrossSection.bin", xsBuilder );
            xsBuilder.setZAID(92234);
            xsListBuilder.add(xsBuilder.build());
            readInPlaceFromFile( "MonteRayTestFiles/92235-65c_MonteRayCrossSection.bin", xsBuilder );
            xsBuilder.setZAID(92235);
            xsListBuilder.add(xsBuilder.build());
            readInPlaceFromFile( "MonteRayTestFiles/92238-69c_MonteRayCrossSection.bin", xsBuilder );
            xsBuilder.setZAID(92238);
            xsListBuilder.add(xsBuilder.build());
            readInPlaceFromFile( "MonteRayTestFiles/1001-66c_MonteRayCrossSection.bin", xsBuilder );
            xsBuilder.setZAID(1001);
            xsListBuilder.add(xsBuilder.build());
            readInPlaceFromFile( "MonteRayTestFiles/8016-70c_MonteRayCrossSection.bin", xsBuilder );
            xsBuilder.setZAID(8016);
            xsListBuilder.add(xsBuilder.build());

            pXsList = std::make_unique<CrossSectionList>(xsListBuilder.build());

            MaterialList::Builder matListBuilder{};

            auto mb = Material::make_builder(*pXsList);

            mb.addIsotope(0.01, 92234);
            mb.addIsotope(0.98, 92235);
            mb.addIsotope(0.01, 92238);
            matListBuilder.addMaterial( 2, mb.build() );


            mb.addIsotope(2.0f/3.0f, 1001);
            mb.addIsotope(1.0f/3.0f, 8016);
            matListBuilder.addMaterial( 3, mb.build() );
            pMatList = std::make_unique<MaterialList>(matListBuilder.build());

            matPropBuilder.renumberMaterialIDs(*pMatList);
            pMatProps = std::make_unique<MaterialProperties>(matPropBuilder.build());
            pTally = std::make_unique<BasicTally>( pGrid->getNumCells() );
        }

        std::unique_ptr<Grid_t> pGrid;
        std::unique_ptr<MaterialList> pMatList;
        std::unique_ptr<MaterialProperties> pMatProps;
        std::unique_ptr<CrossSectionList> pXsList;
        std::unique_ptr<BasicTally> pTally;
    };

#if false

    TEST_FIXTURE(ControllerSetup, ctor ){
        CollisionPointController<Grid_t> controller( 1024,
                1024,
                pGrid.get(),
                pMatList.get(),
                pMatProps.get(),
                pTally.get() );

        CHECK_EQUAL(1000000, controller.capacity());
        CHECK_EQUAL(0, controller.size());
    }

    TEST_FIXTURE(ControllerSetup, setCapacity ){
        CollisionPointController<Grid_t> controller( 1024,
                1024,
                pGrid.get(),
                pMatList.get(),
                pMatProps.get(),
                pTally.get() );

        CHECK_EQUAL(1000000, controller.capacity());
        controller.setCapacity(10);
        CHECK_EQUAL(10, controller.capacity());
    }

    TEST_FIXTURE(ControllerSetup, add_a_particle ){
        CollisionPointController<Grid_t> controller( 1024,
                1024,
                pGrid.get(),
                pMatList.get(),
                pMatProps.get(),
                pTally.get() );

        unsigned i = pGrid->getIndex( 0.0, 0.0, 0.0 );
        controller.add( 0.0, 0.0, 0.0,
                1.0, 0.0, 0.0,
                1.0, 1.0, i);

        CHECK_EQUAL(1, controller.size());
    }
#endif

TEST_FIXTURE(ControllerSetup, compare_with_mcatk ){
    // exact numbers from expected path length tally in mcatk

    CollisionPointController<Grid_t> controller( 256,
            256,
            pGrid.get(),
            pMatList.get(),
            pMatProps.get(),
            pTally.get() );

    RayListInterface<1> bank1(500000);

    double x = 0.0001;
    double y = 0.0001;
    double z = 0.0001;
    double u = 1.0;
    double v = 0.0;
    double w = 0.0;
    double energy = 1.0;
    double weight = 1.0;
    unsigned index = 505050;
    unsigned detectorIndex = 101;
    short int particleType = 0;

    unsigned nI = 2;
    unsigned nJ = 1;
    for( unsigned i = 0; i < nI; ++i ) {
        for( unsigned j = 0; j < nJ; ++j ) {
            ParticleRay_t ray;
            ray.pos[0] = x;
            ray.pos[1] = y;
            ray.pos[2] = z;
            ray.dir[0] = u;
            ray.dir[1] = v;
            ray.dir[2] = w;
            ray.energy[0] = energy;
            ray.weight[0] = weight;
            ray.index = index;
            ray.detectorIndex = detectorIndex;
            ray.particleType = particleType;
            controller.add( ray );
        }
        CHECK_EQUAL( nJ, controller.size() );
        controller.flush(false);
    }
    CHECK_EQUAL( 0, controller.size() );
    controller.flush(true);

    CHECK_CLOSE( 0.601248*nI*nJ, pTally->getTally(index), 5e-6*nI*nJ );
    CHECK_CLOSE( 0.482442*nI*nJ, pTally->getTally(index+1), 5e-6*nI*nJ );

}

#if( true )
TEST_FIXTURE(ControllerSetup, launch_with_collisions_From_file ){
    std::cout << "Debug: ********************************************* \n";
    std::cout << "Debug: Starting rayTrace tester with single looping bank \n";
    CollisionPointController<Grid_t> controller( 1,
            256,
            pGrid.get(),
            pMatList.get(),
            pMatProps.get(),
            pTally.get() );
    controller.setCapacity( 1000000 );

    RayListInterface<1> bank1(50000);
    bool end = false;
    unsigned offset = 0;

    while( ! end ) {
        //    		std::cout << "Debug: reading to bank\n";
        end = bank1.readToBank( "MonteRayTestFiles/collisionsGodivaRCart100x100x100InWater_2568016Rays.bin", offset );
        offset += bank1.size();

        for( unsigned i=0; i<bank1.size(); ++i ) {
            controller.add( bank1.getParticle(i) );
        }

        if( end ) {
            controller.flush(true);
        }

    }

    // TODO - find the discrepancy
    CHECK_CLOSE( 0.0201738, pTally->getTally(24), 1e-5 );  // 0.0201584 is benchmark value - not sure why the slight difference
    CHECK_CLOSE( 0.0504394, pTally->getTally(500182), 1e-4 );

}
#endif

}

}
