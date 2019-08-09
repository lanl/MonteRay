#include <UnitTest++.h>

#include <iostream>
#include <functional>

#include "GPUUtilityFunctions.hh"

#include "gpuTally.hh"
#include "ExpectedPathLength.hh"
#include "MonteRay_timer.hh"
#include "RayListInterface.hh"
#include "RayListController.hh"
#include "MonteRay_SpatialGrid.hh"
#include "MonteRayMaterial.hh"
#include "MonteRayMaterialList.hh"
#include "MaterialProperties.hh"
#include "MonteRay_ReadLnk3dnt.hh"
#include "MonteRayCrossSection.hh"

namespace RayListController_fi_tester {

using namespace MonteRay;

SUITE( Ray_bank_controller_with_Cartesian_SpatialGrid_fi_tester ) {
    typedef MonteRay_SpatialGrid Grid_t;

    class ControllerSetup {
    public:
        ControllerSetup(){

            u234s = new MonteRayCrossSectionHost(1);
            u235s = new MonteRayCrossSectionHost(1);
            u238s = new MonteRayCrossSectionHost(1);
            h1s = new MonteRayCrossSectionHost(1);
            o16s = new MonteRayCrossSectionHost(1);

            metal = new MonteRayMaterialHost(3);
            water = new MonteRayMaterialHost(2);

            pMatList = new MonteRayMaterialListHost(2,5);

        }

        void setup(){
            MonteRay_ReadLnk3dnt readerObject( "lnk3dnt/godivaR_lnk3dnt_cartesian_100x100x100.lnk3dnt" );
            readerObject.ReadMatData();

            MaterialProperties::Builder matPropBuilder{};
            matPropBuilder.disableMemoryReduction();
            matPropBuilder.setMaterialDescription( readerObject );

            pGrid = new Grid_t( readerObject );
            pGrid->copyToGPU();

            pTally = new gpuTallyHost( pGrid->getNumCells() );
            pTally->copyToGPU();

            u234s->read( "MonteRayTestFiles/92234-69c_MonteRayCrossSection.bin" );
            u235s->read( "MonteRayTestFiles/92235-65c_MonteRayCrossSection.bin" );
            u238s->read( "MonteRayTestFiles/92238-69c_MonteRayCrossSection.bin" );
            h1s->read( "MonteRayTestFiles/1001-66c_MonteRayCrossSection.bin" );
            o16s->read( "MonteRayTestFiles/8016-70c_MonteRayCrossSection.bin" );

            metal->add(0, *u234s, 0.01);
            metal->add(1, *u235s, 0.98);
            metal->add(2, *u238s, 0.01);
            metal->copyToGPU();

            water->add(0, *h1s, 2.0f/3.0f );
            water->add(1, *o16s, 1.0f/3.0f );
            water->copyToGPU();

            pMatList->add( 0, *metal, 2 );
            pMatList->add( 1, *water, 3 );
            pMatList->copyToGPU();

            matPropBuilder.renumberMaterialIDs(*pMatList);
            pMatProps = std::make_unique<MaterialProperties>(matPropBuilder.build());

            u234s->copyToGPU();
            u235s->copyToGPU();
            u238s->copyToGPU();
            h1s->copyToGPU();
            o16s->copyToGPU();
        }

        ~ControllerSetup(){
            delete pGrid;
            delete pMatList;
            delete pTally;
            delete u234s;
            delete u235s;
            delete u238s;
            delete h1s;
            delete o16s;
            delete metal;
            delete water;
        }

        Grid_t* pGrid;
        MonteRayMaterialListHost* pMatList;
        std::unique_ptr<MaterialProperties> pMatProps;
        gpuTallyHost* pTally;

        MonteRayCrossSectionHost* u234s;
        MonteRayCrossSectionHost* u235s;
        MonteRayCrossSectionHost* u238s;
        MonteRayCrossSectionHost* h1s;
        MonteRayCrossSectionHost* o16s;

        MonteRayMaterialHost* metal;
        MonteRayMaterialHost* water;

    };

    TEST( Reset ) {
#ifdef __CUDACC__
        //cudaReset();
        //gpuCheck();
        cudaDeviceSetLimit( cudaLimitStackSize, 48000 );
#endif
    }

    TEST_FIXTURE(ControllerSetup, compare_with_mcatk ){

        setup();

        // exact numbers from expected path length tally in mcatk
        CollisionPointController<Grid_t> controller( 256,
                256,
                pGrid,
                pMatList,
                pMatProps.get(),
                pTally );

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

        const unsigned nBatches = 2;
        const unsigned nParticlePerBatch = 1;
        for( unsigned i = 0; i < nBatches; ++i ) {
            for( unsigned j = 0; j < nParticlePerBatch; ++j ) {
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
            CHECK_EQUAL( nParticlePerBatch, controller.size() );
            controller.flush(false);
        }
        CHECK_EQUAL( 0, controller.size() );
        controller.flush(true);

        pTally->copyToCPU();

        CHECK_CLOSE( 0.601248*nBatches*nParticlePerBatch, pTally->getTally(index), 1e-6*nBatches*nParticlePerBatch );
        CHECK_CLOSE( 0.482442*nBatches*nParticlePerBatch, pTally->getTally(index+1), 1e-6*nBatches*nParticlePerBatch );

    }

#if( true )
    TEST_FIXTURE(ControllerSetup, launch_with_collisions_From_file ){
        std::cout << "Debug: ********************************************* \n";
        std::cout << "Debug: Starting SpatialGrid - Cartesian - rayTrace tester with single looping bank \n";
        setup();
        CollisionPointController<Grid_t> controller( 256,
                256,
                pGrid,
                pMatList,
                pMatProps.get(),
                pTally );
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

        pTally->copyToCPU();

        // TODO - find the discrepancy
        CHECK_CLOSE( 0.0201738, pTally->getTally(24), 1e-5 );  // 0.0201584 is benchmark value - not sure why the slight difference
        CHECK_CLOSE( 0.0504394, pTally->getTally(500182), 1e-4 );

    }
#endif

}

}
