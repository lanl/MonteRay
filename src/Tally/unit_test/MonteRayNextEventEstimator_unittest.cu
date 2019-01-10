#include <UnitTest++.h>

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
// #define MEMCHECK 1

namespace nextEventEsimator_unittest{

using namespace MonteRay;

SUITE( NextEventEstimator_Tester ) {

#ifndef MEMCHECK
    TEST(  make_a_PointDetRay ) {
        PointDetRay_t ray;
        CHECK_EQUAL(3, ray.getN() );
        CHECK(true);
    }

    TEST(  MonteRayNextEventEstimator_ctor ) {
        MonteRayNextEventEstimator<GridBins> estimator(1);
        CHECK_EQUAL(0, estimator.size() );
        CHECK_EQUAL(1, estimator.capacity() );
        CHECK_CLOSE( 0.0, estimator.getExclusionRadius(), 1e-6 );
    }

    TEST(  MonteRayNextEventEstimator_get_invalid_X ) {
        MonteRayNextEventEstimator<GridBins> estimator(1);
#ifdef DEBUG
        CHECK_THROW( estimator.getX(10), std::runtime_error );
        CHECK_THROW( estimator.getY(10), std::runtime_error );
        CHECK_THROW( estimator.getZ(10), std::runtime_error );
#endif
    }

    TEST( add ) {
        MonteRayNextEventEstimator<GridBins> estimator(1);
        unsigned id = estimator.add( 1.0, 2.0, 3.0);
        CHECK_EQUAL( 0, id);
        CHECK_CLOSE( 1.0, estimator.getX(0), 1e-6 );
        CHECK_CLOSE( 2.0, estimator.getY(0), 1e-6 );
        CHECK_CLOSE( 3.0, estimator.getZ(0), 1e-6 );
    }

    TEST( add_too_many ) {
        MonteRayNextEventEstimator<GridBins> estimator(1);
        unsigned id = estimator.add( 1.0, 2.0, 3.0);
        CHECK_THROW( estimator.add( 1.0, 2.0, 3.0), std::runtime_error );
    }

    TEST( set_exclusion_radius ) {
        MonteRayNextEventEstimator<GridBins> estimator(1);
        estimator.setExclusionRadius( 1.9 );
        CHECK_CLOSE( 1.9, estimator.getExclusionRadius(), 1e-6 );
    }

    TEST( getDistance ) {
        MonteRayNextEventEstimator<GridBins> estimator(1);
        unsigned id = estimator.add( 3.0, 3.0, 3.0);
        gpuFloatType_t expectedDistance = std::sqrt( (3.0f*3.0f)*3 );

        MonteRay::Vector3D<gpuRayFloat_t> pos(0.0, 0.0, 0.0);

        gpuFloatType_t distance = estimator.distance( 0, pos );

        CHECK_CLOSE( expectedDistance, distance, 1e-6 );
    }

    TEST( getDistanceDirection_PosU ) {
        MonteRayNextEventEstimator<GridBins> estimator(1);
        unsigned id = estimator.add( 3.0, 0.0, 0.0);
        gpuFloatType_t expectedDistance = std::sqrt( 3.0f*3.0f );

        MonteRay::Vector3D<gpuRayFloat_t> pos(0.0, 0.0, 0.0);
        MonteRay::Vector3D<gpuRayFloat_t> dir;

        gpuFloatType_t distance = estimator.getDistanceDirection( 0, pos, dir );

        CHECK_CLOSE( expectedDistance, distance, 1e-6 );
        CHECK_CLOSE( 1.0, dir[0], 1e-6 );
    }

    TEST( getDistanceDirection_NegU ) {
        MonteRayNextEventEstimator<GridBins> estimator(1);
        unsigned id = estimator.add( -3.0, 0.0, 0.0);
        gpuFloatType_t expectedDistance = std::sqrt( 3.0f*3.0f );

        MonteRay::Vector3D<gpuRayFloat_t> pos(0.0, 0.0, 0.0);
        MonteRay::Vector3D<gpuRayFloat_t> dir;

        gpuFloatType_t distance = estimator.getDistanceDirection( 0, pos, dir );

        CHECK_CLOSE( expectedDistance, distance, 1e-6 );
        CHECK_CLOSE( -1.0, dir[0], 1e-6 );
    }

    TEST( getDistanceDirection_PosV ) {
        MonteRayNextEventEstimator<GridBins> estimator(1);
        unsigned id = estimator.add( 0.0, 3.0, 0.0);
        gpuFloatType_t expectedDistance = std::sqrt( 3.0f*3.0f );

        MonteRay::Vector3D<gpuRayFloat_t> pos(0.0, 0.0, 0.0);
        MonteRay::Vector3D<gpuRayFloat_t> dir;

        gpuFloatType_t distance = estimator.getDistanceDirection( 0, pos, dir );

        CHECK_CLOSE( expectedDistance, distance, 1e-6 );
        CHECK_CLOSE( 1.0, dir[1], 1e-6 );
    }

    TEST( getDistanceDirection_NegV ) {
        MonteRayNextEventEstimator<GridBins> estimator(1);
        unsigned id = estimator.add( 0.0, -3.0, 0.0);
        gpuFloatType_t expectedDistance = std::sqrt( 3.0f*3.0f );

        MonteRay::Vector3D<gpuRayFloat_t> pos(0.0, 0.0, 0.0);
        MonteRay::Vector3D<gpuRayFloat_t> dir;

        gpuFloatType_t distance = estimator.getDistanceDirection( 0, pos, dir );

        CHECK_CLOSE( expectedDistance, distance, 1e-6 );
        CHECK_CLOSE( -1.0, dir[1], 1e-6 );
    }

    TEST( getDistanceDirection_PosW ) {
        MonteRayNextEventEstimator<GridBins> estimator(1);
        unsigned id = estimator.add( 0.0, 0.0, 3.0);
        gpuFloatType_t expectedDistance = std::sqrt( 3.0f*3.0f );

        MonteRay::Vector3D<gpuRayFloat_t> pos(0.0, 0.0, 0.0);
        MonteRay::Vector3D<gpuRayFloat_t> dir;

        gpuFloatType_t distance = estimator.getDistanceDirection( 0, pos, dir );

        CHECK_CLOSE( expectedDistance, distance, 1e-6 );
        CHECK_CLOSE( 1.0, dir[2], 1e-6 );
    }

    TEST( getDistanceDirection_NegW ) {
        MonteRayNextEventEstimator<GridBins> estimator(1);
        unsigned id = estimator.add( 0.0, 0.0, -3.0);
        gpuFloatType_t expectedDistance = std::sqrt( 3.0f*3.0f );

        MonteRay::Vector3D<gpuRayFloat_t> pos(0.0, 0.0, 0.0);
        MonteRay::Vector3D<gpuRayFloat_t> dir;

        gpuFloatType_t distance = estimator.getDistanceDirection( 0, pos, dir );

        CHECK_CLOSE( expectedDistance, distance, 1e-6 );
        CHECK_CLOSE( -1.0, dir[2], 1e-6 );
    }

    TEST( getDistanceDirection_PosUV ) {
        MonteRayNextEventEstimator<GridBins> estimator(1);
        unsigned id = estimator.add( 3.0, 3.0, 0.0);
        gpuFloatType_t expectedDistance = std::sqrt( (3.0f*3.0f)*2 );

        MonteRay::Vector3D<gpuRayFloat_t> pos(0.0, 0.0, 0.0);
        MonteRay::Vector3D<gpuRayFloat_t> dir;

        gpuFloatType_t distance = estimator.getDistanceDirection( 0, pos, dir );

        CHECK_CLOSE( expectedDistance, distance, 1e-6 );
        CHECK_CLOSE( 1.0/sqrt(2.0), dir[0], 1e-6 );
        CHECK_CLOSE( 1.0/sqrt(2.0), dir[1], 1e-6 );
    }
#endif

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

#ifndef MEMCHECK
    TEST_FIXTURE(CalcScore_test, calcScore_vacuum ) {
        CHECK_CLOSE( 1.0, pXS->getTotalXS( 0.5 ), 1e-6 );
        CHECK_CLOSE( 1.0, pMat->getTotalXS( 0.5 ), 1e-6 );

        unsigned id = pEstimator->add( 1.0, 0.0, 0.0);
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
        RayWorkInfo<> rayInfo(1,true);
        gpuFloatType_t score = pEstimator->calcScore<1>(particleID, ray, rayInfo );

        gpuFloatType_t expected = ( 1/ (4.0f * MonteRay::pi ) ) * exp(-0.0);
        CHECK_CLOSE( expected, score, 1e-6);
    }

    TEST_FIXTURE(CalcScore_test, calcScore_thru_material ) {
        CHECK_CLOSE( 1.0, pXS->getTotalXS( 0.5 ) , 1e-6 );
        CHECK_CLOSE( 1.0, pMat->getTotalXS( 0.5 ) , 1e-6 );

        unsigned id = pEstimator->add( 2.0, 0.0, 0.0);
        pEstimator->initialize();

        gpuFloatType_t x = 1.0;
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
        RayWorkInfo<> rayInfo(1,true);
        gpuFloatType_t score = pEstimator->calcScore<1>(particleID, ray, rayInfo );

        gpuFloatType_t expected = ( 1/ (4.0f * MonteRay::pi ) ) * exp(-1.0);
        CHECK_CLOSE( expected, score, 1e-6);
    }

    TEST_FIXTURE(CalcScore_test, calcScore_thru_vacuum_and_material ) {
        unsigned id = pEstimator->add( 2.0, 0.0, 0.0);
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
        RayWorkInfo<> rayInfo(1,true);
        gpuFloatType_t score = pEstimator->calcScore<1>(particleID, ray, rayInfo );

        gpuFloatType_t expected = ( 1/ (4.0f * MonteRay::pi * 2.0f*2.0f ) ) * exp(-1.0);
        CHECK_CLOSE( expected, score, 1e-6);
    }

    TEST_FIXTURE(CalcScore_test, getTally ) {
        unsigned id = pEstimator->add( 2.0, 0.0, 0.0);
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
        RayWorkInfo<> rayInfo(1,true);
        pEstimator->calcScore<1>(particleID, ray, rayInfo );

        gpuFloatType_t score = pEstimator->getTally(0,0);

        gpuFloatType_t expected = ( 1/ (4.0f * MonteRay::pi * 2.0f*2.0f ) ) * exp(-1.0);
        CHECK_CLOSE( expected, score, 1e-6);
    }

    TEST_FIXTURE(CalcScore_test, addTimeBins ) {
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
        RayWorkInfo<> rayInfo(1,true);
        pEstimator->calcScore<1>(particleID, ray, rayInfo );

        gpuFloatType_t expected1 = ( 1/ (4.0f * MonteRay::pi * distance1*distance1 ) ) * exp(-1.0);
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

        ray.detectorIndex = 1;
        rayInfo.clear();
        pEstimator->calcScore<1>(particleID, ray, rayInfo );

        gpuFloatType_t expected2 = ( 1/ (4.0f * MonteRay::pi * distance2*distance2 ) ) * exp(-1.0);

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
    }

    TEST_FIXTURE(CalcScore_test, calcScore_thru_material_3_probabilities ) {
        CHECK_CLOSE( 1.0, pXS->getTotalXS( 0.5 ) , 1e-6 );
        CHECK_CLOSE( 1.0, pMat->getTotalXS( 0.5 ) , 1e-6 );
        CHECK_CLOSE( 2.0, pXS->getTotalXS( 1.0 ) , 1e-6 );
        CHECK_CLOSE( 2.0, pMat->getTotalXS( 1.0 ) , 1e-6 );
        CHECK_CLOSE( 4.0, pXS->getTotalXS( 3.0 ) , 1e-6 );
        CHECK_CLOSE( 4.0, pMat->getTotalXS( 3.0 ) , 1e-6 );

        unsigned id = pEstimator->add( 2.0, 0.0, 0.0);
        pEstimator->initialize();

        const unsigned N = 3;

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
        ray.energy[0] = energy[0];
        ray.energy[1] = energy[1];
        ray.energy[2] = energy[2];
        ray.weight[0] = weight[0];
        ray.weight[1] = weight[1];
        ray.weight[2] = weight[2];
        ray.detectorIndex = 0;
        ray.particleType = photon;

        //std:: cout << "Debug: *************************\n";
        unsigned particleID = 0;
        RayWorkInfo<N> rayInfo(1,true);
        gpuFloatType_t score = pEstimator->calcScore<N>(particleID, ray, rayInfo );
        //std:: cout << "Debug: *************************\n";

        gpuFloatType_t expected1 = ( 0.3f / (2.0f * MonteRay::pi * 4.0f ) ) * exp( -1.0*1.0 );
        gpuFloatType_t expected2 = ( 1.0f / (2.0f * MonteRay::pi * 4.0f ) ) * exp( -1.0*2.0 );
        gpuFloatType_t expected3 = ( 2.0f / (2.0f * MonteRay::pi * 4.0f ) ) * exp( -1.0*4.0 );
        CHECK_CLOSE( 0.00439124, expected1, 1e-7);
        CHECK_CLOSE( 0.00538482, expected2, 1e-7);
        CHECK_CLOSE( 0.00145751, expected3, 1e-7);
        CHECK_CLOSE( expected1+expected2+expected3, score, 1e-7);

    }

    TEST_FIXTURE(CalcScore_test, calcScore_with_RayList ) {
        const unsigned N = 3;
        unsigned id = pEstimator->add( 2.0, 0.0, 0.0);
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
        pBank->add( ray );
        pBank->add( ray );

        RayWorkInfo<N> rayInfo(pBank->size(),true);

        //std:: cout << "Debug: **********calcScore_with_RayList***************\n";
        CHECK_CLOSE( 0.0, pEstimator->getTally(0), 1e-7);
        pEstimator->cpuScoreRayList(pBank.get(), &rayInfo);
        gpuTallyType_t value = pEstimator->getTally(0);
        //std:: cout << "Debug: ************************************************\n";

        gpuFloatType_t expected1 = ( 0.3f / (2.0f * MonteRay::pi * 4.0f ) ) * exp( -1.0*1.0 );
        gpuFloatType_t expected2 = ( 1.0f / (2.0f * MonteRay::pi * 4.0f ) ) * exp( -1.0*2.0 );
        gpuFloatType_t expected3 = ( 2.0f / (2.0f * MonteRay::pi * 4.0f ) ) * exp( -1.0*4.0 );

        CHECK_CLOSE( 2*(expected1+expected2+expected3), value, 1e-7);

    }
#endif

    TEST_FIXTURE(CalcScore_test, calc_with_rayList_on_GPU ) {

//        std::cout << "Debug: **********************\n";
//        std::cout << "Debug: MonteRayNextEventEstimator_unittest.cc -- TEST calc_with_rayList_on_GPU\n";
        const unsigned N = 3;
        unsigned id = pEstimator->add( 2.0, 0.0, 0.0);
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
        pBank->add( ray );
        pBank->add( ray );

#ifdef __CUDACC__
        cudaEvent_t start, stop;
        cudaEventCreate(&start);

        pBank->copyToGPU();
        pEstimator->copyToGPU();

        RayWorkInfo<N> rayInfo(1);
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
        pEstimator->launch_ScoreRayList(1,1,pBank.get());
#endif
        gpuTallyType_t value = pEstimator->getTally(0);

        gpuFloatType_t expected1 = ( 0.3f / (2.0f * MonteRay::pi * 4.0f ) ) * exp( -1.0*1.0 );
        gpuFloatType_t expected2 = ( 1.0f / (2.0f * MonteRay::pi * 4.0f ) ) * exp( -1.0*2.0 );
        gpuFloatType_t expected3 = ( 2.0f / (2.0f * MonteRay::pi * 4.0f ) ) * exp( -1.0*4.0 );

        CHECK_CLOSE( 2*(expected1+expected2+expected3), value, 1e-7);
//        std::cout << "Debug: **********************\n";
    }

    TEST_FIXTURE(CalcScore_test, rayListOnGPU_withTimeBins ) {

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
         pBank->add( ray );
         pBank->add( ray );

         ray.detectorIndex = 1;
         pBank->add( ray );
         pBank->add( ray );

 #ifdef __CUDACC__
         cudaEvent_t start, stop;
         cudaEventCreate(&start);

         pBank->copyToGPU();
         pEstimator->copyToGPU();

         RayWorkInfo<N> rayInfo(1);
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
         pEstimator->launch_ScoreRayList(1,1,pBank.get());
 #endif

         gpuFloatType_t expected1  = ( 0.3f / (2.0f * MonteRay::pi * distance1*distance1 ) ) * exp( -1.0*1.0 ) +
                                     ( 1.0f / (2.0f * MonteRay::pi * distance1*distance1 ) ) * exp( -1.0*2.0 ) +
                                     ( 2.0f / (2.0f * MonteRay::pi * distance1*distance1 ) ) * exp( -1.0*4.0 );

         gpuFloatType_t expected2  = ( 0.3f / (2.0f * MonteRay::pi * distance2*distance2 ) ) * exp( -1.0*1.0 ) +
                                     ( 1.0f / (2.0f * MonteRay::pi * distance2*distance2 ) ) * exp( -1.0*2.0 ) +
                                     ( 2.0f / (2.0f * MonteRay::pi * distance2*distance2 ) ) * exp( -1.0*4.0 );

         CHECK_CLOSE( 2*expected1, pEstimator->getTally(0,0), 1e-7);
         CHECK_CLOSE(         0.0, pEstimator->getTally(0,1), 1e-7);
         CHECK_CLOSE(         0.0, pEstimator->getTally(0,2), 1e-7);
         CHECK_CLOSE(         0.0, pEstimator->getTally(0,3), 1e-7);
         CHECK_CLOSE(         0.0, pEstimator->getTally(0,4), 1e-7);
         CHECK_CLOSE(         0.0, pEstimator->getTally(0,5), 1e-7);

         CHECK_CLOSE(         0.0, pEstimator->getTally(1,0), 1e-7);
         CHECK_CLOSE( 2*expected2, pEstimator->getTally(1,1), 1e-7);
         CHECK_CLOSE(         0.0, pEstimator->getTally(1,2), 1e-7);
         CHECK_CLOSE(         0.0, pEstimator->getTally(1,3), 1e-7);
         CHECK_CLOSE(         0.0, pEstimator->getTally(1,4), 1e-7);
         CHECK_CLOSE(         0.0, pEstimator->getTally(1,5), 1e-7);
 //        std::cout << "Debug: **********************\n";
     }

    TEST_FIXTURE(CalcScore_test, write ) {

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
         pBank->add( ray );
         pBank->add( ray );

         ray.detectorIndex = 1;
         pBank->add( ray );
         pBank->add( ray );

         // write out state of MonteRayNextEventEstimator class
         std::string filename = std::string("nee_write_state_test.bin");
         pEstimator->writeToFile( filename );

         // read
         {
             MonteRayNextEventEstimator<GridBins> estimator(0);

             // test file exists
             std::ifstream exists(filename.c_str());
             CHECK_EQUAL( true, exists.good() );
             exists.close();

             estimator.readFromFile( filename );

             CHECK_EQUAL( 2, estimator.size() );
             CHECK_EQUAL( 10, estimator.capacity() );

             CHECK_CLOSE( 2.0, estimator.getX(0), 1e-6 );
             CHECK_CLOSE( 0.0, estimator.getY(0), 1e-6 );
             CHECK_CLOSE( 0.0, estimator.getZ(0), 1e-6 );

             CHECK_CLOSE( 400.0, estimator.getX(1), 1e-6 );
             CHECK_CLOSE( 0.0, estimator.getY(1), 1e-6 );
             CHECK_CLOSE( 0.0, estimator.getZ(1), 1e-6 );

             std::vector<gpuFloatType_t> timeBins = estimator.getTimeBinEdges();
             CHECK_EQUAL( 5, timeBins.size() );
         }
    }

    TEST( run_leak_report ) {

#ifdef MEMCHECK
        std:: cout << "Debug: ********************************\n";
        std:: cout << "Debug: ****** Leak report *************\n";
        AllocationTracker::getInstance().reportLeakedMemory();

        cudaDeviceReset(); // enable leak checking.
        std:: cout << "Debug: ********************************\n";
#endif
    }


}

} // end namespace

